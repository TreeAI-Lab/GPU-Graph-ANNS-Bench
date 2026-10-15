// Modified copy of warp_astar_accelerator.h from SONG (github.com/sunbelbd/song, commit fa2d755).
// Copyright (c) 2021 SONG authors. MIT License, see the LICENSE file of SONG.
//
// Changes from SONG:
//   - the kernel takes max_iterations; the candidate queue and the result list hold max_iterations - 1 entries,
//     and a query stops when the result list is full or after max_iterations queue pops
//   - counters of distance computations (calculate_cnt) and visited vertices (path_len)
//   - SongResidentSearch (end of file): base vectors and graph stay in GPU memory, and each batch of queries is
//     timed as query H2D, kernel and result D2H with CUDA events
// Values below were filled in by SONG's fill_parameters.sh (TOPK 64, DIM 96); src/search/build.sh replaces
// DIM per binary.
#pragma once

#include"data.h"
#include<vector>
#include"config.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cublas_v2.h"
#include<chrono>

#include"smmh2.h"
#include"bin_heap.h"
#include"cuckoofilter.h"
#include"bloomfilter.h"
#include"blocked_bloomfilter.h"
#include"vanilla_list.h"
#include"fixhash.h"

#ifndef __ENABLE_BLOCKED_BLOOM_FILTER
	#define BlockedBloomFilter BloomFilter
#endif

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32
#define N_MULTIQUERY 1
#define CRITICAL_STEP (N_THREAD_IN_WARP/N_MULTIQUERY)
#define N_MULTIPROBE 1
#define FINISH_CNT 1

#define __ENABLE_MEASURE



struct Measure{
	unsigned long long stage1 = 0;
	unsigned long long stage2 = 0;
	unsigned long long stage3 = 0;
};

__global__
void warp_independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift
#ifdef __ENABLE_MEASURE
,Measure* measure
#endif
,unsigned long long* calculate_cnt, unsigned long long* path_len, int max_iterations){
	const int QUEUE_SIZE = max_iterations - 1;
	#define DIM 96
    int bid = blockIdx.x * N_MULTIQUERY;
	const int step = N_THREAD_IN_WARP;
    int tid = threadIdx.x;
	int cid = tid / CRITICAL_STEP;
	int subtid = tid % CRITICAL_STEP;
#define BLOOM_FILTER_BIT64 128
#define BLOOM_FILTER_BIT_SHIFT 7
#define BLOOM_FILTER_NUM_HASH 7

#ifndef __ENABLE_VISITED_DEL
#define HASH_TABLE_CAPACITY (TOPK*4*16)
#else
#define HASH_TABLE_CAPACITY (TOPK*4*2)
#endif

#ifdef __DISABLE_SELECT_INSERT
#undef HASH_TABLE_CAPACITY
#define HASH_TABLE_CAPACITY (TOPK*4*16+500)
#endif

    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    //BloomFilter<64,6,7>* pbf;
    //BloomFilter<64,6,3> bf;
    //VanillaList* pbf;
#ifdef __ENABLE_FIXHASH
    FixHash<int,HASH_TABLE_CAPACITY>* pbf;
#elif __ENABLE_CUCKOO_FILTER
	#define CUCKOO_CAPACITY (BLOOM_FILTER_BIT64 * 2)
	CuckooFilter<CUCKOO_CAPACITY>* pbf;
#else
    //BloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
    BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
#endif
    KernelPair<dist_t,idx_t>* q;
    KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(subtid == 0){
		dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
		q= new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 1];
    	//pbf = new BloomFilter<64,6,7>();
#ifdef __ENABLE_FIXHASH
		pbf = new FixHash<int,HASH_TABLE_CAPACITY>();
#elif __ENABLE_CUCKOO_FILTER
		pbf = new CuckooFilter<CUCKOO_CAPACITY>();
#else
    	//pbf = new BloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
    	pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
#endif
	//pbf = new VanillaList();
	}
    __shared__ int heap_size[N_MULTIQUERY];
	int topk_heap_size;

    __shared__ value_t query_point[N_MULTIQUERY][DIM];
	int iter;
	__shared__ int finished[N_MULTIQUERY];
	__shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
	__shared__ int index_list_len[N_MULTIQUERY];
	value_t start_distance;
	
	if(subtid == 0){
		iter = 0;
	}
	__syncthreads();
	value_t tmp[N_MULTIQUERY];
	#ifdef __USE_COS_DIST
	value_t tmp_data_len[N_MULTIQUERY];
	#endif
	for(int j = 0;j < N_MULTIQUERY;++j){
		tmp[j] = 0;
		#ifdef __USE_COS_DIST
		tmp_data_len[j] = 0;
		#endif
		for(int i = tid;i < DIM;i += step){
			query_point[j][i] = d_query[(bid + j) * DIM + i];
			#ifdef __USE_L2_DIST
			tmp[j] += (query_point[j][i] - d_data[i]) * (query_point[j][i] - d_data[i]); 
			#elif __USE_IP_DIST
			tmp[j] += query_point[j][i] * d_data[i]; 
			#elif __USE_COS_DIST
			//negative cosine
			tmp[j] += query_point[j][i] * d_data[i]; 
			tmp_data_len[j] += d_data[i] * d_data[i];
			#else
			#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
			#endif
		}
		for (int offset = 16; offset > 0; offset /= 2){
				#ifdef __USE_L2_DIST
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				#elif __USE_IP_DIST
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				#elif __USE_COS_DIST
				//negative cosine
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				tmp_data_len[j] += __shfl_xor_sync(FULL_MASK, tmp_data_len[j], offset);
				#else
				#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
				#endif
		}
	}
	if(subtid == 0){
		#ifdef __USE_L2_DIST
		start_distance = tmp[cid];
		#elif __USE_IP_DIST
		start_distance = -tmp[cid];
		#elif __USE_COS_DIST
		//negative cosine
       	int sign = tmp[cid] < 0 ? 1 : -1;
		if(tmp_data_len[cid] != 0)
			start_distance = sign * tmp[cid] * tmp[cid] / tmp_data_len[cid];
		else
			start_distance = 0;
		#else
		#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
		#endif
	}
	__syncthreads();
	
	if(subtid == 0){
    	heap_size[cid] = 1;
		topk_heap_size = 0;
		finished[cid] = false;
		dist_t d = start_distance;
		KernelPair<dist_t,idx_t> kp;
		kp.first = d;
		kp.second = 0;
		smmh2::insert(q,heap_size[cid],kp);
		pbf->add(0);
	}
	__syncthreads();
    while(heap_size[cid] > 1){
#ifdef __ENABLE_MEASURE
		auto stage1_start = clock64();
#endif
		index_list_len[cid] = 0;
		int current_heap_elements = heap_size[cid] - 1;
		for(int k = 0;k < N_MULTIPROBE && k < current_heap_elements;++k){
			KernelPair<dist_t,idx_t> now;
			if(subtid == 0){
				now = smmh2::pop_min(q,heap_size[cid]);
				atomicAdd(&path_len[0],1ULL);
				iter+=1;
#ifdef __ENABLE_VISITED_DEL
				pbf->del(now.second);
#endif
				// SONG also required topk[0].first <= now.first here; the query now ends when the result list is
				// full or after max_iterations pops
				if((k == 0 && topk_heap_size == QUEUE_SIZE)|| iter >= max_iterations){
					++finished[cid];
					iter = 0;
				}
			}
			__syncthreads();
			if(finished[cid] >= FINISH_CNT)
				break;
			if(subtid == 0){
				topk[topk_heap_size++] = now;
				push_heap(topk,topk + topk_heap_size);
#ifdef __ENABLE_VISITED_DEL
				pbf->add(now.second);
#endif
				if(topk_heap_size > QUEUE_SIZE){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(topk[0].second);
#endif
					pop_heap(topk,topk + topk_heap_size);
					--topk_heap_size;
				}
				auto offset = now.second << vertex_offset_shift;
				int degree = d_graph[offset];
				for(int i = 1;i <= degree;++i){
					auto idx = d_graph[offset + i];
					if(subtid == 0){
						if(pbf->test(idx)){
							continue;
						}
#ifdef __DISABLE_SELECT_INSERT
						pbf->add(idx);
#endif
						index_list[cid][index_list_len[cid]++] = idx;
					}
				}
			}
		}
		if(finished[cid] >= FINISH_CNT)
			break;
		__syncthreads();

#ifdef __ENABLE_MEASURE
		auto stage1_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage1,stage1_end - stage1_start);	
		auto stage2_start = clock64();
#endif
		for(int nq = 0;nq < N_MULTIQUERY;++nq){
			for(int i = 0;i < index_list_len[nq];++i){
				//TODO: replace this atomic with reduction in CUB
				value_t tmp = 0;
				#ifdef __USE_COS_DIST
				value_t tmp_data_len = 0;
				#endif
				for(int j = tid;j < DIM;j += step){
					#ifdef __USE_L2_DIST
					tmp += (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]) * (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]); 
					#elif __USE_IP_DIST
					tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
					#elif __USE_COS_DIST
					//negative cosine
					tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
					tmp_data_len += d_data[index_list[nq][i] * DIM + j] * d_data[index_list[nq][i] * DIM + j]; 
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
				for (int offset = 16; offset > 0; offset /= 2){
					#ifdef __USE_L2_DIST
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					#elif __USE_IP_DIST
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					#elif __USE_COS_DIST
					//negative cosine
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					tmp_data_len += __shfl_xor_sync(FULL_MASK, tmp_data_len, offset);
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
				if(tid == nq * CRITICAL_STEP){
					atomicAdd(&calculate_cnt[0],1ULL);
					#ifdef __USE_L2_DIST
					dist_list[i] = tmp;
					#elif __USE_IP_DIST
					dist_list[i] = -tmp;
					#elif __USE_COS_DIST
					//negative cosine
					int sign = tmp < 0 ? 1 : -1;
					if(tmp_data_len != 0)
						dist_list[i] = sign * tmp * tmp / tmp_data_len;
					else
						dist_list[i] = 0;
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
			}
		}

		__syncthreads();
#ifdef __ENABLE_MEASURE
		auto stage2_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage2,stage2_end - stage2_start);	
		auto stage3_start = clock64();
#endif

		if(subtid == 0){
			for(int i = 0;i < index_list_len[cid];++i){
				dist_t d = dist_list[i];
				KernelPair<dist_t,idx_t> kp;
				kp.first = d;
				kp.second = index_list[cid][i];

				if(heap_size[cid] >= QUEUE_SIZE + 1 && q[2].first < kp.first){
					continue;
				}
#ifdef __ENABLE_MULTIPROBE_DOUBLE_CHECK
				if(pbf->test(kp.second))
					continue;
#endif
				smmh2::insert(q,heap_size[cid],kp);
#ifndef __DISABLE_SELECT_INSERT
				pbf->add(kp.second);
#endif
				if(heap_size[cid] >= QUEUE_SIZE + 2){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(q[2].second);
#endif
					smmh2::pop_max(q,heap_size[cid]);
				}
			}
		}
		__syncthreads();
#ifdef __ENABLE_MEASURE
		auto stage3_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage3,stage3_end - stage3_start);	
#endif
    }

	if(subtid == 0){
		for(int i = 0;i < QUEUE_SIZE;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[(bid + cid) * QUEUE_SIZE + QUEUE_SIZE - 1 - i] = now.second;
		}
		delete[] q;
		delete[] topk;
    	delete pbf;
    	delete[] dist_list;
	}
}

class WarpAStarAccelerator{
private:

public:
    static void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,value_t>>>& queries,int k,std::vector<std::vector<idx_t>>& results,value_t* h_data,idx_t* h_graph,int vertex_offset_shift,int num,int dim, unsigned long long& calculate_count, unsigned long long& path_length, int max_iterations){
        value_t* d_data;
		value_t* d_query;
		unsigned long long* calculate_cnt;
		unsigned long long* path_len;
		idx_t* d_result;
		idx_t* d_graph;
		int new_size = max_iterations - 1;
		cudaMalloc(&d_data,sizeof(value_t) * num * dim);
		cudaMalloc(&d_graph,sizeof(idx_t) * (num << vertex_offset_shift));
		cudaMalloc(&calculate_cnt,sizeof(unsigned long long));
		cudaMalloc(&path_len,sizeof(unsigned long long));
		

#ifdef __ENABLE_MEASURE
		Measure* d_measure;
		Measure h_measure;
		cudaMalloc(&d_measure,sizeof(Measure));
		cudaMemcpy(d_measure,&h_measure,sizeof(Measure),cudaMemcpyHostToDevice);
#endif
		std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
				memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
				for(int i = 0;i < queries.size();++i){
					for(auto p : queries[i]){
						*(h_query.get() + i * dim + p.first) = p.second;
					}
				}
				std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * new_size]);
		cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * new_size);
		auto time_begin = std::chrono::steady_clock::now();
		cudaMemcpy(path_len,&path_length,sizeof(unsigned long long),cudaMemcpyHostToDevice);
		cudaMemcpy(calculate_cnt,&calculate_count,sizeof(unsigned long long),cudaMemcpyHostToDevice);
		cudaMemcpy(d_data,h_data,sizeof(value_t) * num * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);
		
		
		
		cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);
		

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point mem_transfer = std::chrono::steady_clock::now();
		fprintf(stderr,"mem transfer %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(mem_transfer - time_begin).count());
		std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();
#endif

		warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,32>>>(d_data,d_query,d_result,d_graph,queries.size(),vertex_offset_shift
#ifdef __ENABLE_MEASURE
, d_measure
#endif
		,calculate_cnt,path_len,max_iterations);

#ifdef __ENABLE_MEASURE
		cudaDeviceSynchronize();
		std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
		fprintf(stderr,"kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());
		std::chrono::steady_clock::time_point back_begin = std::chrono::steady_clock::now();
#endif
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * new_size,cudaMemcpyDeviceToHost);
		cudaMemcpy(&calculate_count, calculate_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		cudaMemcpy(&path_length, path_len, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point back_end = std::chrono::steady_clock::now();
		fprintf(stderr,"transfer back result takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(back_end - back_begin).count());

		cudaMemcpy(&h_measure,d_measure,sizeof(Measure),cudaMemcpyDeviceToHost);
		auto stage_sum = h_measure.stage1 + h_measure.stage2 + h_measure.stage3;
		fprintf(stderr,"stages percentage %.2f %.2f %.2f\n", h_measure.stage1 * 100.0 / stage_sum,
			h_measure.stage2 * 100.0 / stage_sum,h_measure.stage3 * 100.0 / stage_sum);
#endif
		results.clear();
		for(int i = 0;i < queries.size();++i){
			std::vector<idx_t> v(new_size);
			for(int j = 0;j < new_size;++j)
				v[j] = h_result[i * new_size + j];
			results.push_back(v);
		}
		std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		fprintf(stderr,"using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
		//printf("using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
		cudaFree(d_data);
		cudaFree(d_query);
		cudaFree(path_len);
		cudaFree(d_result);
		cudaFree(d_graph);
		cudaFree(calculate_cnt);
    }
};


// Base vectors and graph stay in GPU memory; each batch is timed as query H2D, kernel and result D2H.
// The kernel above runs with max_iterations (queue length max_iterations - 1).
struct SongResidentSearch {
	value_t* d_data = nullptr;
	idx_t* d_graph = nullptr;
	value_t* d_query = nullptr;
	idx_t* d_result = nullptr;
	unsigned long long* d_calc = nullptr;
	unsigned long long* d_path = nullptr;
#ifdef __ENABLE_MEASURE
	Measure* d_measure = nullptr;
#endif
	int num = 0;
	int dim = 0;
	int vertex_offset_shift = 0;
	int max_batch = 0;
	int alloc_iterations = 55;
	int max_iterations = 55;
	int result_width = 54; // max_iterations - 1

	void set_iterations(int it) {
		if (it < 2 || it > alloc_iterations) {
			fprintf(stderr, "set_iterations %d not in [2,%d]\n", it, alloc_iterations);
			exit(1);
		}
		max_iterations = it;
		result_width = it - 1;
	}

	void init(value_t* h_data, idx_t* h_graph, int vertex_offset_shift_, int num_, int dim_,
	          int max_batch_, int max_iterations_) {
		num = num_;
		dim = dim_;
		vertex_offset_shift = vertex_offset_shift_;
		max_batch = max_batch_;
		alloc_iterations = max_iterations_;
		max_iterations = max_iterations_;
		result_width = max_iterations - 1;
		cudaMalloc(&d_data, sizeof(value_t) * (size_t)num * dim);
		cudaMalloc(&d_graph, sizeof(idx_t) * ((size_t)num << vertex_offset_shift));
		cudaMemcpy(d_data, h_data, sizeof(value_t) * (size_t)num * dim, cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph, h_graph, sizeof(idx_t) * ((size_t)num << vertex_offset_shift), cudaMemcpyHostToDevice);
		cudaMalloc(&d_query, sizeof(value_t) * (size_t)max_batch * dim);
		cudaMalloc(&d_result, sizeof(idx_t) * (size_t)max_batch * result_width);
		cudaMalloc(&d_calc, sizeof(unsigned long long));
		cudaMalloc(&d_path, sizeof(unsigned long long));
#ifdef __ENABLE_MEASURE
		cudaMalloc(&d_measure, sizeof(Measure));
#endif
	}

	void search_batch(const value_t* h_query, int nq, idx_t* h_result,
	                  cudaEvent_t e_h2d0, cudaEvent_t e_h2d1,
	                  cudaEvent_t e_k0, cudaEvent_t e_k1,
	                  cudaEvent_t e_d2h0, cudaEvent_t e_d2h1,
	                  float* h2d_ms, float* kernel_ms, float* d2h_ms,
	                  int kernel_chunk = 0) {
		(void)kernel_chunk; // one launch for the whole batch: smaller launches leave the GPU underused
		if (nq > max_batch) { fprintf(stderr, "nq %d > max_batch %d\n", nq, max_batch); exit(1); }
		unsigned long long zero = 0;
		cudaMemcpy(d_calc, &zero, sizeof(zero), cudaMemcpyHostToDevice);
		cudaMemcpy(d_path, &zero, sizeof(zero), cudaMemcpyHostToDevice);
#ifdef __ENABLE_MEASURE
		Measure hm{};
		cudaMemcpy(d_measure, &hm, sizeof(Measure), cudaMemcpyHostToDevice);
#endif
		cudaEventRecord(e_h2d0);
		cudaMemcpy(d_query, h_query, sizeof(value_t) * (size_t)nq * dim, cudaMemcpyHostToDevice);
		cudaEventRecord(e_h2d1);
		cudaEventSynchronize(e_h2d1);
		cudaEventElapsedTime(h2d_ms, e_h2d0, e_h2d1);

		cudaEventRecord(e_k0);
		warp_independent_search_kernel<<<nq / N_MULTIQUERY, 32>>>(
			d_data, d_query, d_result, d_graph, nq, vertex_offset_shift
#ifdef __ENABLE_MEASURE
			, d_measure
#endif
			, d_calc, d_path, max_iterations);
		cudaEventRecord(e_k1);
		cudaEventSynchronize(e_k1);
		cudaEventElapsedTime(kernel_ms, e_k0, e_k1);

		cudaEventRecord(e_d2h0);
		cudaMemcpy(h_result, d_result, sizeof(idx_t) * (size_t)nq * result_width, cudaMemcpyDeviceToHost);
		cudaEventRecord(e_d2h1);
		cudaEventSynchronize(e_d2h1);
		cudaEventElapsedTime(d2h_ms, e_d2h0, e_d2h1);
	}

	void cleanup() {
		if (d_data) cudaFree(d_data);
		if (d_graph) cudaFree(d_graph);
		if (d_query) cudaFree(d_query);
		if (d_result) cudaFree(d_result);
		if (d_calc) cudaFree(d_calc);
		if (d_path) cudaFree(d_path);
#ifdef __ENABLE_MEASURE
		if (d_measure) cudaFree(d_measure);
#endif
		d_data = nullptr; d_graph = nullptr; d_query = nullptr;
		d_result = nullptr; d_calc = d_path = nullptr;
	}
};
