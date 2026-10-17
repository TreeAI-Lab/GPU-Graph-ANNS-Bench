# CAGRA tools against the cuVS C++ library built from source by third_party/build/cuvs.sh.
# Run it through src/search/build.sh, which sources env.sh for W, BIN_DIR and ENV_CUVS.
#
#   cagra_resident_search  search sweep or batched search on a graph-only index (dataset attached in-process)
#   cagra_build_ivfpq      graph-only CAGRA index built with the IVF-PQ graph builder (DEEP100M)
#   make_graph_to_cagra    graph-only CAGRA index from a binary adjacency list
#   extract_graph          binary adjacency list from a CAGRA index
#   attach_dataset         CAGRA index with the base vectors attached
#
# The RAPIDS headers (RAFT, RMM, CCCL, ...) and the logger sources come from the cuVS build tree.
# common.cuh is the unchanged example header of cuVS (examples/cpp/src).

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ENV_CUVS ?= $(W)/envs/cuvs-build
CUVS_SRC ?= $(W)/src/cuvs
# Layout of third_party/build/cuvs.sh: CMake build tree $(W)/build/cuvs/build with the dependency sources in its
# _deps/, install prefix $(W)/build/cuvs/install (the same variables as src/baselines/Makefile).
CUVS_INSTALL ?= $(W)/build/cuvs/install
CUVS_BUILD_DIR ?= $(W)/build/cuvs/build
CUVS_DEPS ?= $(CUVS_BUILD_DIR)/_deps
RAFT_INCLUDE ?= $(CUVS_DEPS)/raft-src/cpp/include
CUTLASS_INCLUDE ?= $(CUVS_DEPS)/nvidiacutlass-src/include
OBJ_DIR ?= $(W)/build/search/cagra_obj

NVCC := $(ENV_CUVS)/bin/nvcc
HOST_CXX ?= $(ENV_CUVS)/bin/x86_64-conda-linux-gnu-g++

# A800 = sm_80
ARCH ?= 80

NVCCFLAGS := -std=c++17 -O3 -DNDEBUG \
	-arch=sm_$(ARCH) \
	--expt-extended-lambda \
	--expt-relaxed-constexpr \
	--compiler-options -fPIC,-fopenmp,-Wno-deprecated-declarations

INCLUDES := \
	-I$(CUVS_INSTALL)/include \
	-I$(ENV_CUVS)/include \
	-I$(ENV_CUVS)/targets/x86_64-linux/include \
	-I$(RAFT_INCLUDE) \
	-I$(CUVS_DEPS)/raft-build/include \
	-I$(CUVS_DEPS)/rmm-src/include \
	-I$(CUVS_DEPS)/rmm-build/include \
	-I$(CUVS_DEPS)/cccl-src/thrust \
	-I$(CUVS_DEPS)/cccl-src/libcudacxx/include \
	-I$(CUVS_DEPS)/cccl-src/cub \
	-I$(CUVS_DEPS)/fmt-src/include \
	-I$(CUVS_DEPS)/spdlog-src/include \
	-I$(CUVS_DEPS)/nvtx3-src/c/include \
	-I$(CUVS_DEPS)/cuco-src/include \
	-I$(CUTLASS_INCLUDE) \
	-I$(CUVS_DEPS)/nvidiacutlass-build/include \
	-I$(CUVS_SRC)/examples/cpp/src \
	-I$(HERE)

DEFINES := \
	-DFMT_HEADER_ONLY=1 \
	-DSPDLOG_FMT_EXTERNAL \
	-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
	-DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
	-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA \
	-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
	-DTHRUST_DISABLE_ABI_NAMESPACE \
	-DTHRUST_IGNORE_ABI_NAMESPACE_ERROR

LIBDIRS := $(CUVS_INSTALL)/lib $(ENV_CUVS)/lib $(ENV_CUVS)/targets/x86_64-linux/lib

LDFLAGS := $(addprefix -L,$(LIBDIRS)) \
	$(foreach d,$(LIBDIRS),-Xlinker -rpath -Xlinker $(d) -Xlinker -rpath-link -Xlinker $(d))

LIBS := -lcuvs -lcudart -lcublas -lcusolver -lcusparse -lcurand -lnvJitLink \
	-lgomp -ldl -lpthread -lm

LOGGER_SRCS := \
	$(CUVS_DEPS)/rmm-build/include/rmm/logger_impl/logger.cpp \
	$(CUVS_DEPS)/raft-build/include/raft/core/logger_impl/logger.cpp

TOOLS := cagra_resident_search cagra_build_ivfpq make_graph_to_cagra extract_graph attach_dataset

.PHONY: all clean $(TOOLS)

all: $(TOOLS)

$(TOOLS): %: $(BIN_DIR)/%

$(BIN_DIR)/cagra_build_ivfpq: $(HERE)cagra_build_ivfpq.cu $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $< $(LOGGER_SRCS) -o $@ $(LDFLAGS) $(LIBS)

$(BIN_DIR)/make_graph_to_cagra: $(HERE)make_graph_to_cagra.cu $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $< $(LOGGER_SRCS) -o $@ $(LDFLAGS) $(LIBS)

$(BIN_DIR)/extract_graph: $(HERE)extract_graph.cu $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $< $(LOGGER_SRCS) -o $@ $(LDFLAGS) $(LIBS)

$(BIN_DIR)/attach_dataset: $(HERE)attach_dataset.cu $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $< $(LOGGER_SRCS) -o $@ $(LDFLAGS) $(LIBS)

$(BIN_DIR)/cagra_resident_search: $(HERE)cagra_resident_search.cu $(LOGGER_SRCS)
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $(OBJ_DIR)/cagra_resident_search.o
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) \
		-c $(CUVS_DEPS)/rmm-build/include/rmm/logger_impl/logger.cpp -o $(OBJ_DIR)/rmm_logger.o
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) \
		-c $(CUVS_DEPS)/raft-build/include/raft/core/logger_impl/logger.cpp -o $(OBJ_DIR)/raft_logger.o
	$(HOST_CXX) -fopenmp -o $@ \
		$(OBJ_DIR)/cagra_resident_search.o $(OBJ_DIR)/rmm_logger.o $(OBJ_DIR)/raft_logger.o \
		$(addprefix -L,$(LIBDIRS)) $(foreach d,$(LIBDIRS),-Wl,-rpath,$(d) -Wl,-rpath-link,$(d)) $(LIBS)

clean:
	rm -f $(addprefix $(BIN_DIR)/,$(TOOLS))
	rm -rf $(OBJ_DIR)
