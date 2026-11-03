#pragma once
/*
 * Recall@r as used throughout the paper: the number of the first r returned ids of a query that are among the r
 * true nearest neighbors (the first r ids of its ground-truth row), divided by r and averaged over the queries.
 * When fewer than r ids are returned (k_search < r), the missing ones count as misses.
 */
#include <algorithm>
#include <cstddef>

namespace dsio {

// ids: nq rows of k_search returned ids; gt: nq rows of gt_k true neighbor ids, nearest first.
template <typename IdT, typename GtT>
inline double recall_at_k(const IdT* ids, size_t k_search, const GtT* gt, size_t gt_k, size_t nq, size_t r) {
    if (r == 0 || nq == 0) {
        return 0.0;
    }
    const size_t n_ret = std::min(k_search, r);
    const size_t n_gt = std::min(gt_k, r);
    double sum = 0.0;
    for (size_t i = 0; i < nq; i++) {
        size_t hits = 0;
        for (size_t a = 0; a < n_ret; a++) {
            const long long id = static_cast<long long>(ids[i * k_search + a]);
            for (size_t b = 0; b < n_gt; b++) {
                if (id == static_cast<long long>(gt[i * gt_k + b])) {
                    hits++;
                    break;
                }
            }
        }
        sum += static_cast<double>(hits) / static_cast<double>(r);
    }
    return sum / static_cast<double>(nq);
}

}  // namespace dsio
