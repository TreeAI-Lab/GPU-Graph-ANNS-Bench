# Datasets

```bash
export DATA_DIR=/path/to/data
bash datasets/download.sh sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
python datasets/prepare.py sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M --gpu
```

`download.sh` puts the raw files in `$DATA_DIR/raw`. `prepare.py` writes one directory per dataset:

| File | Content |
|---|---|
| `base.fbin`, `query.fbin` | int32 n, int32 d, then n x d float32 |
| `base.fvecs`, `query.fvecs` | the same vectors in fvecs, for builders that read only fvecs |
| `gt.ibin` | int32 n, int32 100, then the ids of the 100 nearest base vectors of every query (L2) |

## Datasets

| Name | Base | Queries | Dim | Source | Processing |
|---|---|---|---|---|---|
| `sift1M` | 1,000,000 | 10,000 | 128 | ann-benchmarks `sift-128-euclidean` | none |
| `gist` | 1,000,000 | 1,000 | 960 | ann-benchmarks `gist-960-euclidean` | none |
| `glove200` | 1,183,514 | 10,000 | 200 | ann-benchmarks `glove-200-angular` | L2-normalized |
| `nytimes` | 289,761 | 9,991 | 256 | ann-benchmarks `nytimes-256-angular` | all-zero vectors removed, L2-normalized, ground truth recomputed |
| `deep1M` | 1,000,000 | 10,000 | 96 | first 1M vectors of Deep1B, official queries | ground truth computed |
| `deep10M` | 10,000,000 | 10,000 | 96 | first 10M vectors of Deep1B, official queries | ground truth computed |
| `deep100M` | 100,000,000 | 10,000 | 96 | first 100M vectors of Deep1B, official queries | ground truth computed |
| `mnist8M` | 8,090,000 | 10,000 | 784 | LIBSVM `mnist8m` (as `MNIST8M_data.h5` on Hugging Face) | pixel values / 255; the last 10,000 vectors are the queries |

GloVe200 and NYTimes are angular datasets. On unit vectors the L2 distance ranks neighbors as cosine does, so they
are normalized and searched with L2. Cosine is undefined for all-zero vectors, which NYTimes contains (239 base
vectors and 9 queries); they are removed.

MNIST8M holds 135 versions (the original and its deformations) of each of the 60,000 MNIST training digits, so the
queries, taken from the same sequence, have near-copies in the base set.

For DEEP100M the first 100M vectors of `base.1B.fbin` have the md5 `d600092b4bcd6df3cf174099558a0132` after the
8-byte header.

## Variants

Some experiments ran on the following variants. They are listed per figure in the top-level README.

| Name | Base | Queries | Source | Used by |
|---|---|---|---|---|
| `deep10M-annb` | 9,990,000 | 10,000 | ann-benchmarks `deep-image-96-angular`, its queries | Figures 6, 7 and 10 on DEEP10M |
| `deep1M-annb` | 1,000,000 | 10,000 | first 1M vectors of `deep10M-annb`, its queries | IVF-PQ and brute force on DEEP1M in Figure 7 |
| `nytimes-full` | 290,000 | 10,000 | ann-benchmarks `nytimes-256-angular`, raw vectors, original ground truth, cosine | IVF-PQ and brute force on NYTimes in Figure 7 |

Deep1B vectors and the `deep-image-96` vectors are unit vectors, so L2 and cosine rank neighbors alike.

## Other scripts

| Script | Use |
|---|---|
| `groundtruth.py <dir> [--gpu]` | exact top-100 with Faiss, chunked for base sets larger than GPU memory |
| `sample_deep1b_queries.py` | 5,000,000 queries for the serving experiment on DEEP100M (Figure 6): the official queries followed by Deep1B vectors beyond the first 100M |
| `to_libsvm.py <base.fbin> <out>` | LIBSVM text input for the SONG builder |
| `binio.py` | readers and writers for fbin, ibin, fvecs and ivecs |
