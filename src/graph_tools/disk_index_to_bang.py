"""Convert a DiskANN disk index <prefix>_disk.index into the BANG input files <prefix>_disk.bin and
<prefix>_disk_metadata.bin.

The output format is that of bang_preprocess.py in the BANG repository (byte-identical output on DEEP1M against the
version of commit f1ec2a35): each node is its vector, its degree d (uint32), its first d neighbors in ascending
order, and the remaining DEGREE - d slots copied unchanged; the metadata is medoid (uint64), max_node_len (uint64),
DATATYPE, DIM, DEGREE and the number of nodes (uint32 each).

The only difference from bang_preprocess.py: that script reads nnodes_per_sector nodes from each 4096-byte sector,
which is 0 when a node is larger than one sector (GIST with R = 128: 960 x 4 + 4 + 128 x 4 = 4356 bytes), so it
reads no node at all. This converter locates the nodes as DiskANN's get_node_sector (pq_flash_index.cpp) does:
if nnodes_per_sector > 0, node i is in sector 1 + i / nnodes_per_sector; otherwise it starts at sector
1 + i * ceil(max_node_len / 4096).

usage: python disk_index_to_bang.py <prefix_disk.index> <prefix_disk.bin> <DIM> <DATATYPE: 0 int8, 1 uint8, 2 float> <DEGREE>
"""
import argparse
import sys

import numpy as np

SECTOR = 4096
CHUNK = 200_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='DiskANN <prefix>_disk.index')
    ap.add_argument('dst', help='output <prefix>_disk.bin; the metadata goes to <prefix>_disk_metadata.bin')
    ap.add_argument('dim', type=int)
    ap.add_argument('dtype_code', type=int, choices=(0, 1, 2), help='0 int8, 1 uint8, 2 float')
    ap.add_argument('degree', type=int, help='R of the DiskANN index')
    args = ap.parse_args()
    src, dst, dim, dtype_code, degree = args.src, args.dst, args.dim, args.dtype_code, args.degree

    ts = 4 if dtype_code == 2 else 1
    meta = np.fromfile(src, dtype='<u8', count=9, offset=8)
    total, num_dim, medoid, mnl, nps, filesize = int(meta[0]), int(meta[1]), meta[2], meta[3], int(meta[4]), int(meta[8])
    mnl_i = int(mnl)
    vb = dim * ts
    assert num_dim == dim, (num_dim, dim)
    assert mnl_i == vb + 4 + 4 * degree, (mnl_i, vb, degree)
    mm = np.memmap(src, dtype=np.uint8, mode='r')
    spn = -(-mnl_i // SECTOR)
    print(f'nodes={total} dim={dim} medoid={int(medoid)} max_node_len={mnl_i} nnodes_per_sector={nps} '
          f'sectors_per_node={spn if nps == 0 else 0} filesize={filesize}')
    with open(dst, 'wb') as w:
        for s in range(0, total, CHUNK):
            e = min(total, s + CHUNK)
            if nps > 0:
                s0, s1 = s // nps, (e - 1) // nps + 1
                blk = np.asarray(mm[(1 + s0) * SECTOR:(1 + s1) * SECTOR]).reshape(s1 - s0, SECTOR)[:, :nps * mnl_i]
                rows = blk.reshape((s1 - s0) * nps, mnl_i)[s - s0 * nps:e - s0 * nps]
            else:
                rows = np.asarray(mm[(1 + s * spn) * SECTOR:(1 + e * spn) * SECTOR]).reshape(e - s, spn * SECTOR)[:, :mnl_i]
            rows = np.ascontiguousarray(rows)
            deg = rows[:, vb:vb + 4].copy().view('<u4')[:, 0]
            if np.any(deg == 0) or np.any(deg > degree):
                bad = int(np.flatnonzero((deg == 0) | (deg > degree))[0]) + s
                sys.exit(f'node {bad} has degree {int(deg[bad - s])}, bang_preprocess.py would stop here')
            nb = rows[:, vb + 4:].copy().view('<u4')
            valid = np.arange(degree)[None, :] < deg[:, None]
            key = np.where(valid, nb.astype(np.int64), np.int64(1) << 40)
            order = np.argsort(key, axis=1, kind='stable')
            sorted_nb = np.take_along_axis(nb, order, axis=1)
            out_nb = np.where(valid, sorted_nb, nb)
            rows[:, vb + 4:] = out_nb.view(np.uint8)
            w.write(rows.tobytes())
    md = dst[:-4] + '_metadata' + dst[-4:]
    with open(md, 'wb') as w1:
        np.array([medoid, mnl], dtype='<u8').tofile(w1)
        np.array([dtype_code, dim, degree, total], dtype='<u4').tofile(w1)
    print(f'wrote {dst} and {md}, nodes={total}')


if __name__ == '__main__':
    main()
