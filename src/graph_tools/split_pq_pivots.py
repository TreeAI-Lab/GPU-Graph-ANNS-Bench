"""Split the _pq_pivots.bin of current DiskANN into the three files that BANG reads.

The current file starts with a bin block (int32 4, int32 1, then 4 uint64 offsets). At the offsets follow the
pivots block (int32 256, int32 D, 256 x D float), the centroid block (int32 D, int32 1, D float) and the chunk
offsets block (int32 n+1, int32 1, n+1 uint32). BANG reads 256 x D floats from byte 8 of _pq_pivots.bin and reads
_pq_pivots.bin_centroid.bin and _pq_pivots.bin_chunk_offsets.bin separately, so each block is written unchanged
to its own file. The input file is not modified; the output goes under another prefix.

usage: python split_pq_pivots.py <DiskANN _pq_pivots.bin> <output prefix>
  writes <prefix>_pq_pivots.bin, <prefix>_pq_pivots.bin_centroid.bin and <prefix>_pq_pivots.bin_chunk_offsets.bin
"""
import argparse
import struct
import sys


def read_block(buf: bytes, off: int, itemsize: int) -> bytes:
    """Return the bin block that starts at off, with its 8-byte header."""
    rows, cols = struct.unpack_from('<ii', buf, off)
    return buf[off:off + 8 + rows * cols * itemsize]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='_pq_pivots.bin written by DiskANN build_disk_index')
    ap.add_argument('prefix', help='output prefix, for example <dir>/bang_format/deep1M')
    args = ap.parse_args()
    src, prefix = args.src, args.prefix
    with open(src, 'rb') as f:
        buf = f.read()
    n, one = struct.unpack_from('<ii', buf, 0)
    if one != 1 or n < 4:
        sys.exit(f'{src}: not the multi-block DiskANN format (header {n} x {one})')
    offs = struct.unpack_from(f'<{n}Q', buf, 8)
    pivots = read_block(buf, offs[0], 4)
    centroid = read_block(buf, offs[1], 4)
    chunks = read_block(buf, offs[2], 4)
    if offs[3] > len(buf):
        sys.exit(f'{src}: last offset {offs[3]} beyond file size {len(buf)}')
    out = {'_pq_pivots.bin': pivots, '_pq_pivots.bin_centroid.bin': centroid,
           '_pq_pivots.bin_chunk_offsets.bin': chunks}
    for suffix, data in out.items():
        with open(prefix + suffix, 'wb') as f:
            f.write(data)
        r, c = struct.unpack_from('<ii', data, 0)
        print(f'{prefix + suffix}: {r} x {c}, {len(data)} bytes')


if __name__ == '__main__':
    main()
