"""Write base.fbin as LIBSVM text for the SONG builder (label 0, 1-based feature ids).

Floats are printed with %.9g, which round-trips float32 exactly.
Usage: python datasets/to_libsvm.py <base.fbin> <out.libsvm>
"""
import sys

from binio import read_bin


def main() -> None:
    x = read_bin(sys.argv[1], mmap=True)
    ids = range(1, x.shape[1] + 1)
    with open(sys.argv[2], 'w') as f:
        for start in range(0, x.shape[0], 10000):
            f.write(''.join('0 ' + ' '.join(f'{j}:{v:.9g}' for j, v in zip(ids, row)) + '\n'
                            for row in x[start:start + 10000]))


if __name__ == '__main__':
    main()
