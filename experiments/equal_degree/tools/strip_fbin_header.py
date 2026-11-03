"""Write the vectors of an fbin file without its 8-byte header (int32 n, int32 d).

The SONG search program loads the base vectors as raw float32 data (bfsg.data) of n x d values.

Usage: python3 strip_fbin_header.py <base.fbin> <bfsg.data>
"""
import argparse
import struct


def strip_header(input_file: str, output_file: str) -> None:
    with open(input_file, 'rb') as f:
        num, dim = struct.unpack('<II', f.read(8))
        data = f.read()

    expected_size = num * dim * 4
    if len(data) != expected_size:
        print(f'[WARN] File size mismatch! Expected {expected_size} bytes but got {len(data)}.')
    else:
        print(f'[OK] Detected num={num}, dim={dim}, total floats={num * dim}')

    with open(output_file, 'wb') as f:
        f.write(data)

    print(f'[DONE] Wrote pure float data to {output_file}')
    print(f'[INFO] You can now load it with Data({num}, {dim})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input_file', help='fbin file')
    parser.add_argument('output_file', help='raw float32 output')
    args = parser.parse_args()
    strip_header(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
