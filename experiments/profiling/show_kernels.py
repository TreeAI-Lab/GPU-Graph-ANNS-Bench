"""Print the first rows of an Nsight Systems kernel summary (cuda_gpu_kern_sum exported as csv):
share of kernel time, total time, launches and name."""
import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('kern_sum', type=Path, help='kern_sum.csv written by find_kernel.sh')
    ap.add_argument('top', type=int, nargs='?', default=5, help='number of kernels to print')
    args = ap.parse_args()
    with open(args.kern_sum) as f:
        rows = list(csv.DictReader(f))
    for r in rows[:args.top]:
        print(f"{float(r['Time (%)']):6.1f}%  {int(r['Total Time (ns)']):>14,}  x{r['Instances']:>3}  {r['Name'][:170]}")


if __name__ == '__main__':
    main()
