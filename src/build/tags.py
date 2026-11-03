"""Directory tag that src/build/build.sh gives a configuration.

For example tag_of("cagra", "GD=8 IGD=16 NITER=5") returns "gd8_igd16_it5". The arguments must name every
parameter of the tag; PilotANN configurations always use the defaults DP = 192 and RATIO = 0.25.
Usage: python src/build/tags.py <system> "<KEY=VALUE ...>"
"""
import argparse

TAGS = {
    "song": "sd{SD}_shift{SHIFT}_b{BUDGET}",
    "ganns": "k{K}_m{M}",
    "ggnn": "k{K}_tau{TAU}_r{REFINE}",
    "cagra": "gd{GD}_igd{IGD}_it{NITER}",
    "pilotann": "{TYPE}{NN}_efc{EFC}_dp192_r0.25",
    "hnsw": "m{M}_efc{EFC}",
    "vamana": "r{R}_l{L}_a{ALPHA}",
}


def tag_of(system: str, args: str) -> str:
    kv = dict(a.split("=") for a in args.split())
    return TAGS[system].format(**kv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=sorted(TAGS))
    ap.add_argument("args")
    a = ap.parse_args()
    print(tag_of(a.system, a.args))


if __name__ == "__main__":
    main()
