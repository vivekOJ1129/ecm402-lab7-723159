"""Reproduce every number, table and figure:  python src/main.py

Seeds (all fixed):
  sources np.random.seed=0 | Gaussian sources default_rng(1) | rotation baseline default_rng(123)
  FastICA random_state=0   | modulation dataset seed=0       | KMeans/GMM/Spectral random_state=0
  make_moons random_state=0
Ground-truth labels are used only for evaluation (ARI/NMI/contingency) or plot colouring.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

import bss                       # noqa: E402
import clustering                # noqa: E402
import datasets                  # noqa: E402
import dimred                    # noqa: E402


def main():
    res = {}
    print("Part A ..."); res["A"] = bss.run(FIG, TAB)
    print("Part B ...")
    Xf, y = datasets.get_features(snr_db=20, seed=0)          # dataset used: synthetic modulation
    res["B"], Xs, Z2 = dimred.run(Xf, y, FIG, TAB)
    print("Part C ..."); res["C"] = clustering.run_C(Xs, y, Z2, FIG, TAB)
    res["C12"] = clustering.run_C12(FIG, TAB)
    print("Part D ...")
    res["D13"] = clustering.run_D13(Xs, y, TAB)
    res["D14"] = clustering.run_D14(FIG)
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done -> results/")


if __name__ == "__main__":
    main()
