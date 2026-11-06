"""Parts C and D: k-means, model selection, PCA-before-clustering, SNR sweep,
GMM / agglomerative comparison, and the make_moons failure case.

LABEL POLICY: ground-truth labels y are used ONLY to compute external
metrics (ARI/NMI), a contingency table, or to colour plots.  They are never
passed to any fit(), nor used to choose k or any hyperparameter -- except in
the supervised-baseline stand-in for Lab 3 in C12(c), which is explicitly a
supervised classifier and is labelled as such.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score as ARI, normalized_mutual_info_score as NMI,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from datasets import MODS, get_features

SEED = 0


def km(X, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def run_C(Xs, y, Z2, fig_dir, tab_dir):
    out = {}
    # ---- C9
    m = km(Xs, 4)
    out["k4"] = {"inertia": float(m.inertia_),
                 "silhouette": float(silhouette_score(Xs, m.labels_)),
                 "ARI": float(ARI(y, m.labels_)), "NMI": float(NMI(y, m.labels_))}
    ct = pd.crosstab(pd.Series(y, name="true").map(dict(enumerate(MODS))),
                     pd.Series(m.labels_, name="cluster"))
    ct.to_csv(f"{tab_dir}/C9_contingency.csv")
    out["contingency"] = ct.values.tolist()
    out["contingency_index"] = list(ct.index)

    # ---- C10: k sweep (ARI computed but never used to pick k)
    rows = []
    for k in range(2, 9):
        mk = km(Xs, k)
        rows.append({"k": k, "inertia": float(mk.inertia_),
                     "silhouette": float(silhouette_score(Xs, mk.labels_)),
                     "ARI_eval_only": float(ARI(y, mk.labels_)),
                     "NMI_eval_only": float(NMI(y, mk.labels_))})
    sw = pd.DataFrame(rows)
    sw.to_csv(f"{tab_dir}/C10_k_sweep.csv", index=False)
    out["k_sweep"] = rows
    out["k_best_sil"] = int(sw.k[sw.silhouette.idxmax()])
    out["k_best_ari"] = int(sw.k[sw.ARI_eval_only.idxmax()])
    # simple elbow heuristic (internal only): largest 2nd difference of inertia
    inert = sw.inertia.values
    d2 = inert[:-2] - 2 * inert[1:-1] + inert[2:]
    out["elbow_k_2nd_diff"] = int(sw.k.values[1:-1][np.argmax(d2)])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(sw.k, sw.inertia, "o-"); ax[0].set_title("Inertia (elbow)")
    ax[1].plot(sw.k, sw.silhouette, "o-", c="C1"); ax[1].set_title("Silhouette (internal)")
    ax[2].plot(sw.k, sw.ARI_eval_only, "o-", c="C2"); ax[2].set_title("ARI (evaluation only)")
    for a in ax:
        a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/C10_k_selection.png", dpi=150); plt.close(fig)

    # ---- C11: full 12-D vs 2-D PCA
    m2 = km(Z2, 4)
    cmp = pd.DataFrame([
        {"space": "12-D standardized", "ARI": ARI(y, m.labels_), "NMI": NMI(y, m.labels_),
         "silhouette": silhouette_score(Xs, m.labels_)},
        {"space": "2-D PCA", "ARI": ARI(y, m2.labels_), "NMI": NMI(y, m2.labels_),
         "silhouette": silhouette_score(Z2, m2.labels_)}])
    cmp.to_csv(f"{tab_dir}/C11_pca_before_clustering.csv", index=False)
    out["C11"] = cmp.to_dict("records")
    # silhouette of the *12-D clustering* measured in 2-D space, for like-for-like comparison
    out["C11_sil_12Dlabels_in_2D"] = float(silhouette_score(Z2, m.labels_))
    return out


def run_C12(fig_dir, tab_dir, snrs=(20, 15, 10, 5, 0)):
    rows = []
    for snr in snrs:
        X, y = get_features(snr_db=snr, seed=SEED)
        Xs = StandardScaler().fit_transform(X)
        mk = km(Xs, 4)
        # supervised stand-in for the Lab-3 curve: logistic regression, 70/30 split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
        # supplementary robustness check: mean/std ARI over 10 independent datasets (seeds 0..9)
        aris = []
        for s in range(10):
            Xr, yr = get_features(snr_db=snr, seed=s)
            aris.append(ARI(yr, km(StandardScaler().fit_transform(Xr), 4, seed=s).labels_))
        rows.append({"snr_db": snr, "ARI": float(ARI(y, mk.labels_)),
                     "NMI": float(NMI(y, mk.labels_)),
                     "silhouette": float(silhouette_score(Xs, mk.labels_)),
                     "supervised_acc_LR": float(clf.score(Xte, yte)),
                     "ARI_mean_10": float(np.mean(aris)), "ARI_std_10": float(np.std(aris, ddof=1))})
    df = pd.DataFrame(rows)
    df.to_csv(f"{tab_dir}/C12_snr_sweep.csv", index=False)
    # random-assignment ARI baseline (empirical)
    rng = np.random.default_rng(SEED)
    rnd = [ARI(y, rng.integers(0, 4, len(y))) for _ in range(200)]
    base = {"mean": float(np.mean(rnd)), "std": float(np.std(rnd))}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(df.snr_db, df.ARI, "o-"); ax[0].axhline(0, c="k", lw=0.6)
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("ARI"); ax[0].set_title("k-means (k=4): ARI vs SNR")
    ax[0].invert_xaxis()
    ax[1].plot(df.snr_db, df.ARI, "o-", label="k-means ARI")
    ax[1].plot(df.snr_db, df.supervised_acc_LR, "s--", label="supervised acc (LR stand-in)")
    ax[1].set_xlabel("SNR (dB)"); ax[1].legend(fontsize=8); ax[1].invert_xaxis()
    ax[1].set_title("Unsupervised vs supervised")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/C12_snr_sweep.png", dpi=150); plt.close(fig)
    return {"rows": rows, "random_ari": base}


def run_D13(Xs, y, tab_dir, n_seeds=10):
    def models(seed):
        return {"KMeans": KMeans(4, n_init=10, random_state=seed),
                "GMM (full)": GaussianMixture(4, covariance_type="full", random_state=seed),
                "Agglomerative (ward)": AgglomerativeClustering(4, linkage="ward")}
    rows = []
    for name, mdl in models(SEED).items():
        lab = mdl.fit_predict(Xs)
        rows.append({"method": name, "ARI": float(ARI(y, lab)), "NMI": float(NMI(y, lab))})
    single = pd.DataFrame(rows)
    single.to_csv(f"{tab_dir}/D13_comparison.csv", index=False)
    # supplementary: 10 independent datasets (seeds 0..9) to judge variability
    sup = {k: [] for k in ["KMeans", "GMM (full)", "Agglomerative (ward)"]}
    supn = {k: [] for k in sup}
    for s in range(n_seeds):
        Xf, ys = get_features(20, seed=s)
        Xz = StandardScaler().fit_transform(Xf)
        for name, mdl in models(s).items():
            lab = mdl.fit_predict(Xz)
            sup[name].append(ARI(ys, lab)); supn[name].append(NMI(ys, lab))
    multi = pd.DataFrame([{"method": k, "ARI_mean": np.mean(v), "ARI_std": np.std(v, ddof=1),
                           "NMI_mean": np.mean(supn[k]), "NMI_std": np.std(supn[k], ddof=1)}
                          for k, v in sup.items()])
    multi.to_csv(f"{tab_dir}/D13_comparison_10datasets.csv", index=False)
    return {"single": rows, "multi": multi.to_dict("records"), "n_seeds": n_seeds}


def run_D14(fig_dir):
    X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
    lk = KMeans(2, n_init=10, random_state=SEED).fit_predict(X)
    ls = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                            random_state=SEED).fit_predict(X)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(X[:, 0], X[:, 1], c=lk, cmap="coolwarm", s=10); ax[0].set_title(
        f"k-means (ARI={ARI(y, lk):.2f})")
    ax[1].scatter(X[:, 0], X[:, 1], c=ls, cmap="coolwarm", s=10); ax[1].set_title(
        f"Spectral, kNN affinity (ARI={ARI(y, ls):.2f})")
    for a in ax:
        a.set_aspect("equal")
    fig.suptitle("make_moons(n=500, noise=0.05, seed=0); labels used only for ARI in titles")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/D14_moons.png", dpi=150); plt.close(fig)
    return {"kmeans_ARI": float(ARI(y, lk)), "spectral_ARI": float(ARI(y, ls)),
            "kmeans_NMI": float(NMI(y, lk)), "spectral_NMI": float(NMI(y, ls))}
