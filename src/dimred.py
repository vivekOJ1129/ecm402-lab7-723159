"""Part B: PCA -- scree, reconstruction error, 2-D projection, loadings.

Labels y are used ONLY to colour the scatter plot (never to fit PCA).
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datasets import MODS, FEATURE_NAMES


def run(Xf, y, fig_dir, tab_dir):
    out = {}
    Xs = StandardScaler().fit_transform(Xf)
    N, d = Xs.shape

    # ---- B6: scree
    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    out["evr"] = evr.tolist()
    out["cum_evr"] = cum.tolist()
    out["n90"] = int(np.argmax(cum >= 0.90) + 1)
    out["n95"] = int(np.argmax(cum >= 0.95) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(range(1, d + 1), evr, color="C0")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("explained variance ratio")
    ax[0].set_title("Scree plot")
    ax[1].plot(range(1, d + 1), cum, "o-")
    ax[1].axhline(0.9, ls="--", c="C3", lw=1); ax[1].axhline(0.95, ls=":", c="C2", lw=1)
    ax[1].set_xlabel("# components"); ax[1].set_ylabel("cumulative EVR")
    ax[1].set_title("Cumulative explained variance (dashed 90%, dotted 95%)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/B6_scree.png", dpi=150); plt.close(fig)

    # ---- B7: reconstruction error, MSE(q) = mean_n ||x_n - xhat_n||^2
    lam = pca.explained_variance_          # ddof = N-1
    rows = []
    for q in range(1, d + 1):
        p = PCA(n_components=q).fit(Xs)
        Xh = p.inverse_transform(p.transform(Xs))
        mse = float(np.mean(np.sum((Xs - Xh) ** 2, axis=1)))
        pred = float((N - 1) / N * lam[q:].sum())
        rows.append({"q": q, "MSE_measured": mse, "MSE_predicted": pred,
                     "abs_diff": abs(mse - pred)})
    rec = pd.DataFrame(rows)
    rec.to_csv(f"{tab_dir}/B7_reconstruction_mse.csv", index=False)
    out["recon"] = rows
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(rec.q, rec.MSE_measured, "o-", label="measured")
    ax.plot(rec.q, rec.MSE_predicted, "x--", label="(N-1)/N * sum discarded eigenvalues")
    ax.set_xlabel("q"); ax.set_ylabel("reconstruction MSE (per sample)")
    ax.set_title("PCA reconstruction error"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/B7_reconstruction.png", dpi=150); plt.close(fig)

    # ---- B8: 2-D projection (labels only colour the plot)
    p2 = PCA(n_components=2).fit(Xs)
    Z2 = p2.transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, m in enumerate(MODS):
        ax.scatter(Z2[y == k, 0], Z2[y == k, 1], s=8, alpha=0.6, label=m)
    ax.set_xlabel(f"PC1 ({p2.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({p2.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA projection (labels used only for colour)")
    ax.legend(markerscale=2)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/B8_pca_scatter.png", dpi=150); plt.close(fig)

    load = pca.components_[0]
    order = np.argsort(-np.abs(load))
    out["pc1_loadings"] = {FEATURE_NAMES[i]: float(load[i]) for i in order}
    out["pc2_loadings"] = {FEATURE_NAMES[i]: float(pca.components_[1][i])
                           for i in np.argsort(-np.abs(pca.components_[1]))}
    pd.DataFrame({"feature": [FEATURE_NAMES[i] for i in order],
                  "PC1_loading": load[order],
                  "PC2_loading": pca.components_[1][order]}
                 ).to_csv(f"{tab_dir}/B8_loadings.csv", index=False)

    # class-centroid separation in PC space (descriptive only; uses y for evaluation)
    cent = np.array([Z2[y == k].mean(0) for k in range(4)])
    spread = np.array([Z2[y == k].std(0).mean() for k in range(4)])
    pair = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pair[f"{MODS[i]}-{MODS[j]}"] = float(
                np.linalg.norm(cent[i] - cent[j]) / ((spread[i] + spread[j]) / 2))
    out["pc2d_pair_separation"] = pair
    return out, Xs, Z2
