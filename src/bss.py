"""Part A: blind source separation -- sources, mixing, whitening, ICA vs PCA,
and the Gaussian (unidentifiable) case.

No ground-truth information is passed to fit(); the true sources S are used
only to *evaluate* the recovered components (matched |correlation|).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA

SEED_SOURCES = 0      # np.random.seed for the illustrative sources
SEED_GAUSS = 1        # default_rng seed for the Gaussian sources
SEED_ROT = 123        # random-rotation baseline
SEED_ICA = 0          # FastICA random_state
N = 2000

A_MIX = np.array([[1.0, 1.0, 0.6],
                  [0.6, 1.8, 1.0],
                  [1.4, 0.7, 1.3]])


def make_sources():
    np.random.seed(SEED_SOURCES)
    t = np.linspace(0, 8, N)
    s1 = np.sin(2 * np.pi * 1.0 * t)
    s2 = signal.square(2 * np.pi * 0.7 * t)
    s3 = signal.sawtooth(2 * np.pi * 1.3 * t)
    S = np.c_[s1, s2, s3]
    S += 0.05 * np.random.randn(*S.shape)
    S /= S.std(axis=0)
    X = S @ A_MIX.T
    return t, S, X


def match_corr(S_true, S_est):
    """|Pearson| matrix, one-to-one (Hungarian) matching, matched scores."""
    k = S_true.shape[1]
    C = np.abs(np.corrcoef(S_true.T, S_est.T)[:k, k:])
    r, c = linear_sum_assignment(-C)
    return C, r, c, C[r, c]


def whiten(X):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)           # ddof = 1 throughout
    lam, E = np.linalg.eigh(cov)
    Xw = Xc @ E @ np.diag(lam ** -0.5)
    return Xw, cov, lam, E


def dep_matrix(Z):
    """D_ij = E[zi^2 zj^2] - E[zi^2]E[zj^2] on unit-variance outputs."""
    Z = (Z - Z.mean(0)) / Z.std(0)
    Q = Z ** 2
    return (Q.T @ Q) / len(Z) - np.outer(Q.mean(0), Q.mean(0))


def _plot_six(t, S, X, path):
    fig, ax = plt.subplots(6, 1, figsize=(9, 9), sharex=True)
    for i in range(3):
        ax[i].plot(t, S[:, i], lw=0.8, color="C0")
        ax[i].set_ylabel(f"$s_{i+1}$")
        ax[i + 3].plot(t, X[:, i], lw=0.8, color="C3")
        ax[i + 3].set_ylabel(f"$x_{i+1}$")
    ax[0].set_title("Sources (top three) and observed mixtures (bottom three)")
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_recovered(t, S, S_ica, c_idx, r_idx, path):
    fig, ax = plt.subplots(3, 2, figsize=(11, 6), sharex=True)
    for row, (i, j) in enumerate(zip(r_idx, c_idx)):
        sgn = np.sign(np.corrcoef(S[:, i], S_ica[:, j])[0, 1])
        ax[row, 0].plot(t, S[:, i], lw=0.8)
        ax[row, 0].set_ylabel(f"true $s_{i+1}$")
        ax[row, 1].plot(t, sgn * S_ica[:, j], lw=0.8, color="C1")
        ax[row, 1].set_ylabel(f"ICA comp {j}")
    ax[0, 0].set_title("True sources")
    ax[0, 1].set_title("FastICA components (sign-aligned for display)")
    ax[-1, 0].set_xlabel("time (s)")
    ax[-1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(fig_dir, tab_dir):
    out = {}
    t, S, X = make_sources()
    _plot_six(t, S, X, f"{fig_dir}/A1_sources_mixtures.png")

    # ---- A1(b) condition number
    out["cond_A"] = float(np.linalg.cond(A_MIX))
    out["det_A"] = float(np.linalg.det(A_MIX))
    out["sv_A"] = np.linalg.svd(A_MIX, compute_uv=False).tolist()

    # ---- A2 whitening
    Xw, cov, lam, E = whiten(X)
    out["cov_X"] = cov.tolist()
    out["eig_cov_X"] = lam.tolist()
    out["cov_Xw"] = np.cov(Xw, rowvar=False).tolist()
    out["whiten_maxdev"] = float(np.abs(np.cov(Xw, rowvar=False) - np.eye(3)).max())

    # ---- A3 ICA (X only)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ica = FastICA(n_components=3, whiten="unit-variance",
                      random_state=SEED_ICA, max_iter=1000)
        S_ica = ica.fit_transform(X)
    out["ica_warnings"] = [str(x.message) for x in w]
    out["ica_n_iter"] = int(ica.n_iter_)
    C, r, c, sc = match_corr(S, S_ica)
    out["ica_corr_matrix"] = C.tolist()
    out["ica_match"] = {"true_idx": r.tolist(), "est_idx": c.tolist()}
    out["ica_matched"] = sc.tolist()
    out["ica_mean_matched"] = float(sc.mean())
    _plot_recovered(t, S, S_ica, c, r, f"{fig_dir}/A3_ica_recovered.png")

    # ---- A4 PCA vs ICA
    S_pca = PCA(n_components=3).fit_transform(X)
    Cp, rp, cp, scp = match_corr(S, S_pca)
    out["pca_corr_matrix"] = Cp.tolist()
    out["pca_matched"] = scp.tolist()
    out["pca_mean_matched"] = float(scp.mean())
    out["cov_pca"] = np.cov(S_pca, rowvar=False).tolist()
    out["cov_ica"] = np.cov(S_ica, rowvar=False).tolist()
    D_pca, D_ica = dep_matrix(S_pca), dep_matrix(S_ica)
    D_src = dep_matrix(S)
    out["D_pca"], out["D_ica"], out["D_src"] = D_pca.tolist(), D_ica.tolist(), D_src.tolist()
    off = ~np.eye(3, dtype=bool)
    out["D_pca_offdiag_meanabs"] = float(np.abs(D_pca[off]).mean())
    out["D_ica_offdiag_meanabs"] = float(np.abs(D_ica[off]).mean())
    out["D_src_offdiag_meanabs"] = float(np.abs(D_src[off]).mean())

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (M, ttl) in zip(ax, [(Cp, "PCA vs true"), (C, "ICA vs true"), (None, "")]):
        if M is None:
            continue
        im = a.imshow(M, vmin=0, vmax=1, cmap="viridis")
        a.set_title(f"|corr|: {ttl}")
        a.set_xlabel("estimated comp.")
        a.set_ylabel("true source")
        for (i, j), v in np.ndenumerate(M):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=9)
    pairs = [(0, 1), (0, 2), (1, 2)]
    w = 0.35
    xs = np.arange(3)
    ax[2].bar(xs - w / 2, [abs(D_pca[i, j]) for i, j in pairs], w, label="PCA")
    ax[2].bar(xs + w / 2, [abs(D_ica[i, j]) for i, j in pairs], w, label="ICA")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["(1,2)", "(1,3)", "(2,3)"])
    ax[2].set_ylabel("|D_ij|"); ax[2].set_title("Off-diagonal dependence |D_ij|")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/A4_pca_vs_ica.png", dpi=150)
    plt.close(fig)

    # ---- A5 Gaussian sources
    rng = np.random.default_rng(SEED_GAUSS)
    Sg = rng.standard_normal((N, 3))
    Sg /= Sg.std(axis=0)
    Xg = Sg @ A_MIX.T
    Xgw, *_ = whiten(Xg)
    rrng = np.random.default_rng(SEED_ROT)
    scores = []
    for _ in range(200):
        Q, R = np.linalg.qr(rrng.standard_normal((3, 3)))
        Q = Q @ np.diag(np.sign(np.diag(R)))
        scores.append(match_corr(Sg, Xgw @ Q)[3].mean())
    scores = np.array(scores)
    out["rot_baseline"] = {"mean": float(scores.mean()),
                           "p5": float(np.percentile(scores, 5)),
                           "p95": float(np.percentile(scores, 95))}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        icag = FastICA(n_components=3, whiten="unit-variance",
                       random_state=SEED_ICA, max_iter=1000)
        Sg_ica = icag.fit_transform(Xg)
    out["gauss_ica_warnings"] = [str(x.message) for x in w]
    out["gauss_ica_n_iter"] = int(icag.n_iter_)
    out["gauss_ica_mean_matched"] = float(match_corr(Sg, Sg_ica)[3].mean())
    out["gauss_ica_matched"] = match_corr(Sg, Sg_ica)[3].tolist()
    # stability: different FastICA seeds on the Gaussian mixture
    seeds = [match_corr(Sg, FastICA(n_components=3, whiten="unit-variance",
                                    random_state=s, max_iter=1000).fit_transform(Xg))[3].mean()
             for s in range(10)]
    out["gauss_ica_seed_scores"] = [float(v) for v in seeds]
    # same stability test on the non-Gaussian case
    seeds2 = [match_corr(S, FastICA(n_components=3, whiten="unit-variance",
                                    random_state=s, max_iter=1000).fit_transform(X))[3].mean()
              for s in range(10)]
    out["ica_seed_scores"] = [float(v) for v in seeds2]
    out["kurt_nongauss"] = stats.kurtosis(S, fisher=True).tolist()
    out["kurt_gauss"] = stats.kurtosis(Sg, fisher=True).tolist()
    # theoretical excess kurtosis: sine -1.5, square -2, sawtooth(uniform) -1.2
    out["kurt_theory"] = [-1.5, -2.0, -1.2]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(scores, bins=20, color="C7", alpha=0.8, label="200 random rotations")
    ax.axvline(out["gauss_ica_mean_matched"], color="C3", lw=2, label="FastICA (Gaussian)")
    ax.axvline(out["ica_mean_matched"], color="C2", lw=2, label="FastICA (original signals)")
    ax.set_xlabel("mean matched |corr|"); ax.set_ylabel("count")
    ax.set_title("Gaussian sources: ICA vs random-rotation baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/A5_gaussian_baseline.png", dpi=150)
    plt.close(fig)
    return out
