"""
Reproduces three figures from the NSNR paper:

  (1) scatter.png         -- Section 6.1, first experiment: metrics vs each other
                             (NSNR on y-axis, KL and MSE on x-axis)
  (2) metric_vs_pd.png    -- Section 6.1, second experiment: each metric vs the
                             empirical probability of detection P_d of the AMF detector
  (3) bound_tightness.png -- after Theorem 2: ratio d_NSNR / d_invariant_KL as the
                             interior eigenvalues of Q sweep between q_min and q_max

Run:  python run_section61_and_tightness.py
Output: three .png files in the current folder.

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.linalg import sqrtm, toeplitz
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# Metric definitions (all depend on C, Chat only through Q = Chat^-1/2 C Chat^-1/2)
# ----------------------------------------------------------------------
def _project_pd(C, floor=1e-2):
    """Project a symmetric matrix onto the PD cone (small floor on eigenvalues)."""
    w, V = np.linalg.eigh((C + C.conj().T) / 2)
    return (V * np.maximum(w, floor)) @ V.conj().T


def ratio_eigs(C, Chat):
    """Eigenvalues of Q = Chat^{-1/2} C Chat^{-1/2} (real, positive)."""
    isq = np.linalg.inv(sqrtm(_project_pd(Chat)))
    Q = isq @ C @ isq
    q = np.linalg.eigvalsh((Q + Q.conj().T) / 2)
    return np.maximum(np.real(q), 1e-12)


def d_nsnr(C, Chat):
    """Worst-case NSNR distance: 1/2 log[ ((qmin+qmax)/2)^2 / (qmin qmax) ]."""
    q = ratio_eigs(C, Chat)
    qmin, qmax = q.min(), q.max()
    return 0.5 * np.log(((qmin + qmax) / 2) ** 2 / (qmin * qmax))


def d_kl(C, Chat):
    """Gaussian KL divergence (through Q)."""
    q = ratio_eigs(C, Chat)
    D = len(q)
    return 0.5 * (q.sum() - D - np.log(q).sum())


def d_invkl(C, Chat):
    """Invariant (scale-free) KL: (D/2) log(AM/GM) of eigenvalues of Q."""
    q = ratio_eigs(C, Chat)
    D = len(q)
    return (D / 2) * np.log(q.mean() / np.exp(np.log(q).mean()))


def d_mse(C, Chat):
    """Normalized MSE between the matrices (Frobenius, scale-normalized)."""
    return np.linalg.norm(C - Chat, "fro") ** 2 / np.linalg.norm(C, "fro") ** 2


def sample_cov(C, M):
    """Sample covariance of M Gaussian samples drawn from N(0, C)."""
    D = C.shape[0]
    X = rng.multivariate_normal(np.zeros(D), C, size=M)
    return np.cov(X, rowvar=False)


# ----------------------------------------------------------------------
# (1) Section 6.1, first experiment: metric scatter
# ----------------------------------------------------------------------
def plot_scatter(D=10, M=200, n_trials=1000, out="scatter.png"):
    O = np.ones((D, D))
    C = 0.5 * O + np.eye(D)            # single very large eigenvalue
    nsnr_v, kl_v, mse_v = [], [], []
    for _ in range(n_trials):
        Chat = _project_pd(sample_cov(C, M))
        nsnr_v.append(d_nsnr(C, Chat))
        kl_v.append(d_kl(C, Chat))
        mse_v.append(d_mse(C, Chat))
    nsnr_v, kl_v, mse_v = map(np.array, (nsnr_v, kl_v, mse_v))

    r_kl = np.corrcoef(kl_v, nsnr_v)[0, 1]
    r_mse = np.corrcoef(mse_v, nsnr_v)[0, 1]
    print(f"[scatter] Pearson KL-NSNR = {r_kl:.2f}, MSE-NSNR = {r_mse:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(kl_v, nsnr_v, s=10, alpha=0.5)
    ax[0].set_xlabel("KL"); ax[0].set_ylabel("NSNR")
    ax[0].set_title(f"KL vs NSNR (r = {r_kl:.2f})")
    ax[1].scatter(mse_v, nsnr_v, s=10, alpha=0.5, color="tab:orange")
    ax[1].set_xlabel("MSE"); ax[1].set_ylabel("NSNR")
    ax[1].set_title(f"MSE vs NSNR (r = {r_mse:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scatter] saved {out}")


# ----------------------------------------------------------------------
# (2) Section 6.1, second experiment: metrics vs probability of detection
# ----------------------------------------------------------------------
def amf_pd(C, Chat, snr_db=10.0, pfa=1e-3, n_targets=5, n_mc=2000):
    """Empirical probability of detection of the AMF detector that uses Chat,
    averaged over several random targets, at the given Pfa and SNR."""
    D = C.shape[0]
    Cinv_hat = np.linalg.inv(_project_pd(Chat))
    L = np.linalg.cholesky(C)
    snr = 10 ** (snr_db / 10)
    pds = []
    for _ in range(n_targets):
        s = rng.standard_normal(D)
        s = s / np.sqrt(s @ np.linalg.inv(C) @ s)      # normalize output SNR
        a = np.sqrt(snr)
        w = Cinv_hat @ s
        # H0 noise samples -> threshold for the target Pfa
        n0 = L @ rng.standard_normal((D, n_mc))
        t0 = np.abs(w @ n0) ** 2 / (w @ C @ w)
        thr = np.quantile(t0, 1 - pfa)
        # H1 samples (target present)
        n1 = L @ rng.standard_normal((D, n_mc))
        y1 = a * s[:, None] + n1
        t1 = np.abs(w @ y1) ** 2 / (w @ C @ w)
        pds.append(np.mean(t1 > thr))
    return float(np.mean(pds))


def plot_metric_vs_pd(D=10, rho=0.9, snr_db=10.0, pfa=1e-3, B=900, out="metric_vs_pd.png"):
    C = toeplitz(rho ** np.arange(D))
    nsnr_v, kl_v, nmse_v, pd_v = [], [], [], []
    for _ in range(B):
        M = int(rng.integers(18, 36))                  # 18..35
        Chat = _project_pd(sample_cov(C, M))
        nsnr_v.append(d_nsnr(C, Chat))
        kl_v.append(d_kl(C, Chat))
        nmse_v.append(d_mse(C, Chat))
        pd_v.append(amf_pd(C, Chat, snr_db=snr_db, pfa=pfa))
    nsnr_v, kl_v, nmse_v, pd_v = map(np.array, (nsnr_v, kl_v, nmse_v, pd_v))

    r_nsnr = np.corrcoef(nsnr_v, pd_v)[0, 1]
    r_kl = np.corrcoef(kl_v, pd_v)[0, 1]
    r_nmse = np.corrcoef(nmse_v, pd_v)[0, 1]
    print(f"[metric_vs_pd] Pearson with Pd: NSNR={r_nsnr:.2f}, KL={r_kl:.2f}, NMSE={r_nmse:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, x, name, r, col in [
        (ax[0], nsnr_v, "NSNR", r_nsnr, "tab:blue"),
        (ax[1], kl_v, "KL", r_kl, "tab:green"),
        (ax[2], nmse_v, "NMSE", r_nmse, "tab:red"),
    ]:
        a.scatter(x, pd_v, s=10, alpha=0.5, color=col)
        a.set_xlabel(name); a.set_ylabel(r"$P_d$")
        a.set_title(f"{name} vs $P_d$ (r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metric_vs_pd] saved {out}")


# ----------------------------------------------------------------------
# (3) Bound tightness: ratio d_NSNR / d_invariant_KL vs interior eigenvalue position
# ----------------------------------------------------------------------
def d_nsnr_from_eigs(q):
    qmin, qmax = q.min(), q.max()
    return 0.5 * np.log(((qmin + qmax) / 2) ** 2 / (qmin * qmax))


def d_invkl_from_eigs(q):
    D = len(q)
    return (D / 2) * np.log(q.mean() / np.exp(np.log(q).mean()))


def plot_bound_tightness(D=10, qmin=1.0, qmax=10.0, n=200, out="bound_tightness.png"):
    positions = np.linspace(qmin, qmax, n)
    ratios = []
    for p in positions:
        q = np.concatenate(([qmin], np.full(D - 2, p), [qmax]))   # interior all at p
        dn = d_nsnr_from_eigs(q)
        di = d_invkl_from_eigs(q)
        ratios.append(dn / di if di > 1e-12 else np.nan)
    ratios = np.array(ratios)
    am = (qmin + qmax) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(positions, ratios, lw=2)
    ax.axvline(am, ls="--", color="gray", label=f"arithmetic mean = {am:g}")
    ax.set_xlabel("interior eigenvalue value")
    ax.set_ylabel(r"$d_{\rm NSNR} / d_{\rm invariant\ KL}$")
    ax.set_title("Tightness of the bound in Theorem 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[bound_tightness] saved {out} (max ratio ~ {np.nanmax(ratios):.3f} at the mean)")


if __name__ == "__main__":
    plot_scatter()
    plot_metric_vs_pd()
    plot_bound_tightness()
    print("Done. Three PNGs written to the current folder.")
