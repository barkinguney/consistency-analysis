# for a given reaction, collect arrhenius parameters from multiple sources. 
# calculate their standard deviations
# calculate a single uncertainty factor either on rate coefficients or on A, but it should capture effects of all 3 parameters.
import pandas as pd

from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

def fit_kT_logA(T, k, k_sigma=None, log10A0=0.0, n0=1.0, Ea0=0.0,
                bounds=((-np.inf, -np.inf, -np.inf),
                        ( np.inf,  np.inf,  np.inf))):
    """
    Fit k(T) = 10^(log10A) * T^n * exp(-Ea/T) using scipy.optimize.least_squares.

    Parameters
    ----------
    T : array_like
        Temperatures (T > 0).
    k : array_like
        Measured k(T) (k > 0).
    k_sigma : array_like or None
        1-sigma uncertainties of k(T). If provided, weighted least squares is used.
    log10A0, n0, Ea0 : float
        Initial guesses.
    bounds : 2-tuple
        Bounds for (log10A, n, Ea).

    Returns
    -------
    dict with keys:
        log10A, n, Ea
        log10A_std, n_std, Ea_std
        cov, corr, dof, chi2, red_chi2, lsq_result
    """
    T = np.asarray(T, float).ravel()
    k = np.asarray(k, float).ravel()

    if T.size != k.size:
        raise ValueError("T and k must have the same length.")
    if np.any(T <= 0) or np.any(k <= 0):
        raise ValueError("All T and k must be > 0.")

    if k_sigma is not None:
        k_sigma = np.asarray(k_sigma, float).ravel()
        if k_sigma.size != k.size:
            raise ValueError("k_sigma must match k.")
        if np.any(k_sigma <= 0):
            raise ValueError("All k_sigma must be > 0.")

    # Model
    def model(p):
        log10A, n, Ea = p
        return (10.0**log10A) * (T**n) * np.exp(-Ea / T)

    # Residuals (weighted if k_sigma is given)
    def residuals(p):
        r = model(p) - k
        return r if k_sigma is None else r / k_sigma

    p0 = np.array([log10A0, n0, Ea0], dtype=float)

    # Least-squares fit
    lsq = least_squares(
        residuals,
        p0,
        jac="2-point",
        bounds=bounds,
        method="trf"
    )

    # Best-fit parameters
    log10A_hat, n_hat, Ea_hat = lsq.x

    # --- Uncertainties from covariance matrix ---
    m = k.size
    p = lsq.x.size
    dof = max(0, m - p)

    # Jacobian of residuals
    J = lsq.jac

    # Covariance ≈ s^2 * (J^T J)^(-1)
    JTJ_inv = np.linalg.pinv(J.T @ J)

    if dof > 0:
        s_sq = 2.0 * lsq.cost / dof   # residual variance
    else:
        s_sq = np.nan

    cov = s_sq * JTJ_inv
    std = np.sqrt(np.diag(cov))

    log10A_std, n_std, Ea_std = std

    # Correlation matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)

    chi2 = 2.0 * lsq.cost
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    return {
        "log10A": log10A_hat,
        "n": n_hat,
        "Ea": Ea_hat,
        "log10A_std": log10A_std,
        "n_std": n_std,
        "Ea_std": Ea_std,
        "cov": cov,
        "corr": corr,
        "dof": dof,
        "chi2": chi2,
        "red_chi2": red_chi2,
        "lsq_result": lsq,
    }

def fit_kT_logspace(T, k, k_sigma=None, log10A0=0.0, n0=1.0, Ea0=0.0,
                    bounds=((-np.inf, -np.inf, -np.inf),
                            ( np.inf,  np.inf,  np.inf))):
    T = np.asarray(T, float).ravel()
    k = np.asarray(k, float).ravel()

    if T.size != k.size:
        raise ValueError("T and k must have the same length.")
    if np.any(T <= 0) or np.any(k <= 0):
        raise ValueError("All T and k must be > 0.")

    ln_k = np.log(k)

    if k_sigma is not None:
        k_sigma = np.asarray(k_sigma, float).ravel()
        if k_sigma.size != k.size:
            raise ValueError("k_sigma must match k.")
        if np.any(k_sigma <= 0):
            raise ValueError("All k_sigma must be > 0.")
        # Approx. sigma for ln(k): sigma_ln_k ≈ sigma_k / k
        sigma_ln_k = k_sigma / k
    else:
        sigma_ln_k = None

    ln10 = np.log(10.0)

    def ln_model(p):
        log10A, n, Ea = p
        return ln10 * log10A + n * np.log(T) - Ea / T

    def residuals(p):
        r = ln_model(p) - ln_k
        return r if sigma_ln_k is None else r / sigma_ln_k

    p0 = np.array([log10A0, n0, Ea0], dtype=float)

    lsq = least_squares(residuals, p0, jac="2-point", bounds=bounds, method="trf")

    # Covariance from Jacobian of residuals
    m = k.size
    p = lsq.x.size
    dof = max(0, m - p)

    J = lsq.jac
    JTJ_inv = np.linalg.pinv(J.T @ J)
    s_sq = (2.0 * lsq.cost / dof) if dof > 0 else np.nan
    cov = s_sq * JTJ_inv
    std = np.sqrt(np.diag(cov))

    return {
        "log10A": lsq.x[0], "n": lsq.x[1], "Ea": lsq.x[2],
        "log10A_std": std[0], "n_std": std[1], "Ea_std": std[2],
        "cov": cov, "dof": dof, "chi2": 2.0 * lsq.cost,
        "red_chi2": (2.0 * lsq.cost / dof) if dof > 0 else np.nan,
        "lsq_result": lsq,
    }

def fit_kT_logspace_reparam(T, k, k_sigma=None, Tref=None, b0_0=0.0, n0=1.0, Ea0=0.0,
                            bounds=((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf))):
    T = np.asarray(T, float).ravel()
    k = np.asarray(k, float).ravel()
    ln_k = np.log(k)

    if Tref is None:
        # geometric mean is usually a good centering point for Arrhenius data
        Tref = np.exp(np.mean(np.log(T)))

    if k_sigma is not None:
        k_sigma = np.asarray(k_sigma, float).ravel()
        sigma_ln = k_sigma / k
    else:
        sigma_ln = None

    x1 = np.log(T / Tref)
    x2 = (1.0 / T) - (1.0 / Tref)

    def ln_model(p):
        b0, n, Ea = p
        return b0 + n * x1 - Ea * x2

    def residuals(p):
        r = ln_model(p) - ln_k
        return r if sigma_ln is None else r / sigma_ln

    p0 = np.array([b0_0, n0, Ea0], float)
    lsq = least_squares(residuals, p0, jac="2-point", bounds=bounds, method="trf")

    m = k.size
    p = 3
    dof = max(0, m - p)
    J = lsq.jac
    JTJ_inv = np.linalg.pinv(J.T @ J)
    s_sq = (2.0 * lsq.cost / dof) if dof > 0 else np.nan
    cov_b = s_sq * JTJ_inv

    b0, n, Ea = lsq.x
    return {"b0": b0, "n": n, "Ea": Ea, "Tref": Tref, "cov_b": cov_b, "dof": dof, "lsq_result": lsq, "red_chi2": s_sq}

def convert_reparam_to_original(fit_b):
    b0, n, Ea, Tref = fit_b["b0"], fit_b["n"], fit_b["Ea"], fit_b["Tref"]
    ln10 = np.log(10.0)

    log10A = (b0 - n*np.log(Tref) + Ea/Tref) / ln10

    # Jacobian mapping (b0,n,Ea) -> (log10A,n,Ea)
    J = np.array([
        [1.0/ln10, -np.log(Tref)/ln10, (1.0/Tref)/ln10],
        [0.0,      1.0,               0.0],
        [0.0,      0.0,               1.0]
    ])

    cov_theta = J @ fit_b["cov_b"] @ J.T
    std = np.sqrt(np.diag(cov_theta))
    return {"log10A": log10A, "n": n, "Ea": Ea, "cov": cov_theta,
            "log10A_std": std[0], "n_std": std[1], "Ea_std": std[2], "Tref": Tref}

def save_plot_fit_results(T, k, source, fit_result, output_dir, k_band=None):

    T_plot = np.linspace(np.min(T), np.max(T), T.size)
    log10A = fit_result["log10A"]
    n = fit_result["n"]
    Ea = fit_result["Ea"]
    log10A_std = fit_result["log10A_std"]
    n_std = fit_result["n_std"]
    Ea_std = fit_result["Ea_std"]
    k_fit = (10.0**log10A) * (T_plot**n) * np.exp(-Ea / T_plot)
    k_upper = (10.0**(log10A + log10A_std)) * (T_plot**(n + n_std)) * np.exp(-(Ea - Ea_std) / T_plot)
    k_lower = (10.0**(log10A - log10A_std)) * (T_plot**(n - n_std)) * np.exp(-(Ea + Ea_std) / T_plot)
    uncertainty_factor = np.log10(k_fit/k_lower)


    T = 1000.0 / T  # Convert to 1000/T for better plotting scale
    T_plot = 1000.0 / T_plot

    plt.figure()
    fig, ax = plt.subplots(figsize=(8, 6))

    #Step 1: group data by source
    grouped = defaultdict(lambda: {'T': [], 'k': []})
    for t, ki, s in zip(T, k, source):
        grouped[s]['T'].append(t)
        grouped[s]['k'].append(ki)

    # Step 2: plot each source with a different color
    for s, values in grouped.items():
        plt.semilogy(values['T'], values['k'], '-',label=s)

    plt.semilogy(T_plot, k_fit, '-', label='Fit')
    plt.fill_between(T_plot, k_lower, k_upper, color='gray', alpha=0.5, label='1-sigma interval')
    plt.xlabel('1000/T [1/K]')
    plt.ylabel('Rate Coefficient [m^3/(mol*s)]')
    plt.title(f'Arrhenius Fit, f ={np.min(uncertainty_factor):.3f} - {np.max(uncertainty_factor):.3f}')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(output_dir)

def mc_band(T, theta_hat, cov, param="log10A", n_samp=20000, q=(0.025,0.975), rng=None):
    """
    theta_hat: (lnA, n, Ea) if param='lnA'
               (log10A, n, Ea) if param='log10A'
    cov: matching covariance
    Returns pointwise quantile bands for k(T).
    """
    rng = np.random.default_rng(rng)
    T = np.asarray(T, float)

    draws = rng.multivariate_normal(mean=theta_hat, cov=cov, size=n_samp)

    if param == "log10A":
        log10A, n, Ea = draws.T
        A = 10.0**log10A
    else:
        lnA, n, Ea = draws.T
        A = np.exp(lnA)

    # compute k for each draw and T: (n_samp, m)
    K = (A[:, None] * (T[None, :]**n[:, None]) * np.exp(-Ea[:, None]/T[None, :]))

    lo = np.quantile(K, q[0], axis=0)
    hi = np.quantile(K, q[1], axis=0)
    med = np.quantile(K, 0.5, axis=0)
    return med, lo, hi

def corr_from_cov(cov):
    cov = np.asarray(cov, float)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)

if __name__ == "__main__":

    input_dir = Path("arrhenius_uncertainty/results/rate_coefficients")
    out_dir_plot = Path("arrhenius_uncertainty/results/fit_plots")
    out_dir_fit = Path("arrhenius_uncertainty/results/ls_fits")
    out_dir_plot.mkdir(parents=True, exist_ok=True)
    out_dir_fit.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.csv"):
        print(f"\n{'='*60}")
        try:
            rate_coeff_df = pd.read_csv(file)
        except:
            print(f"Could not read {file}")
            continue
        print(file.name, rate_coeff_df.shape)

        T = rate_coeff_df['Temperature (K)'].to_numpy(dtype=float)
        k = rate_coeff_df['Rate Coefficient (m^3/(mol*s))'].to_numpy(dtype=float)
        source = rate_coeff_df['Library'].to_numpy(dtype=str)

        # try:
        #     ls_fit = fit_kT_logspace(
        #         T, k,
        #         log10A0=30.0, n0=2.0, Ea0=500.0
        #     )
        # except Exception as e:
        #     print(f"Fit failed for {file}: {e}")
        #     continue

        
        try:
            fit = fit_kT_logspace_reparam(
                T, k,
            )
            ls_fit = convert_reparam_to_original(fit)
        except Exception as e:
            print(f"Fit failed for {file}: {e}")
            continue

        print("Fit results (1-sigma):")
        print(f"log10(A) = {ls_fit['log10A']:.6g} ± {ls_fit['log10A_std']:.3g}")
        print(f"n        = {ls_fit['n']:.6g} ± {ls_fit['n_std']:.3g}")
        print(f"Ea       = {ls_fit['Ea']:.6g} ± {ls_fit['Ea_std']:.3g}")
        print(f"DoF = {fit['dof']}, reduced chi2 = {fit['red_chi2']:.3g}")

        T_range = np.linspace(np.min(T), np.max(T), T.size)   

        # C = corr_from_cov(ls_fit["cov"])
        # print(C)
        plot_name = file.stem.replace("rate_coefficients_", "")
        plot_path = out_dir_plot / f"fit_{plot_name}.png"
        save_plot_fit_results(T, k, source, ls_fit, plot_path)

        fit_save = {
            "reaction": rate_coeff_df['Reaction'].iloc[0],  # assuming all rows have the same reaction
            "function": "k(T) = 10^(log10A) * T^n * exp(-Ea/T)",
            "p_hat": [ls_fit["log10A"], ls_fit["n"], ls_fit["Ea"]],   # best-fit parameters [b0, n, Ea]
            "cov": ls_fit["cov"],            # covariance matrix
            "dof": fit["dof"],              # degrees of freedom
            "red_chi2": fit["red_chi2"],    # reduced chi2
            "Tref": ls_fit["Tref"],            # reference temperature
        }

        # Save to disk
        with open(out_dir_fit / f"fit_{plot_name}.pkl", "wb") as f:
            pickle.dump(fit_save, f)



            
