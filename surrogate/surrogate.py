import numpy as np
from scipy.stats import qmc
import itertools
from scipy.optimize import linprog
import cantera as ct 
import pandas as pd
import pickle
from pathlib import Path

from sensitivity.cantera_related_functions import calc_IDT_constV, multiply_rates, modify_reaction



def sobol_design(n_samples: int, lb: np.ndarray, ub: np.ndarray, seed: int = 0):
    d = len(lb)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    # Sobol prefers power-of-two; if n_samples not power-of-two, still works but less ideal.
    X01 = sampler.random(n=n_samples)          # in [0,1]
    X = qmc.scale(X01, lb, ub)                 # to [lb, ub]
    return X

def lhs_design(n_samples: int, lb: np.ndarray, ub: np.ndarray, seed: int = 0):
    d = len(lb)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    X01 = sampler.random(n=n_samples)
    X = qmc.scale(X01, lb, ub)
    return X

def scale_to_unit(x, lb, ub):
    # map [lb,ub] -> [-1,1]
    return 2.0*(x - lb)/(ub - lb) - 1.0

def monomial_powers(n_dim: int, deg: int):
    """
    Returns list of exponent tuples for all monomials up to total degree = deg.
    Includes constant term (0,...,0).
    """
    powers = []
    for total in range(deg + 1):
        for exps in itertools.product(range(total + 1), repeat=n_dim):
            if sum(exps) == total:
                powers.append(exps)
    return powers

def design_matrix(Z: np.ndarray, powers):
    """
    Z: (n_samples, n_dim) scaled to [-1,1]
    powers: list of exponent tuples
    returns X: (n_samples, n_terms)
    """
    n_samples, n_dim = Z.shape
    X = np.ones((n_samples, len(powers)))
    for j, p in enumerate(powers):
        # product_i Z_i ** p_i
        col = np.ones(n_samples)
        for k in range(n_dim):
            if p[k] != 0:
                col *= Z[:, k] ** p[k]
        X[:, j] = col
    return X

def fit_l2(X, y):
    # solves min ||Xc - y||_2
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c

def fit_linf_lp(X, y):
    n_samples, n_terms = X.shape
    # Variables: [c (n_terms), t (1)]
    # Objective: minimize t
    c_obj = np.zeros(n_terms + 1)
    c_obj[-1] = 1.0

    # Inequalities:
    #  Xc - y <= t  -> [X, -1] [c;t] <= y
    # -Xc + y <= t  -> [-X, -1] [c;t] <= -y
    A_ub = np.vstack([
        np.hstack([ X, -np.ones((n_samples, 1))]),
        np.hstack([-X, -np.ones((n_samples, 1))]),
    ])
    b_ub = np.hstack([y, -y])

    bounds = [(None, None)] * n_terms + [(0, None)]  # t >= 0

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    c = res.x[:-1]
    t = res.x[-1]
    return c, t

def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def cross_validate_poly(Z, y, powers, fit_fn, k=5, seed=0):
    folds = kfold_indices(len(y), k=k, seed=seed)
    errs = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        Xtr = design_matrix(Z[train_idx], powers)
        ytr = y[train_idx]
        Xte = design_matrix(Z[test_idx], powers)
        yte = y[test_idx]

        coef = fit_fn(Xtr, ytr)
        if isinstance(coef, tuple):  # for linf returning (c,t)
            coef = coef[0]
        pred = Xte @ coef
        errs.append(np.max(np.abs(pred - yte)))  # L∞ error on fold
    return float(np.mean(errs)), float(np.max(errs))

def build_idt_surrogate(
    lb, ub,
    gas, operating_condition,
    n_samples=500, #use 10*n_monomials if possible 
    deg=2,
    design="sobol",
    fit="l2",
    seed=0
):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # 1) Design
    if design == "sobol":
        X = sobol_design(n_samples, lb, ub, seed=seed)
    elif design == "lhs":
        X = lhs_design(n_samples, lb, ub, seed=seed)
    else:
        raise ValueError("design must be 'sobol' or 'lhs'")

    # 2) Evaluate log IDT
    for sample in X:   
        multiply_rates(gas, X)
    y = calc_IDT_constV(gas, operating_condition)

    # 3) Scale inputs and build basis
    Z = scale_to_unit(X, lb, ub)
    powers = monomial_powers(n_dim=Z.shape[1], deg=deg)
    Phi = design_matrix(Z, powers)

    # 4) Fit
    if fit == "l2":
        coef = fit_l2(Phi, y)
        linf_train = np.max(np.abs(Phi @ coef - y))
    elif fit == "linf":
        coef, t = fit_linf_lp(Phi, y)
        linf_train = t
    else:
        raise ValueError("fit must be 'l2' or 'linf'")

    # 5) CV
    if fit == "l2":
        fit_fn = fit_l2
    else:
        fit_fn = fit_linf_lp

    mean_cv, worst_cv = cross_validate_poly(Z, y, powers, fit_fn, k=5, seed=seed)

    model = {
        "lb": lb,
        "ub": ub,
        "deg": deg,
        "powers": powers,
        "coef": coef,
        "target": "log_idt",
        "train_linf": float(linf_train),
        "cv_mean_linf": mean_cv,
        "cv_worst_linf": worst_cv
    }
    return model

def predict_idt(model, x):
    x = np.asarray(x, dtype=float)
    z = scale_to_unit(x, model["lb"], model["ub"]).reshape(1, -1)
    Phi = design_matrix(z, model["powers"])
    log_idt = float(Phi @ model["coef"])
    return float(np.exp(log_idt))




if __name__ == "__main__":
    
    # we need to create a surrogate for each operating condition and QOI (IDT)
    
    mechanism = 'Supplementary-3_syngas.yaml'
    unc_factors_df = pd.read_csv("arrhenius_uncertainty/results/uncertainty_factors.csv")
    ls_fit_path = Path("arrhenius_uncertainty/results/ls_fits")
    operating_conditions = None
    active_reactions = unc_factors_df['Reaction'].tolist()
    unc_factors_df['Uncertainty Factor'].fillna(1.0, inplace=True)
    unc_factors = unc_factors_df['Uncertainty Factor'].to_numpy()
    upper_bound_multipliers = np.power(10, unc_factors)
    lower_bound_multipliers = 1 / np.power(10, unc_factors)
    #data_path = "sensitivity/results/impact_factors_ignition_delay.csv"
    
    gas = ct.Solution(mechanism)
    # df_data = pd.read_csv(data_path)
    
    for file_path in ls_fit_path.glob("*.pkl"):
        with open(file_path, "rb") as f:
            ls_fit = pickle.load(f)
            print(f"Loaded {file_path.name}")
            log10A, n, Ea = ls_fit['p_hat']
            A = 10 ** log10A

            modify_reaction(gas, ls_fit['reaction'], [A, n, Ea])
           
    

        
        
    
    lb= []
    ub = []
    for equation in active_reactions:
        rxn_idx = gas.reaction_equations().index(equation)
        net_rate = gas.net_rates_of_progress[rxn_idx]
        
    
    for condition in operating_conditions:
        assert (len(active_reactions)+ 1) == len(monomial_powers(n_dim=len(active_reactions), deg=2))
        no_monomials = len(active_reactions) + 1 
        
        # build_idt_surrogate(
        #     lb=lower_bound_multipliers,
        #     ub=upper_bound_multipliers,
        #     gas=gas,
        #     operating_condition=condition,