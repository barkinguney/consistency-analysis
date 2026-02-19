import numpy as np
from scipy.stats import qmc
import itertools
from scipy.optimize import linprog
import cantera as ct 
import pandas as pd
import pickle
from pathlib import Path

from sensitivity.cantera_related_functions import calc_IDT_constV, multiply_rates, modify_reaction, reset_rates



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

# def lhs_sampling(factors_list, n, seed=1327):
#     dim = len(factors_list)

#     sampler = qmc.LatinHypercube(d=dim, seed=seed)
#     sample = sampler.random(n=n)
#     z = qmc.scale(sample, -factors_list, factors_list)
#     m = 10.0 ** z
#     return m
    
#     # l_bounds = 1/np.pow(10, factors_list)
#     # u_bounds = 1*np.pow(10, factors_list)
#     # scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
#     # return scaled_sample

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
    lb, ub, active_reactions_idx, factors_list,
    gas, operating_condition,
    n_samples=None, #use 10*n_monomials if possible 
    deg=2,
    design="lhs",
    fit="l2",
    seed=0, 
    idt_t_max = 0.1
):
    if n_samples is None:
        n_monomials = len(monomial_powers(n_dim=len(active_reactions_idx), deg=deg))
        n_samples = 10 * n_monomials
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
    y = []
    valid_samples = []
    for i, sample in enumerate(X):
        multiply_rates(gas, sample, active_reactions_idx)
        idt = calc_IDT_constV(gas, operating_condition, t_max=idt_t_max, debug=False)
        if idt < idt_t_max*0.99:
            y.append(np.log(idt))
            valid_samples.append(sample)
        else:
            print(f"Sample {i} failed to ignite! Skipping...")
        
        if (i+1) % 50 == 0:
            print(f"Evaluated {i+1}/{len(X)} samples")
    y = np.asarray(y, dtype=float)
    X_filtered = np.array(valid_samples)

    
    # 3) Scale inputs and build basis
    Z = scale_to_unit(X_filtered, lb, ub)
    Z = np.asarray(Z, dtype=float)
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
    operating_conditions = pd.read_csv("sensitivity/results/operating_conditions.csv")
    output_dir = Path("surrogate/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    operating_conditions =operating_conditions.fillna("")
    active_reactions = unc_factors_df['Reaction'].tolist()
    active_reactions_idx = [i for i, rxn in enumerate(ct.Solution(mechanism).reactions()) if rxn.equation in active_reactions]
    unc_factors_df['Uncertainty Factor'].fillna(1.0, inplace=True)# if no f, f=1
    unc_factors = unc_factors_df['Uncertainty Factor'].to_numpy()
    #unc_factors = np.full(len(active_reactions_idx), 0.3) 
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
        
    print(active_reactions)
    print(active_reactions_idx)
    
    for condition in operating_conditions.itertuples(index=False):
     
        model = build_idt_surrogate(
            lb=lower_bound_multipliers,
            ub=upper_bound_multipliers,
            active_reactions_idx=active_reactions_idx,
            factors_list=unc_factors,
            gas=gas,
            operating_condition=condition)
        
        design_point = (1.0,) * len(active_reactions_idx)
        pred = predict_idt(model, design_point)
        print(f"Surrogate predicted IDT for condition {condition}: {pred}")
        reset_rates(gas)
        cantera_pred = calc_IDT_constV(gas, condition)
        print(f"Cantera predicted IDT for condition {condition}: {cantera_pred}")
        
        with open(output_dir / f"{condition[0]}_{condition[1]}_{condition[2]}.pkl", "wb") as f:
            pickle.dump(model, f)
            print(f"Saved surrogate model for condition {condition} to {output_dir / f'{condition[0]}_{condition[1]}_{condition[2]}.pkl'}")