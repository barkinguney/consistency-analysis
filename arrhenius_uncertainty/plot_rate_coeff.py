import pandas as pd

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

folder = Path("arrhenius_uncertainty/results")

for csv_file in folder.glob("*.csv"):
    if csv_file.name.startswith("rate_coefficients_"):
        try:
            rate_coeff_df = pd.read_csv(csv_file)
        except:
            print(f"Could not read {csv_file}")
            continue
        print(csv_file.name, rate_coeff_df.shape)
        T = rate_coeff_df['Temperature (K)'].to_numpy(dtype=float)
        k = rate_coeff_df['Rate Coefficient (m^3/(mol*s))'].to_numpy(dtype=float)
        T = 1000.0 / T  # Convert to 1000/T for better plotting scale
        plt.figure()
        #plt.loglog(T, k, 'o', label='Data')
        plt.semilogy(T, k, 'o', label='Data')
        plt.xlabel('1000/Temperature (K)')
        plt.ylabel('Rate Coefficient (m^3/(mol*s))')
        plt.title(f'Rate Coefficients from {csv_file.name}')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.show()
