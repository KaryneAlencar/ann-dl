import numpy as np
import pandas as pd
from data_shared import rng

mu_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
amostras_A = rng.multivariate_normal(mu_A, Sigma_A, size=500)
amostras_B = rng.multivariate_normal(mu_B, Sigma_B, size=500)
nomes_colunas = ['x1', 'x2', 'x3', 'x4', 'x5']
df_A = pd.DataFrame(amostras_A, columns=nomes_colunas)
df_A['classe'] = 'A'
df_B = pd.DataFrame(amostras_B, columns=nomes_colunas)
df_B['classe'] = 'B'
df_dataset_1 = pd.concat([df_A, df_B], ignore_index=True)

if __name__ == '__main__':
    print(f'Shape do Dataset I: {df_dataset_1.shape}')
    print(df_dataset_1.head().to_string(index=False))
    print(df_dataset_1.tail().to_string(index=False))
