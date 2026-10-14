import numpy as np
import pandas as pd
from ex2a import rng

def gerar_casca(n_amostras, raio_media, raio_std, nome_classe, random_state=None):
    if random_state is None:
        random_state = rng
    v = random_state.standard_normal((n_amostras, 5))
    normas = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / normas
    rho = random_state.normal(loc=raio_media, scale=raio_std, size=(n_amostras, 1))
    x = rho * u
    df = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    df['classe'] = nome_classe
    return df

df_C = gerar_casca(500, 2.0, 0.4, 'C')
df_D = gerar_casca(500, 5.0, 0.4, 'D')
df_dataset_2 = pd.concat([df_C, df_D], ignore_index=True)

if __name__ == '__main__':
    print(f'Shape do Dataset II: {df_dataset_2.shape}')
    print(df_dataset_2.head().to_string(index=False))
    print(df_dataset_2.tail().to_string(index=False))
