import numpy as np
import pandas as pd

# Um único gerador para os Exercícios 1 e 2, na ordem do notebook.
rng = np.random.default_rng(42)

parametros = {
    0: {'media': [2, 3], 'desvio': [0.8, 2.5]},
    1: {'media': [5, 6], 'desvio': [1.2, 1.9]},
    2: {'media': [8, 1], 'desvio': [0.9, 0.9]},
    3: {'media': [15, 4], 'desvio': [0.5, 2.0]},
}

def gerar_dataset(s=1.0, random_state=None):
    if random_state is None:
        random_state = rng
    dados = []
    for classe, p in parametros.items():
        pontos = random_state.normal(
            loc=p['media'], scale=np.array(p['desvio']) * s, size=(100, 2)
        )
        df_classe = pd.DataFrame(pontos, columns=['x', 'y'])
        df_classe['classe'] = classe
        dados.append(df_classe)
    return pd.concat(dados, ignore_index=True)

# Ordem exata: Exercício 1A e depois Exercício 1B.
df_s1 = gerar_dataset(1.0)
fatores_s = [0.5, 1.0, 2.0, 4.0]
datasets_s = {s: gerar_dataset(s) for s in fatores_s}
