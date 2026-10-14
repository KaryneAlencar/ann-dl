import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_shared import parametros, fatores_s, datasets_s

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for idx, s in enumerate(fatores_s):
    ax = axes.flat[idx]
    sns.scatterplot(data=datasets_s[s], x='x', y='y', hue='classe', palette='tab10',
                    s=40, alpha=0.7, ax=ax, legend=(idx == 0))
    for p in parametros.values():
        ax.scatter(p['media'][0], p['media'][1], color='red', marker='X', s=100, edgecolor='black')
    ax.set_title(f'Espalhamento com s = {s}')
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
plt.suptitle('Figura 2: Impacto do Fator de Escala (s) no Espalhamento')
plt.tight_layout()
plt.savefig('fig2.png')
plt.show()

classes = list(parametros.keys())
resultados_rij = []
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        c1, c2 = classes[i], classes[j]
        mu_i, mu_j = np.array(parametros[c1]['media']), np.array(parametros[c2]['media'])
        sigma_i, sigma_j = np.array(parametros[c1]['desvio']), np.array(parametros[c2]['desvio'])
        r_ij = np.linalg.norm(mu_i - mu_j) / (np.mean(sigma_i) + np.mean(sigma_j))
        resultados_rij.append({'Par_de_Classes': f'{c1} e {c2}', 'r_ij': r_ij})
df_rij = pd.DataFrame(resultados_rij)
print('Razões de separação em s=1:')
print(df_rij.to_string(index=False))
idx_menor = df_rij['r_ij'].idxmin()
menor_par = df_rij.loc[idx_menor, 'Par_de_Classes']
menor_rij_s1 = df_rij.loc[idx_menor, 'r_ij']
print(f'Par com menor r_ij: {menor_par}; s=1: {menor_rij_s1:.4f}; s=2: {menor_rij_s1 / 2:.4f}')

medias = np.array([parametros[c]['media'] for c in classes])
taxas_mistura = []
for s in fatores_s:
    df_s = datasets_s[s]
    pontos = df_s[['x', 'y']].values
    distancias = np.linalg.norm(pontos[:, None, :] - medias[None, :, :], axis=2)
    classes_preditas = np.argmin(distancias, axis=1)
    taxa = np.mean(classes_preditas != df_s['classe'].values)
    taxas_mistura.append(taxa)
    print(f'Taxa de mistura para s={s}: {taxa:.4f}')
plt.figure(figsize=(8, 5))
sns.lineplot(x=fatores_s, y=taxas_mistura, marker='o', linewidth=2)
plt.title('Figura 3: Taxa de Mistura vs Fator de Escala ($s$)')
plt.xlabel('Fator de Escala ($s$)')
plt.ylabel('Taxa de Mistura')
plt.xticks(fatores_s)
plt.tight_layout()
plt.savefig('fig3.png')
plt.show()
