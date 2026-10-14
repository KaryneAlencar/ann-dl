import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from ex2a import mu_A, mu_B, df_dataset_1
from ex2b import df_C, df_D, df_dataset_2

features = ['x1', 'x2', 'x3', 'x4', 'x5']
X1, y1 = df_dataset_1[features], df_dataset_1['classe']
X2, y2 = df_dataset_2[features], df_dataset_2['classe']
pca1 = PCA(n_components=2, random_state=42)
pca2 = PCA(n_components=2, random_state=42)
df_pca1 = pd.DataFrame(pca1.fit_transform(X1), columns=['PC1', 'PC2'])
df_pca1['classe'] = y1
df_pca2 = pd.DataFrame(pca2.fit_transform(X2), columns=['PC1', 'PC2'])
df_pca2['classe'] = y2

var_exp1 = pca1.explained_variance_ratio_.sum()
var_exp2 = pca2.explained_variance_ratio_.sum()
print(f'Variância explicada pelos 2 primeiros componentes (Dataset I): {var_exp1:.4f}')
print(f'Variância explicada pelos 2 primeiros componentes (Dataset II): {var_exp2:.4f}')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=df_pca1, x='PC1', y='PC2', hue='classe', ax=axes[0], palette='tab10', alpha=0.7)
sns.scatterplot(data=df_pca2, x='PC1', y='PC2', hue='classe', ax=axes[1], palette='Set2', alpha=0.7)
axes[0].set_title('Dataset I (Gaussianas Deslocadas)')
axes[1].set_title('Dataset II (Cascas Concêntricas)')
for ax in axes:
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
plt.suptitle('Figura 4: Projeção PCA em 2D')
plt.tight_layout()
plt.savefig('fig4.png')
plt.show()

dist_centros_1 = np.linalg.norm(mu_A - mu_B)
mu_C = df_C[features].mean().values
mu_D = df_D[features].mean().values
dist_centros_2 = np.linalg.norm(mu_C - mu_D)
print(f'Distância entre os centros em 5D (Dataset I): {dist_centros_1:.4f}')
print(f'Distância entre os centros em 5D (Dataset II): {dist_centros_2:.4f}')
df_dataset_1['raio'] = np.linalg.norm(df_dataset_1[features].values, axis=1)
df_dataset_2['raio'] = np.linalg.norm(df_dataset_2[features].values, axis=1)
plt.figure(figsize=(10, 6))
sns.histplot(data=df_dataset_2, x='raio', hue='classe', palette='Set2', bins=50, kde=True, alpha=0.6)
plt.title('Figura 5: Histograma dos Raios - Dataset II')
plt.xlabel('Raio do ponto em 5D')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('fig5.png')
plt.show()
