import matplotlib.pyplot as plt
import seaborn as sns
from data_shared import parametros, df_s1

print('Amostras:', len(df_s1))
print(df_s1['classe'].value_counts().sort_index().to_string())
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_s1, x='x', y='y', hue='classe', palette='tab10', s=50, alpha=0.7)
for p in parametros.values():
    plt.scatter(p['media'][0], p['media'][1], color='red', marker='X', s=200, edgecolor='black')
plt.title('Figura 1: Nuvens de Pontos em 2D (s=1.0)')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend(title='Classe')
plt.tight_layout()
plt.savefig('fig1.png')
plt.show()
