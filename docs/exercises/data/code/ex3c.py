import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from ex3b import X_train, X_test

colunas_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
colunas_cat = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

X_train_num = pd.DataFrame(imputer_num.fit_transform(X_train[colunas_num]), columns=colunas_num, index=X_train.index)
X_test_num = pd.DataFrame(imputer_num.transform(X_test[colunas_num]), columns=colunas_num, index=X_test.index)

X_train_cat = pd.DataFrame(imputer_cat.fit_transform(X_train[colunas_cat]), columns=colunas_cat, index=X_train.index)
X_test_cat = pd.DataFrame(imputer_cat.transform(X_test[colunas_cat]), columns=colunas_cat, index=X_test.index)

colunas_gasto = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

X_train_num['TotalSpend'] = X_train_num[colunas_gasto].sum(axis=1)
X_test_num['TotalSpend'] = X_test_num[colunas_gasto].sum(axis=1)

colunas_num_com_total = colunas_num + ['TotalSpend']

foodcourt_antes = X_train_num['FoodCourt'].copy()

colunas_para_log = colunas_gasto + ['TotalSpend']
for col in colunas_para_log:
    X_train_num[col] = np.log1p(X_train_num[col])
    X_test_num[col] = np.log1p(X_test_num[col])

foodcourt_depois = X_train_num['FoodCourt'].copy()

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_cat_encoded = pd.DataFrame(encoder.fit_transform(X_train_cat), columns=encoder.get_feature_names_out(), index=X_train.index)
X_test_cat_encoded = pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(), index=X_test.index)

scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=colunas_num_com_total, index=X_train.index)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=colunas_num_com_total, index=X_test.index)

X_train_final = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_test_final = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)

print(f"Mínimo no treino: {X_train_final[colunas_num_com_total].values.min():.4f}")
print(f"Máximo no treino: {X_train_final[colunas_num_com_total].values.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(foodcourt_antes, bins=40, ax=axes[0], color='blue')
axes[0].set_title('FoodCourt ANTES de log(1+x)')
sns.histplot(foodcourt_depois, bins=40, ax=axes[1], color='green')
axes[1].set_title('FoodCourt DEPOIS de log(1+x)')
axes[0].set_xlabel('Gasto em FoodCourt')
axes[1].set_xlabel('log(1 + FoodCourt)')
axes[0].set_ylabel('Contagem')
axes[1].set_ylabel('Contagem')
plt.suptitle('Figura 6: Efeito da Transformação Logarítmica')
plt.tight_layout()
plt.savefig('fig6.png')
plt.show()
total_nan_treino = X_train_final.isna().sum().sum()
total_nan_teste = X_test_final.isna().sum().sum()

shape_treino = X_train_final.shape
shape_teste = X_test_final.shape

min_val = X_train_final[colunas_num_com_total].values.min()
max_val = X_train_final[colunas_num_com_total].values.max()

print("--- CHECAGENS FINAIS ---")
print(f"Total de NaNs no Treino: {total_nan_treino}")
print(f"Total de NaNs no Teste: {total_nan_teste}")
print(f"Shape do Treino: {shape_treino}")
print(f"Shape do Teste: {shape_teste}")
print(f"Intervalo de valores contínuos: [{min_val:.4f}, {max_val:.4f}]")
