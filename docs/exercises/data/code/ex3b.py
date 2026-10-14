from sklearn.model_selection import train_test_split
from ex3a import df

X = df.drop(columns=['Transported'])
y = df['Transported']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

if __name__ == '__main__':
    print(f'Tamanho do Treino: {X_train.shape[0]} linhas')
    print(f'Tamanho do Teste: {X_test.shape[0]} linhas')

print("Média de FoodCourt no treino:", X_train["FoodCourt"].mean())
print("Mediana de FoodCourt no treino:", X_train["FoodCourt"].median())