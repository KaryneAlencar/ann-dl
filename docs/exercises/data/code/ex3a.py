from pathlib import Path
import pandas as pd

# Aceita train.csv na pasta do script ou no diretório atual.
candidatos = [Path(__file__).resolve().parent / 'train.csv', Path.cwd() / 'train.csv']
caminho = next((p for p in candidatos if p.is_file()), None)
if caminho is None:
    raise FileNotFoundError('Coloque train.csv (Spaceship Titanic) na mesma pasta de ex3a.py ou na pasta de execução.')
df = pd.read_csv(caminho)

if __name__ == '__main__':
    balanceamento = df['Transported'].value_counts(normalize=True) * 100
    print('Balanceamento da coluna Transported:')
    print(balanceamento.to_string())
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    print('Numéricas:', numericas)
    print('Categóricas:', categoricas)
    faltantes_abs = df.isnull().sum()
    faltantes_pct = (df.isnull().mean() * 100).round(2)
    tabela_faltantes = pd.DataFrame({'Contagem Absoluta': faltantes_abs, 'Percentual (%)': faltantes_pct})
    tabela_faltantes = tabela_faltantes[tabela_faltantes['Contagem Absoluta'] > 0].sort_values('Percentual (%)', ascending=False)
    print('\nValores faltantes:')
    print(tabela_faltantes.to_string())
    colunas_gasto = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    estatisticas_gasto = df[colunas_gasto].agg(['mean', 'median', 'max']).T
    print('\nEstatísticas de gasto:')
    print(estatisticas_gasto.to_string())
