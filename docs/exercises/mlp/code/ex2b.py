from ex2a import Xbt, Xbe, ybt, ybe
from sklearn.linear_model import LogisticRegression

base_b = LogisticRegression(max_iter=2000).fit(Xbt, ybt).score(Xbe, ybe)
print(f'Acurácia da linha de base: {base_b:.2%}')
