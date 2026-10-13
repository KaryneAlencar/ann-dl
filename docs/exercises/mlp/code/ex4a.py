deep = MLP([4,24,24,3], task='multi', seed=42)
hdeep = deep.fit(Xmt, ymt, lr=LR, epochs=EPOCHS)
print(f'Acurácia da rede profunda: {deep.score(Xme, yme):.2%}')
