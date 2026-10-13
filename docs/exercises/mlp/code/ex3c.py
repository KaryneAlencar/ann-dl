base_m=LogisticRegression(max_iter=2000).fit(Xmt,ymt).score(Xme,yme)

# Mesma implementação MLP com argumentos alterados.
LR=.2; EPOCHS=2200
shallow=MLP([4,24,3],task='multi',seed=42);hs=shallow.fit(Xmt,ymt,lr=LR,epochs=EPOCHS)
acc_s=shallow.score(Xme,yme)
print('Linha de base multiclasse:',f'{base_m:.2%}','MLP:',f'{acc_s:.2%}')
plt.plot(np.arange(1,len(hs)+1),hs,label='Treino');plt.title('Figura 4: Perda de treino — multiclasse');plt.xlabel('Época');plt.ylabel('Entropia cruzada categórica');plt.legend();savefig(4)
cm=confusion_matrix(yme,shallow.predict(Xme));print('Matriz de confusão:\n',cm)
plt.imshow(cm,cmap='Blues');plt.colorbar(label='Quantidade')

for i in range(3):
 for j in range(3):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.xticks(range(3));plt.yticks(range(3));plt.title('Figura 5: Matriz de confusão — MLP');plt.xlabel('Classe prevista');plt.ylabel('Classe real');savefig(5)
pairs={(i,j):int(cm[i,j]+cm[j,i]) for i in range(3) for j in range(i+1,3)}
print('Par mais confundido:',max(pairs,key=pairs.get),'erros bidirecionais:',max(pairs.values()))

