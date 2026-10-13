# A única diferença de arquitetura é acrescentar uma camada oculta de 24 neurônios.
seeds=[42,43,44];runs=[]

for seed in seeds:
    a=MLP([4,24,3],task='multi',seed=seed); ha=a.fit(Xmt,ymt,lr=LR,epochs=EPOCHS)
    b=MLP([4,24,24,3],task='multi',seed=seed); hd=b.fit(Xmt,ymt,lr=LR,epochs=EPOCHS)
    runs.append((seed,a.score(Xme,yme),b.score(Xme,yme),ha[-1],hd[-1]))
    print('Semente, acurácia rasa, profunda, perda rasa, profunda:',runs[-1])

R=np.array(runs,float);deep=MLP([4,24,24,3],task='multi',seed=42);hdeep=deep.fit(Xmt,ymt,lr=LR,epochs=EPOCHS)
print('Parâmetros rasa/profunda:',shallow.count_params(),deep.count_params())
print('Médias acurácia rasa/profunda:',R[:,1].mean(),R[:,2].mean(),'diferença profunda-rasa:',R[:,2].mean()-R[:,1].mean())
plt.plot(hs,label='1 camada oculta');plt.plot(hdeep,label='2 camadas ocultas');plt.title('Figura 6: Perda de treino — comparação de profundidade');plt.xlabel('Época');plt.ylabel('Entropia cruzada categórica');plt.legend();savefig(6)

