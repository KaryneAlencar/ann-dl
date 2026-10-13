binary=MLP([2,16,1],task='binary',seed=42)
hb=binary.fit(Xbt,ybt,lr=.3,epochs=1800)
acc_b=binary.score(Xbe,ybe)
print('Ex. 2 arquitetura [2,16,1], lr=0.3, épocas=1800, acurácia:',f'{acc_b:.2%}')
plt.plot(np.arange(1,len(hb)+1),hb,label='Treino');plt.title('Figura 2: Perda de treino — binário');plt.xlabel('Época');plt.ylabel('Entropia cruzada binária');plt.legend();savefig(2)
xmin,xmax=Xb[:,0].min()-1,Xb[:,0].max()+1; ymin,ymax=Xb[:,1].min()-1,Xb[:,1].max()+1
xx,yy=np.meshgrid(np.linspace(xmin,xmax,240),np.linspace(ymin,ymax,240)); zz=binary.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,zz,alpha=.22,levels=[-.5,.5,1.5],cmap='coolwarm')
for k in [0,1]:plt.scatter(*Xbe[ybe==k].T,s=17,label=f'Classe {k}',alpha=.75)
plt.title('Figura 3: Fronteira de decisão do MLP');plt.xlabel('x1');plt.ylabel('x2');plt.legend();savefig(3)

