\---



exercise: mlp



ai_use: "Apoio na estruturação do relatório"



\---



**## Exercício 1**



**### A — Passagem direta**



Para a entrada $x=[0.5,-0.2]$ e os parâmetros fornecidos, a pré-ativação oculta é



$$z^{(1)}=W^{(1)}x+b^{(1)}=[0.270000,-0.180000].$$



Aplicando a tangente hiperbólica:



$$h^{(1)}=\tanh(z^{(1)})=[0.263625,-0.178081].$$



Na saída, temos



$$u^{(2)}=0.5(0.263625)-0.3(-0.178081)+0.2=0.385237,$$



$$\hat y=\tanh(u^{(2)})=0.367247,$$



$$L=(1-\hat y)^2=0.400377.$$



**### B — Passagem reversa**



Começando pela derivada da perda em relação à saída:



$$*\\*&#x66;rac{\partial L}{\partial\hat y}=2(\hat y-y)=-1.265507.$$



Como $\tanh'(u)=1-\tanh^2(u)$, o delta da camada de saída é



$$\delta^{(2)}=*\\*&#x66;rac{\partial L}{\partial u^{(2)}}=2(\hat y-y)(1-\hat y^2)=-1.094828.$$



Os gradientes da camada de saída são



$$*\\*&#x66;rac{\partial L}{\partial W^{(2)}}=\delta^{(2)}h^{(1)}=[-0.288624,0.194968],$$



$$*\\*&#x66;rac{\partial L}{\partial b^{(2)}}=\delta^{(2)}=-1.094828.$$



Propagando o erro à camada oculta:



$$*\\*&#x66;rac{\partial L}{\partial h^{(1)}}=(W^{(2)})^T\delta^{(2)}=[-0.547414,0.328448],$$



$$\delta^{(1)}=*\\*&#x66;rac{\partial L}{\partial z^{(1)}}=*\\*&#x66;rac{\partial L}{\partial h^{(1)}}\odot(1-(h^{(1)})^2)=[-0.509370,0.318032].$$



Por fim, os gradientes dos parâmetros ocultos são



$$*\\*&#x66;rac{\partial L}{\partial W^{(1)}}=\delta^{(1)}x^T=*\\*&#x62;egin{bmatrix}-0.254685&0.101874\\\0.159016&-0.06360&#x36;*\\*&#x65;nd{bmatrix},$$



$$*\\*&#x66;rac{\partial L}{\partial b^{(1)}}=[-0.509370,0.318032].$$



**### C — Atualização dos parâmetros**



Usando $*\\*&#x74;heta\_{novo}=\theta-0.3\nabla\_\theta L$, os oito parâmetros escalares ficam:



$$W^{(1)}\_{novo}=*\\*&#x62;egin{bmatrix}0.376405&-0.130562\\\0.152295&0.41908&#x32;*\\*&#x65;nd{bmatrix},\qquad b^{(1)}\_{novo}=[0.252811,-0.295410],$$



$$W^{(2)}\_{novo}=[0.586587,-0.358490],\qquad b^{(2)}\_{novo}=0.528448.$$



A perda após outra passagem direta com os parâmetros atualizados é **\*\*0.083417\*\***, menor que a perda inicial de **\*\*0.400377\*\***. Isso acontece porque o gradiente descendente deslocou os parâmetros na direção oposta ao gradiente, reduzindo a perda neste passo.



**## Exercício 2**



**### A — Gere os dados**



Foram geradas 1000 amostras com \`make_blobs\`, utilizando centros $[0,0]$, $[3,3]$ e $[-3,-3]$, desvio padrão de 1.2 e \`random_state=42\`. A Classe 0 contém 500 amostras do grupo central, enquanto a Classe 1 reúne os dois grupos externos, com 250 amostras cada. Os dados foram divididos em 80% para treino e 20% para teste, com estratificação.



![Figura 1: Dados binários]\(figures/fig1.png)



A Figura 1 mostra a Classe 1 em duas regiões opostas em relação à Classe 0. Por isso, uma única reta não consegue separar corretamente as duas classes.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex2a.py"



\`\`\`



**### B — Linha de base**



A regressão logística treinada no conjunto de treino obteve **\*\*47.50%\*\*** de acurácia no teste. Como sua fronteira de decisão é linear, ela não consegue representar as duas regiões distintas que pertencem à Classe 1.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex2b.py"



\`\`\`



**### C — Implemente o MLP**



O MLP foi implementado do zero com NumPy, usando uma classe que recebe os tamanhos das camadas. A arquitetura escolhida foi **\*\*[2, 16, 1]\*\***, com uma camada oculta de 16 neurônios e ativação \`tanh\`. A saída utiliza sigmoide e a perda é a entropia cruzada binária:



$$L=-\frac1N\sum_i[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)].$$



A retropropagação calcula o delta da saída como $\delta=(\hat y-y)/N$. Em cada camada, os gradientes são $\partial L/\partial W=a\_{anterior}^T\delta$ e $\partial L/\partial b=\sum_i\delta_i$. O delta é propagado usando os pesos e a derivada da \`tanh\`. Todos os pesos e biases são atualizados por gradiente descendente, sem diferenciação automática.



O treinamento utilizou **\*\*taxa de aprendizado 0.3\*\***, **\*\*1800 épocas\*\*** e semente 42. Para garantir reprodutibilidade, os dados foram gerados e divididos com \`random_state=42\`, enquanto os pesos da classe \`MLP\` foram inicializados com \`np.random.default_rng(seed)\`; neste experimento, \`seed=42\`. A comparação do Exercício 4 altera apenas essa semente para 42, 43 e 44, mantendo os demais parâmetros. A acurácia no teste foi de **\*\*92.50%\*\***.



![Figura 2: Perda binária]\(figures/fig2.png)



A Figura 2 apresenta a redução da entropia cruzada durante o treinamento.



![Figura 3: Fronteira de decisão]\(figures/fig3.png)



A Figura 3 apresenta a fronteira de decisão não linear encontrada pelo MLP sobre os pontos de teste.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/mlp_core.py"



\`\`\`



O trecho a seguir realiza o treinamento, mede a acurácia e gera as Figuras 2 e 3.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex2c.py"



\`\`\`



**### D — Análise**



A regressão logística ficou abaixo de 50% porque tenta separar as classes com apenas uma reta. Entretanto, a Classe 1 ocupa duas regiões distantes, em lados opostos da Classe 0. Na Figura 3, o MLP aprende uma região central predominantemente associada à Classe 0 e regiões externas associadas à Classe 1. Essa separação exige uma fronteira não linear. A camada oculta combina ativações de vários neurônios e permite que o MLP represente essa geometria, alcançando **\*\*92.50%\*\***, contra **\*\*47.50%\*\*** da linha de base.



**## Exercício 3**



**### A — Gere os dados**



Foram geradas **\*\*1500 amostras\*\***, com **\*\*4 features informativas\*\***, **\*\*3 classes\*\*** e **\*\*2 grupos por classe\*\***, usando \`make_classification\`, \`class_sep=1.0\` e \`random_state=42\`. A divisão treino/teste foi de 80/20, com estratificação.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex3a.py"



\`\`\`



**### B — Adapte a rede**



A implementação central \`mlp_core.py\` é a mesma utilizada no Exercício 2. Não foi necessário criar outra classe nem modificar o algoritmo de retropropagação ou atualização dos parâmetros: o argumento \`task\` seleciona a saída sigmoide com entropia cruzada binária (\`binary\`) ou a saída softmax com entropia cruzada categórica (\`multi\`). Dessa forma, a mesma estrutura de camadas e treinamento é reutilizada nos dois problemas. A arquitetura agora possui três saídas, com softmax e entropia cruzada categórica:



$$p_k=*\\*&#x66;rac{e^{u_k}}{\sum_j e^{u_j}},\qquad L=-\sum_k y_k\log p_k.$$



Para derivar o gradiente, usamos $\log p_k=u_k-\log\sum_j e^{u_j}$, então



$$L=-\sum_k y_ku_k+\left(\sum_k y_k\right)\log\sum_j e^{u_j}.$$



Como os rótulos são one-hot, $\sum_k y_k=1$. Derivando em relação ao logit $u_j$:



$$*\\*&#x66;rac{\partial L}{\partial u_j}=-y_j+*\\*&#x66;rac{e^{u_j}}{\sum_k e^{u_k}}=p_j-y_j.$$



Para a perda média sobre $N$ amostras, o delta utilizado no código é **\*\*$(P-Y)/N$\*\***. Assim, a mesma retropropagação funciona para os dois tipos de classificação, mudando apenas a função de saída e a perda.



**### C — Treine e avalie**



A arquitetura foi **\*\*[4, 24, 3]\*\***, com uma camada oculta de 24 neurônios, \`tanh\`, **\*\*taxa de aprendizado 0.2\*\***, **\*\*2200 épocas\*\***, treinamento em lote completo e semente 42. A acurácia no teste foi de **\*\*85.67%\*\***.



![Figura 4: Perda multiclasse]\(figures/fig4.png)



A Figura 4 mostra a redução da perda de treino no problema multiclasse.



![Figura 5: Matriz de confusão]\(figures/fig5.png)



A matriz de confusão do conjunto de teste foi:



| Classe real / prevista | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 87 | 6 | 7 |
| 1 | 10 | 81 | 9 |
| 2 | 5 | 6 | 89 |



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex3c.py"



\`\`\`



**### D — Análise**



O par mais confundido foi Classes 0 e 1, com 16 erros bidirecionais (6 amostras da Classe 0 classificadas como 1 e 10 da Classe 1 classificadas como 0). Como o conjunto possui quatro dimensões informativas e dois grupos por classe, as classes não correspondem necessariamente a regiões simples e compactas. Grupos de classes diferentes podem se aproximar ou se sobrepor no espaço de quatro dimensões, dificultando a separação. A regressão logística obteve **\*\*66.33%\*\***, enquanto o MLP alcançou **\*\*85.67%\*\***: a camada oculta acrescentou a capacidade de representar fronteiras não lineares entre esses grupos.



**## Exercício 4**



**### A — Treine a rede mais profunda**



Foi utilizada a arquitetura **\*\*[4, 24, 24, 3]\*\***, com **\*\*duas camadas ocultas\*\*** de 24 neurônios cada. Foram mantidos os mesmos dados, divisão treino/teste, ativação, taxa de aprendizado **\*\*0.2\*\***, **\*\*2200 épocas\*\*** e semente 42 do Exercício 3. A acurácia no teste foi **\*\*84.00%\*\***.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex4a.py"



\`\`\`



**### B — Compare**



![Figura 6: Comparação das perdas]\(figures/fig6.png)



A Figura 6 compara as perdas de treinamento das duas arquiteturas sob o mesmo orçamento de épocas.



| Modelo | Camadas ocultas | Parâmetros | Perda final de treino (semente 42) | Acurácia no teste (semente 42) |
|---|---:|---:|---:|---:|
| Exercício 3 — [4,24,3] | 1 | 195 | 0.269152 | 85.67% |
| Exercício 4 — [4,24,24,3] | 2 | 795 | 0.204812 | 84.00% |



Para verificar se a diferença depende da inicialização, foram realizadas três execuções com sementes diferentes, mantendo os demais hiperparâmetros fixos:



| Semente | Rede rasa | Rede profunda |
|---|---:|---:|
| 42 | 85.67% | 86.67% |
| 43 | 84.67% | 87.00% |
| 44 | 87.33% | 84.00% |
| **Média** | **85.89%** | **85.89%** |



As duas arquiteturas apresentaram a mesma acurácia média de 85.89% nas três sementes, resultando em uma diferença média de 0.00 ponto percentual. Entretanto, o desempenho variou entre as inicializações, mostrando que a comparação com apenas uma semente poderia levar a conclusões diferentes.



\`\`\`python



\--8<-- "docs/exercises/mlp/code/ex4b.py"



\`\`\`



**### C — Análise**



A adição de uma segunda camada oculta não trouxe uma melhoria consistente na acurácia de teste. Embora a rede profunda tenha apresentado menor perda de treinamento, as duas arquiteturas obtiveram a mesma acurácia média de **85.89%** nas três sementes analisadas. Os resultados individuais variaram conforme a inicialização: a rede profunda foi superior nas sementes 42 e 43, enquanto a rede rasa apresentou melhor desempenho na semente 44. Isso mostra a importância de comparar múltiplas inicializações, em vez de tirar conclusões a partir de uma única execução.

A rede rasa possui **195 parâmetros**, enquanto a rede profunda possui **795 parâmetros**. Neste conjunto de dados, com seis grupos distribuídos entre três classes, uma camada oculta de 24 neurônios já parece oferecer capacidade suficiente para representar as fronteiras de decisão. A segunda camada aumenta a capacidade do modelo, mas também sua complexidade, sem apresentar ganho médio de acurácia neste experimento. Redes mais profundas podem ser vantajosas em problemas com padrões hierárquicos mais complexos, como imagens, nos quais diferentes camadas aprendem representações sucessivamente mais abstratas.



**## Resumo dos resultados**



| # | Quantidade | Valor |
|---|---|---|
| 1 | Ex. 1 — $z^{(1)}$ e $h^{(1)}$ | $[0.270000,-0.180000]$ e $[0.263625,-0.178081]$ |
| 2 | Ex. 1 — $u^{(2)}$, $\hat y$ e $L$ | 0.385237; 0.367247; 0.400377 |
| 3 | Ex. 1 — oito parâmetros atualizados | $W^{(1)}=[[0.376405,-0.130562],[0.152295,0.419082]]$; $b^{(1)}=[0.252811,-0.295410]$; $W^{(2)}=[0.586587,-0.358490]$; $b^{(2)}=0.528448$ |
| 4 | Ex. 2 — linha de base linear | 47.50% |
| 5 | Ex. 2 — arquitetura e acurácia | [2,16,1]; 92.50% |
| 6 | Ex. 3 — arquitetura e acurácia | [4,24,3]; 85.67% |
| 7 | Ex. 3 — par mais confundido | Classes 0 e 1; 16 erros |
| 8 | Ex. 4 — arquitetura e acurácia | [4,24,24,3]; 84.00% |
| 9 | Ex. 4 — diferença média de acurácia (3 sementes) | 0.00 ponto percentual (profunda − rasa) |
