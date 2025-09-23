# Introdução
Este projeto tem como objetivo investigar a performance do classificador **SVM (Support Vector Machine)** aplicado ao conjunto de dados **Breast Cancer Wisconsin**. A análise concentra-se no efeito das estratégias de validação cruzada estratificada sobre as estimativas de desempenho, comparando dois esquemas: **StratifiedKFold com K=10** e **StratifiedKFold com K=12**. O foco é verificar se a escolha do número de dobras altera de maneira significativa métricas como acurácia média e composição da matriz de confusão.

O conjunto de dados utilizado possui **569 amostras e 30 atributos numéricos**, e foi obtido do repositório público Kaggle. No fluxo experimental, os dados foram pré-processados, incluindo a remoção de colunas não informativas e a codificação dos rótulos em formato binário, divididos em treino/teste com partição estratificada (80/20) e submetidos a ajuste de hiperparâmetros via **GridSearchCV** para combinações de **kernel**, **C** e **gamma**. Foram então comparadas as métricas médias obtidas com StratifiedKFold-10 e StratifiedKFold-12, registrando-se tanto as acurácias médias quanto as matrizes de confusão correspondentes.

* Investigar a performance do classificador SVM aplicado ao conjunto de dados Breast Cancer Wisconsin.
* Comparar diferentes estratégias de validação cruzada estratificada, avaliando o impacto do número de dobras (K=10 versus K=12) sobre as métricas de desempenho.
* Ajustar hiperparâmetros do modelo SVM (kernel, C e gamma) utilizando GridSearchCV para identificar combinações mais eficazes.
* Avaliar métricas de classificação como acurácia média, matriz de confusão e ROC-AUC para compreender o comportamento do modelo.
***
## Antes de começar

### Executando este notebook online
Este projeto foi inteiramente desenvolvido no Google Colab.
Mesmo o arquvio ipynb estando neste repositório, recomendamos que você também o execute por lá para reproduzir os resultados de uma maneira mais fiel a nossa experiência.
* <a href="LINK_DO_NOTEBOOK_NO_COLAB" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

⚠ _O Colab oferece um ambiente temporário: tudo o que você fizer será excluído depois de um tempo, portanto, certifique-se de salvar todos os dados que lhe interessam._

<details>

Outros serviços que também podem funcionar, mas não foram testados completamente, além de que certas etapas como o upload do dataset não foram documentadas para estes serviços:

* <a href="LINK_KAGGLE"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>

* <a href="LINK_BINDER"><img src="https://mybinder.org/badge_logo.svg" alt="Launch binder" /></a>

* <a href="LINK_DEEPNOTE"><img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg" alt="Launch in Deepnote" /></a>
</details>

***
## Fluxo de Trabalho do Modelo
![alttext](AQUI_VAI_O_LINK_DO_DIAGRAMAraw=true)

#  Organização do Notebook  

## 1. Imports e Configurações Globais  

Utilizamos bibliotecas amplamente conhecidas. Cada uma tem um papel específico no pipeline:  

- **`from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV`**  
  - `train_test_split`: realiza a divisão do conjunto de dados em treino e teste.  
  - `StratifiedKFold`: cria partições estratificadas para validação cruzada, mantendo a proporção das classes em cada fold.  
  - `cross_val_score`: executa validação cruzada de maneira simplificada para avaliação rápida.  
  - `GridSearchCV`: faz busca exaustiva em uma grade de hiperparâmetros, avaliando cada combinação por validação cruzada.  

- **`from sklearn.svm import SVC`**  
  - Importa o modelo de máquina de vetores de suporte para classificação (Support Vector Classifier).  

- **`from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay`**  
  - `accuracy_score`: calcula a acurácia do modelo (proporção de acertos).  
  - `ConfusionMatrixDisplay`: exibe visualmente a matriz de confusão para facilitar a interpretação dos erros.  

- **`import pandas as pd`**  
  - Biblioteca para manipulação e análise de dados em estruturas como DataFrames.  

- **`import numpy as np`**  
  - Suporte para operações numéricas vetorizadas e manipulação de arrays.  

- **`import matplotlib.pyplot as plt`**  
  - Ferramenta para criação de gráficos e visualizações básicas.   

- **`%matplotlib inline`**  
  - Comando que faz com que os gráficos do Matplotlib sejam exibidos dentro do notebook.  

Além dos imports, são definidas variáveis globais para manter consistência ao longo de todo o notebook.  
A variável **`scale = 0`** foi configurada para facilitar a definição dos valores de `gamma` durante os testes.  
A constante **`RANDOM_STATE = 42`** garante reprodutibilidade dos resultados, fixando a semente do gerador aleatório.  
Por fim, são instanciados dois objetos de validação cruzada estratificada: **`cv_KFold10`** e **`cv_KFold12`**, que criam partições estratificadas em 10 e 12 folds, respectivamente, permitindo comparar o impacto do número de dobras na avaliação do modelo.  
 
## 2. Upload e Visualização Inicial do Dataset  

Este projeto utiliza o dataset *Breast Cancer Wisconsin*.  
No Google Colab, é possível carregar o arquivo CSV diretamente do seu computador, o dataset está presente neste reposítorio.  

1. Clique no ícone de pasta no painel lateral do Colab.  
![Print1 - Abrir painel de arquivos](https://github.com/leo-vasi/breast-cancer-wisconsin-ml/blob/master/img/upload_dataset1.png)
2. Clique em “Upload” e selecione o arquivo `Breast_Cancer_Wisconsin.csv` do seu computador.   
![Print2 - Botão Upload](https://github.com/leo-vasi/breast-cancer-wisconsin-ml/blob/master/img/upload_dataset2.png)
3. Após o upload, o arquivo aparecerá na lista de arquivos disponíveis.  
![Print3 - Arquivo carregado no Colab](https://github.com/leo-vasi/breast-cancer-wisconsin-ml/blob/master/img/upload_dataset3.png)  

Depois disso, execute o bloco abaixo para ler o arquivo e visualizar as primeiras linhas:  

```python
bc = pd.read_csv('Breast_Cancer_Wisconsin.csv')
bc.head()
```
## 3. Tratamento do Dataset  

Antes do pré-processamento, algumas colunas não relevantes foram removidas para garantir que o modelo trabalhasse apenas com os **30 atributos numéricos significativos**.  

- **`id`**: identificador do paciente, não influencia no diagnóstico.  
- **`Unnamed: 32`**: coluna vazia (resíduo do CSV).  

Essa limpeza evita que informações irrelevantes ou nulas interfiram no treinamento do modelo.  

> *Exemplo de código:*  
> ```python
> bc.drop(['Unnamed: 32','id'], axis=1, inplace=True)
> ```

---
## 4. Pré-processamento de Dados (Parte 1)  

Em seguida, os dados foram organizados em:  

- **`X`**: todos os atributos numéricos (variáveis independentes).  
- **`y`**: coluna `diagnosis`, contendo os rótulos **Maligno (M)** ou **Benigno (B)** (variável-alvo).  

Separar atributos e alvo é essencial para alimentar corretamente os algoritmos de aprendizado.  

> *Exemplo de código:*  
> ```python
> X = bc.loc[:, bc.columns != "diagnosis"]
> y = bc.loc[:, "diagnosis"]
> ```

---
## 5. Pré-processamento de Dados (Parte 2 – Divisão do Treino)  

O conjunto de dados foi dividido em **80% para treino** e **20% para teste** usando `train_test_split` com o parâmetro `stratify=y`.  
A estratificação garante que a proporção entre classes (M e B) seja mantida em ambos os conjuntos, evitando que um deles fique desbalanceado.  

Como os rótulos estavam em formato de string, foi usado `pd.factorize()` para convertê-los em inteiros (0 e 1), facilitando o uso de funções como `np.bincount` e garantindo compatibilidade com algoritmos de aprendizado.  

Ao final desta etapa, são exibidas:  
- **As dimensões de treino e teste** (quantidade de amostras e atributos).  
- **A distribuição das classes** em cada conjunto, validando que a separação representativa foi respeitada.  

> *Exemplo de código resumido:*  
> ```python
> X_train, X_test, y_train, y_test = train_test_split(
>     X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
>
> y_train_num, _ = pd.factorize(y_train)
> y_test_num, _ = pd.factorize(y_test)
> ```

## 6. Processamento de Dados  
Detalhamos as transformações finais e operações realizadas nos dados antes do início dos treinamentos, garantindo que as entradas estejam no formato correto.  

## 7. Treino de Dados com Kernel Linear  
Mostramos o treinamento do modelo SVM utilizando Kernel Linear, explicando as vantagens desse kernel em diferentes cenários.  

## 8. Treino de Dados com Kernel Não-linear (RBF)  
Apresentamos o treinamento do modelo SVM utilizando o Kernel RBF, destacando suas diferenças em relação ao Kernel Linear e quando seu uso é indicado.  

## 9. StratifiedKFold (K=10) com GridSearchCV – Apenas Kernel (SKF-10)  
Nesta etapa aplicamos validação cruzada estratificada com 10 folds, ajustando apenas o Kernel para encontrar melhores resultados.  

## 10. StratifiedKFold (K=10) com GridSearchCV – Kernel, C e Gamma (SKF-10)  
Expandimos a busca para incluir também os parâmetros C e Gamma, explicando o impacto dessas variações no desempenho do modelo.  

## 11. Varredura de Parâmetros (C e Gamma) – SKF-10  
Mostramos os resultados obtidos ao variar os parâmetros C e Gamma com validação estratificada em 10 folds.  

## 12. StratifiedKFold (K=12) com GridSearchCV – Apenas Kernel (SKF-12)  
Repetimos o processo anterior, agora utilizando 12 folds para validar o impacto do aumento no número de partições.  

## 13. StratifiedKFold (K=12) com GridSearchCV – Kernel, C e Gamma (SKF-12)  
Aplicamos novamente GridSearchCV variando Kernel, C e Gamma, explicando as diferenças encontradas em relação ao teste com 10 folds.  

## 14. Varredura de Parâmetros – StratifiedKFold (K=12)  
Por fim, mostramos a variação dos parâmetros com validação estratificada em 12 folds, consolidando os resultados para análise final.  
