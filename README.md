# Introdução
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ac lacus ligula. Nullam rutrum auctor tortor eu ullamcorper. Mauris efficitur ligula ut arcu blandit congue. Aenean iaculis imperdiet ultrices. Proin tempus nisi non dui vulputate, in fermentum turpis hendrerit. Fusce hendrerit porta commodo. Nunc volutpat erat dui, in placerat enim congue nec. Praesent at libero iaculis, rutrum elit vel, interdum leo. Nunc hendrerit in dui at gravida. Ut scelerisque rutrum risus efficitur vulputate. Praesent molestie ultrices nisi, at mollis velit ullamcorper sit amet.

Sed luctus facilisis nisl, ac elementum mauris luctus id. Pellentesque fermentum pellentesque posuere. Quisque sit amet semper odio. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nulla vitae scelerisque odio. Cras vestibulum ornare dolor, vitae convallis mi. Aliquam finibus libero vitae tincidunt commodo. Quisque vel mollis erat. Maecenas dictum lectus et mauris feugiat, at maximus neque finibus. Pellentesque feugiat consectetur arcu vel cursus. Nullam malesuada, arcu ut varius molestie, magna enim euismod urna, vel gravida leo quam sit amet orci. Fusce pharetra tristique lacus luctus vulputate. Aliquam nisl massa, rhoncus sit amet lorem a, consequat consectetur diam. In hac habitasse platea dictumst. In vel augue sit amet nisl posuere tincidunt in fermentum risus. Nunc volutpat nisl nibh, sed scelerisque mauris cursus volutpat.

* Sed risus dui, vehicula quis hendrerit non, pretium a felis. Lorem ipsum dolor sit amet, consectetur adipiscing elit
* Curabitur porta urna dapibus, efficitur neque et, blandit felis
* Praesent sollicitudin laoreet mauris at iaculis.
* Praesent dolor orci, volutpat sed ante et, tempus placerat augue.
* 
***
## Antes de começar

### Executando este notebook online
Este projeto foi inteiramente desenvolvido no Google Colab.
Mesmo o arquvio ipynb estando neste repositório, recomendamos que você também o execute por lá para reproduzir os resultados de uma maneira mais fiel a nossa experiência.
* <a href="LINK_DO_NOTEBOOK_NO_COLAB" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

⚠ _O Colab oferece um ambiente temporário: tudo o que você fizer será excluído depois de um tempo, portanto, certifique-se de baixar todos os dados que lhe interessam._

<details>

Outros serviços que também podem funcionar, mas não foram testados completamente:

* <a href="LINK_KAGGLE"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>

* <a href="LINK_BINDER"><img src="https://mybinder.org/badge_logo.svg" alt="Launch binder" /></a>

* <a href="LINK_DEEPNOTE"><img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg" alt="Launch in Deepnote" /></a>
</details>

***
## Fluxo de Trabalho do Modelo
![alttext](AQUI_VAI_O_LINK_DO_DIAGRAMAraw=true)

#  Organização do Notebook  

## 1. Imports e Configurações Globais  
Nesta seção são apresentados os pacotes e bibliotecas necessários para a execução do notebook, bem como configurações globais importantes para manter o ambiente padronizado e funcional.  

## 2. Upload e Visualização Inicial do Dataset  
Aqui mostramos como realizar o upload do dataset para o Google Colab e sua visualização inicial, incluindo instruções passo a passo e prints ilustrativos para auxiliar novos usuários.  

## 3. Tratamento do Dataset  
Explicamos quais modificações foram realizadas no dataset antes da manipulação, descrevendo a justificativa para cada alteração para garantir a consistência dos dados.  

## 4. Pré-processamento de Dados (Parte 1)  
Descrevemos as técnicas iniciais aplicadas para preparar os dados para análise, como limpeza, codificação e normalização.  

## 5. Pré-processamento de Dados (Parte 2 – Divisão do Treino)  
Apresentamos o processo de divisão do conjunto de dados entre treino e teste, explicando a importância dessa etapa para a avaliação de modelos.  

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
