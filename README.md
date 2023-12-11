
## Propagandas e Cliques

O projeto final do curso Quero Ser Data Analytics em parceria com a ADA e a Raia Drogasil, traz uma análise completa sobre uma base de dados que diz respeito a propagandas e cliques em um determinado site da internet.
## Apêndice

Nosso projeto foi composto, primeiramente, pela formulação do problema e geração de contexto. Sobre esse aspecto, podemos destacar a necessidade de entender qual resultado traria insights valiosos para a área de negócios.
Decidimos por seguir pelo caminho de: tal e tal.

Em seguida, pudemos construir uma análise exploratória dos dados da nossa base, em termos de processamento de visualização, quanto na parte de estatística e ciclo de modelagem.

Buscamos na apresentação final, definir quais os aspectos da discussão fariam sentido para que a área de negócio fosse impactada positivamente com os insights, ao levar perguntas e respostas que obtivemos com as análises e contraposições feitas.

## Documentação de cores

| Cor               | Hexadecimal                                                |
| ----------------- | ---------------------------------------------------------------- |
| Cor 1       | ![#237a6f](https://via.placeholder.com/10/237a6f?text=+) #237a6f |
| Cor 2    | ![#ffffff](https://via.placeholder.com/10/ffffff?text=+) #ffffff |
| Cor 3     | ![#73c294](https://via.placeholder.com/10/73c294?text=+) #73c294 |
| Cor 4      | ![#2bc8b4](https://via.placeholder.com/10/2bc8b4?text=+) #2bc8b4 |


## Stack utilizada

**Linguagem:** Python

**Bibliotecas:** NumPy, Pandas, SeaBorn, MatPlotLib

**Modelos:** Regressão Logística, Árvore de Decisão, KNN e SVM.



## Roadmap

- Nosso DataFrame possui mais de 100 mil linhas

- Utilizamos um parâmetro que diminui cerca de 6% da base, baseado na idade de 18-60 anos.

- Está na pasta "Dataset" o DataFrame completo e também o com o tratamento de dados.

- Além do tratamento que está em "TratamentoDados", fizemos avaliações estatísticas com o nosso DataFrame, onde buscavámos índices interessantes e insights para o clique ou não clique em uma propaganda, essa parte está em "projetoml_estat".

- Utilizamos também séries temporais e partimos para nosso modelo de Machine Learning. Você pode encontrar as análises dos Hiperparâmetros, Modelos e Matriz de Confusão na pasta "Modelo de Machine Learning".

- O melhor modelo a ser utilizado foi "Regressão Logística".



## Documentação

[Seaborn](https://seaborn.pydata.org/)

[Matplotlib](https://matplotlib.org/stable/users/index.html)


## FAQ

#### Como uso o data frame?

- Primeiro passo é utilizar o arquivo na pasta "Dataset". Logo em seguida, seria interessante já abrir o CSV do arquivo com a função:

```http
 df = pd.read_csv('advertising_full.csv')
```
- Utilizamos pandas e importamos essas bibliotecas antes de visualizar o DF.

#### Qual o principal modelo?

- O principal modelo utilizado em nossas análises foi a Regressão Logística. Pode ser acessado com o comando:

```http
modelo = joblib.load(modelo_regressao_logistica.joblib)
```


## Autores

- [@nataliagiardini](https://www.github.com/nataliagiardini)
- [@rodrigueslucelia](https://www.github.com/rodrigueslucelia)
- [@pandolfiz](https://github.com/Pandolfiz)
- [@coleone7](https://github.com/coleone7)
