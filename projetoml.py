# Advertisement - Click on Ad dataset

# O marketing digital se mostra como uma ferramenta eficaz para alcançar diferentes clientes e atingir o objetivo principal de qualquer produto : a venda.

#De fato, o uso de anúncios se apresenta extremamente imporntante, uma vez que, o mesmo é essencial para :

#Alcance de público-alvo
#Mensuralidade e Flexibilidade
#Inovação e criativade
#Segmentação Geográfica e Retenção de Clientes
#Fidelização do clientes
#O estudo e a análise da base de dados Advertising mostra como os 
# dados de clientes tem impacto na eficácia dos anúncios. 
# É através dos ad's que muitas estudam se suas campanhas denotam-se eficazes ou não.

#Abaixo tem-se um estudo a cerca da base de dados e quais insights pode-se retirar da mesma.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('advertising_full.csv')
df.head(10)
df.tail(10)
df.info()

df.shape
df.dtypes
df.isnull().sum()

df.describe().T

cor1 = (108 / 255, 108 / 255, 100 / 255)
my_palette = sns.color_palette([cor1, '#6ecb98'])
sns.set_style("whitegrid", {'axes.grid' : False})
sns.pairplot(df, hue='Clicked on Ad', palette=my_palette)
plt.show()

#Percebe-se que a base de dados não possui nenhum valor faltante, assim como, também não possui valores
#"Não Números" (NaN). Além disso, tem-se alguns dados negativos que serão tratados futuramente para 
#não prejudicar a previsão do modelo.

# Tratamento dos atributos preditivos e análise das variáveis

#Clicked on Ad (target)

#- Compreende-se como a várival a ser predita. A variável "Clicked on Ad" é do tipo binária variando entre 0(não clicou) e 1(clicou), indicando se o indivíduo clicou ou não no anuncio.
#- 60770 Pessoas **Clicaram** no anúncio
#- 40680 Pessoas **Não Clicaram** no anúncio

df['Clicked on Ad'].value_counts()

fig = go.Figure()

cor1 = (108 / 255, 108 / 255, 100 / 255)
cor2 = (00/ 255, 80/ 255,52/255)
cores = [cor1, cor2]
plt.figure(figsize=(10,6))
ax = sns.countplot(x = df['Clicked on Ad'], palette = cores)
plt.title('Clique nos anúncios')
plt.ylabel('Quantidade')
plt.xlabel('Clique na propaganda')
plt.grid(False)
plt.gca().set_facecolor('white')
ax.set(xticklabels=["Não Clique (0)", "Clique (1)"])
for p in ax.patches:
  plt.annotate(int(p.get_height()), xy=(p.get_x()+0.25, p.get_height()), fontsize=10)
  
  

clicou = df.query('`Clicked on Ad` == 1')
nao_clicou = df.query('`Clicked on Ad` == 0')

clicou.describe().T

nao_clicou.describe().T

#Age

#- Por "Age" compreende-se a idade dos individuos presentes no dataframe, 
# englobando diferentes faixas etárias.
#- Os valores da idade variam de -51 a 114, sabendo que esses valores em sua prática real 
# são impossíveis (não existe idade de pessoas negativas, e valores acima de 100 anos 
# são praticamente ilusórios), os mesmos serão tratados para não prejudicar o modelo
#- Como o intuito é prever os cliques nos ADs, será definido uma idade para o grupo de interesse, 
# definida em um público adulto (18-60 anos)

df['Age'].describe()

summary = df['Age'].describe()
summary = summary.drop('count')
cor2 = (00/ 255, 80/ 255,52/255)
plt.bar(summary.index, summary.values,color=[cor2])
plt.xlabel('Estatísticas Descritivas')
plt.ylabel('Valores')
plt.grid(False)
plt.show()

df = df[(df['Age']>18) & (df['Age']<60)]  # Filtro na aplicado na dataframe pela coluna 'Age' para trazer somente as idade entre 18 e 60 anos.
cor =(00/ 255, 80/ 255,52/255)
df['Age'].plot(kind='hist', bins = 30, color= cor )
plt.ylabel('Frequência')
plt.xlabel('Idade')
plt.title('Distribuição da Frequência da Idade')
plt.gca().grid(False)
plt.show()

plt.boxplot(df['Age'])
plt.gca().grid(False)
plt.title('Idade')
plt.show()

#Clicked on Ad (Target)

#Como houve uma seleção na idade, pode ter alterado a proporção dos dados do Target

df['Clicked on Ad'].value_counts(normalize= True)

cor1 = (108 / 255, 108 / 255, 100 / 255)
cor2 = (00/ 255, 80/ 255,52/255)
cores = [cor1, cor2]
plt.figure(figsize=(10,6))
ax = sns.countplot(x = df['Clicked on Ad'], palette = cores)
plt.title('Clique nos anúncios')
plt.ylabel('Quantidade')
plt.xlabel('Clique na propaganda')
plt.grid(False)
plt.gca().set_facecolor('white')
for p in ax.patches:
  plt.annotate(int(p.get_height()), xy=(p.get_x()+0.25, p.get_height()), fontsize=10)


# Daily Time Spent on Site
#- O daily time spent se mostra como o tempo que o indivíduo passou em um determinado em um site, 
#variando esse tempo de 0 a 178.4, tendo uma maior quantidade de acesso variando de 51.36 a 78.5475

df['Daily Time Spent on Site'].describe().T

df = df[df['Daily Time Spent on Site']>=0]

sns.set(style="darkgrid")
cor = (00/ 255, 80/ 255,52/255)
sns.kdeplot(df['Daily Time Spent on Site'],shade = True, color= cor)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.title('Tempo no site')
plt.ylabel('Densidade')
plt.xlabel('Minutos')
plt.show()

## Area Income

#- Compreende-se como a renda da área em que o consumidor se encontra
# Há dados negativos que serão tratados, assim como os outliers

df['Area Income'].describe()

df= df[df['Area Income']>=0] # filtro na coluna 'Area Income' filtrando apenas valores maiores que 0, eliminando assim os negativos.

def get_limits(data_variable):
    q1=data_variable.quantile(0.25)
    q3=data_variable.quantile(0.75)
    iqr=q3-q1
    lim_sup=q3+1.5*iqr
    lim_inf=q1-1.5*iqr
    return (lim_inf,lim_sup)

limites_AreaIncome = get_limits(df['Area Income'])
lim_inf = limites_AreaIncome[0]
lim_sup = limites_AreaIncome[1]
lim_inf, lim_sup

df = df.drop((df[df['Area Income'] > lim_sup].index) | (df[df['Area Income'] < lim_inf].index))

sns.set(style="darkgrid")
cor = (00/ 255, 80/ 255,52/255)
sns.kdeplot(df['Area Income'],shade = True, color= cor)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.title('Renda por Região')
plt.ylabel('Densidade')
plt.xlabel('Dólares')
plt.show()

## Daily Internet Usage

#- Compreende-se como o tempo que o indivíduo está utilizando a internet

df['Daily Internet Usage'].describe()

sns.set(style="darkgrid")
cor = (00/ 255, 80/ 255,52/255)
sns.kdeplot(df['Daily Internet Usage'],shade = True, color= cor)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.title('Tempo de uso de internet')
plt.ylabel('Densidade')
plt.xlabel('Minutos')
plt.show()

df = df[df['Daily Internet Usage']>=0]
cor = (00/ 255, 80/ 255,52/255)
sns.kdeplot(df['Daily Internet Usage'],shade = True, color= cor)
plt.gca().set_facecolor('white')
plt.grid(False)
plt.title('Tempo de uso de internet')
plt.ylabel('Densidade')
plt.xlabel('Minutos')
plt.show()

# Ad Topic Line
#- Compreende-se como o título do anúncio que está em vigor. Para essa variável, 
# por se tratar de 87849 títulos, foi feita uma núvem de palavras para identificar quais 
# as palavras que mais se repetiam nos títulos dos anúncios.

df['Ad Topic Line']

df['Ad Topic Line'].value_counts()

from wordcloud import WordCloud

df['Ad Topic Line'] = df['Ad Topic Line'].astype(str)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap = "viridis").generate(' '.join(df['Ad Topic Line']))


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Nuvem de Palavras - Linha de Assunto do Anúncio')
plt.show()



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')

texto_completo = ' '.join(df['Ad Topic Line'].astype(str))

palavras = word_tokenize(texto_completo.lower())

stop_words = set(stopwords.words('english'))
palavras = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]

frequencia_palavras = Counter(palavras)
df_frequencia_palavras = pd.DataFrame(frequencia_palavras.most_common(20), columns=['Palavra', 'Frequência'])
cor = (00/ 255, 80/ 255,52/255)
plt.figure(figsize=(12, 6))
plt.barh(df_frequencia_palavras['Palavra'], df_frequencia_palavras['Frequência'], color=cor)
plt.title('Palavras mais frequentes na coluna "Ad Topic Line"')
plt.xlabel('Frequência')
plt.ylabel('Palavra')

for index, value in enumerate(df_frequencia_palavras['Frequência']):
    plt.text(value + 5, index, str(value), ha='left', va='center')

plt.grid(False)
plt.gca().set_facecolor('white')
plt.tight_layout()
plt.show()


# City

#- Se mostra como as cidades que o consumidor se encontra

city_counts = df['City'].value_counts()
top_20_citys = city_counts.head(20)

df['City'].value_counts()
contagem_cidades = df['City'].value_counts().head(20)
contagem_cidades = contagem_cidades.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
contagem_cidades.plot(kind='barh', color=cor)
plt.title('As 20 cidades mais frequentes')
plt.xlabel('Contagem')
plt.ylabel('Cidade')
plt.xticks(rotation=45)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.tight_layout()
plt.show()

# Male
#- Compreende-se como o sexo do consumidor
#- 0 para Mulheres, sendo **45328** Mulheres
#- 1 para Homens, sendo **44852** Homens

df= df.rename(columns = {'Male':'Sex'})
df

df['Sex'].value_counts()

plt.figure(figsize=(7, 7))
cor1 = (108 / 255, 108 / 255, 100 / 255)
cor2 = (00/ 255, 80/ 255,52/255)
cores = [cor1, cor2]
ax = sns.countplot(x=df['Sex'], palette = cores)
plt.title('Gênero dos consumidores')
plt.xlabel('Gênero')
plt.ylabel('Quantidade')
ax.set(xticklabels=["Mulher (0)", "Homem (1)"])
for p in ax.patches:
    plt.annotate(p.get_height(), xy=(p.get_x() + 0.3, p.get_height()), fontsize=10)
    plt.gca().set_facecolor('white')
plt.grid(False)
plt.show()


# Country
#- Se mostra quais são os países dos consumidores

contagem_paises = df['Country'].value_counts().head(20)

contagem_paises = contagem_paises.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
contagem_paises.plot(kind='barh', color=cor)
plt.title('Os 20 paises mais frequentes')
plt.xlabel('Contagem')
plt.ylabel('Paises')
plt.xticks(rotation=45)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.tight_layout()
plt.show()



## Timestamp
#- Se apresenta como a hora em que o consumidor clicou no anúncio ou fechou a janela

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Month'] = df['Timestamp'].dt.month
df['Month']

data_count = df['Month'].value_counts()

data = {'Mês': [1, 2, 3, 4, 5, 6, 7],
        'Contagem': [147, 160, 156, 147, 147, 142, 101]}
df_data = pd.DataFrame({'Mês': ['1', '2', '3', '4', '5', '6', '7'], 'Contagem': (data_count).values})
df_data


plt.figure(figsize=(10, 6))
plt.bar(df_data['Mês'], df_data['Contagem'], color=cor)
plt.title('Contagem de Meses')
plt.xlabel('Mês')
plt.ylabel('Contagem')
plt.grid(False)
plt.gca().set_facecolor('white')
plt.show()



# Matriz de Correlação

#-  É uma tabela que mostra as correlações entre várias variáveis ou conjuntos de dados, 
#fornecendo uma visão geral das relações lineares entre as variáveis, destacando a intensidade e a 
#direção dessas relações, a mesma se apresenta uma excelente maneira de analisar como váriaveis estão 
#relacionadas e se a as mesmas tem influência no desfecho do atributo alvo a ser predito.

plt.figure(figsize = (14,10))
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95, reverse=True, as_cmap=True)
sns.heatmap(df.corr(), annot=True, fmt=".3f", linewidths=0.7, cmap=cmap)
plt.show()


my_palette = sns.color_palette([cor1, '#6ecb98'])
sns.pairplot(df, hue='Clicked on Ad', palette=my_palette)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.show()
