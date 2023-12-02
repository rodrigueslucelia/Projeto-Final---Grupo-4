import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("advertising_full.csv")

df.head()

df.info()

df.shape

df.isnull().sum()

df.describe().T

cor1 = (108 / 255, 108 / 255, 100 / 255)
my_palette = sns.color_palette([cor1, '#6ecb98'])
sns.set_style("whitegrid", {'axes.grid' : False})
sns.pairplot(df, hue='Clicked on Ad', palette=my_palette)
plt.show()

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

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp'].dt.tz_localize('UTC')

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')

df['Clicou no anuncio'] = df['Clicked on Ad'].map({0: False, 1: True})
df.head()


#PARA COLUNA AGE

media_age = df['Age'].mean()
print("Média da Idade:", media_age)

mediana_age = df['Age'].median()
print("Mediana da idade", mediana_age)

minimo_age = df['Age'].min()
print("Minimo da idade", minimo_age)

maximo_age = df['Age'].max()
print("Maximo da idade", maximo_age)

desvio_padrao_age = df['Age'].std()
print("Desvio Padrão", desvio_padrao_age)

#PARA COLUNA AREA INCOME

media_areaincome = df['Area Income'].mean
print("Média da Renda Per Capita:", media_areaincome)

mediana_areaincome = df['Area Income'].median()
print("Mediana da Renda Per Capita", mediana_areaincome)

minimo_areaincome = df['Area Income'].min()
print("Minimo da Renda Per Capita", minimo_areaincome)

maximo_areaincome = df['Area Income'].max()
print("Máximo da Renda Per Capita", maximo_areaincome)

desvio_padrao_area = df['Area Income'].std()
print("Desvio Padrão", desvio_padrao_area)

#PARA DAILY INTERNET USAGE

media_internet = df['Daily Internet Usage'].mean
print("Média de Uso diário da internet:", media_internet)

mediana_internet = df['Daily Internet Usage'].median()
print("Mediana de Uso diário da internet ", mediana_internet)

minimo_internet = df['Daily Internet Usage'].min()
print("Minimo de uso diário da internet", minimo_internet)

maximo_internet = df['Daily Internet Usage'].max()
print("Maximo de uso diário da internet", maximo_internet)

desvio_padrao_internet = df['Daily Internet Usage'].std()
print("Desvio Padrão de uso diário da internet", desvio_padrao_internet)


#PARA AGE

col_age = 'Age'
sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

sns.histplot(df[col_age], kde=False, ax=axes[0])
axes[0].set_title('Histograma')

sns.kdeplot(df[col_age], ax=axes[1])
axes[1].set_title('Gráfico de Densidade')

sns.boxplot(x=df[col_age], ax=axes[2])
axes[2].set_title('Box Plot')

plt.tight_layout()
plt.show()

#PARA AREA INCOME

col_area_income = 'Area Income'
sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

sns.histplot(df[col_area_income], kde=False, ax=axes[0])
axes[0].set_title('Histograma')

sns.kdeplot(df[col_area_income], ax=axes[1])
axes[1].set_title('Gráfico de Densidade')

sns.boxplot(x=df[col_area_income], ax=axes[2])
axes[2].set_title('Box Plot')

plt.tight_layout()
plt.show()

#PARA DAILY INTERNET USAGE

col_internet = 'Daily Internet Usage'
sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

sns.histplot(df[col_internet], kde=False, ax=axes[0])
axes[0].set_title('Histograma')

sns.kdeplot(df[col_internet], ax=axes[1])
axes[1].set_title('Gráfico de Densidade')

sns.boxplot(x=df[col_internet], ax=axes[2])
axes[2].set_title('Box Plot')

plt.tight_layout()
plt.show()


plt.figure(figsize = (14,10))
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95, reverse=True, as_cmap=True)
sns.heatmap(df.corr(), annot=True, fmt=".3f", linewidths=0.7, cmap=cmap)
plt.show()

colunas_categoricas = ['Sex', 'Clicked on Ad']
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=len(colunas_categoricas), ncols=1, figsize=(10, 15))

for i, col in enumerate(colunas_categoricas):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'Distribuição de {col}')

    if len(df[col].unique()) > 10:
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

coluna_ad_topic_line = 'Ad Topic Line'
sns.set(style="whitegrid")

# Obter os 10 maiores valores
top_n = 10
top_valores = df[coluna_ad_topic_line].value_counts().nlargest(top_n)

plt.figure(figsize=(15, 8))
sns.barplot(x=top_valores.index, y=top_valores.values)
plt.xticks(rotation=45, ha='right')

plt.title(f'Distribuição dos {top_n} maiores valores em {coluna_ad_topic_line}')
plt.xlabel(coluna_ad_topic_line)
plt.ylabel('Contagem')

plt.show()

coluna_city = 'City'
sns.set(style="whitegrid")

# Obter os 10 maiores valores
top_n = 10
top_valores_city = df[coluna_city].value_counts().nlargest(top_n)

plt.figure(figsize=(15, 8))
sns.barplot(x=top_valores_city.index, y=top_valores_city.values)
plt.xticks(rotation=45, ha='right')

plt.title(f'Distribuição dos {top_n} maiores valores em {coluna_city}')
plt.xlabel(coluna_city)
plt.ylabel('Contagem')

plt.show()

coluna_country = 'Country'
sns.set(style="whitegrid")

# Obter os 10 maiores valores
top_n = 10
top_valores_country = df[coluna_country].value_counts().nlargest(top_n)

plt.figure(figsize=(15, 8))
sns.barplot(x=top_valores_country.index, y=top_valores_country.values)
plt.xticks(rotation=45, ha='right')

plt.title(f'Distribuição dos {top_n} maiores valores em {coluna_country}')
plt.xlabel(coluna_country)
plt.ylabel('Contagem')


plt.show()

coluna_idade = 'Age'
coluna_tempo_diario = 'Daily Time Spent on Site'
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=coluna_idade, y=coluna_tempo_diario, data=df)

plt.title(f'Relação entre {coluna_idade} e {coluna_tempo_diario}')
plt.xlabel(coluna_idade)
plt.ylabel(coluna_tempo_diario)

plt.show()

correlacao_idade_tempo_diario = df[coluna_idade].corr(df[coluna_tempo_diario])
print(f"Correlação entre {coluna_idade} e {coluna_tempo_diario}: {correlacao_idade_tempo_diario}")

coluna_genero = 'Sex'
coluna_tempo_diario = 'Daily Time Spent on Site'

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x=coluna_genero, y=coluna_tempo_diario, data=df)

plt.title(f'Diferença no {coluna_tempo_diario} entre os Gêneros')
plt.xlabel('Gênero')
plt.ylabel(coluna_tempo_diario)

plt.show()

coluna_idade = 'Age'
coluna_clicou_no_anuncio = 'Clicked on Ad'

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df[df[coluna_clicou_no_anuncio] == 1][coluna_idade], bins=20, kde=True)
plt.title('Distribuição de Idade - Clicaram no Anúncio')

plt.subplot(1, 2, 2)
sns.histplot(df[df[coluna_clicou_no_anuncio] == 0][coluna_idade], bins=20, kde=True)
plt.title('Distribuição de Idade - Não Clicaram no Anúncio')

plt.tight_layout()
plt.show()


coluna_clicou_no_anuncio = 'Clicked on Ad'
coluna_tempo_diario = 'Daily Time Spent on Site'

correlacao_clicou_tempo_diario = df[coluna_clicou_no_anuncio].corr(df[coluna_tempo_diario])

print(f"Correlação entre {coluna_clicou_no_anuncio} e {coluna_tempo_diario}: {correlacao_clicou_tempo_diario}")

sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribuição de Idades')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

sns.histplot(x= df['Area Income'])
plt.title('Distribuição de Renda')
plt.xlabel('Renda')
plt.show()

sns.boxplot(x=df['Area Income'])
plt.title('Distribuição de Renda')
plt.xlabel('Renda')
plt.show()

sns.histplot(df['Timestamp'])
plt.title('Distribuição de Tempo')
plt.xlabel('Tempo')
plt.ylabel('Frequência')
plt.show()

df['Idade_Quartis'] = pd.qcut(df['Age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Adicionando uma coluna de quartis para 'Renda'
df['Renda_Quartis'] = pd.qcut(df['Area Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

plt.figure(figsize=(8, 6))
sns.boxplot(x='Idade_Quartis', y='Age', data=df, order=['Q1', 'Q2', 'Q3', 'Q4'])
plt.title('Boxplot para Idade por Quartil')
plt.xlabel('Quartil')
plt.ylabel('Idade')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Renda_Quartis', y='Area Income', data=df, order=['Q1', 'Q2', 'Q3', 'Q4'])
plt.title('Boxplot para Renda por Quartil')
plt.xlabel('Quartil')
plt.ylabel('Renda')
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.regplot(x='Age', y='Daily Internet Usage', data=df)

plt.title('Gráfico de Dispersão com Regressão Linear')
plt.show()

sns.regplot(data = df, x = 'Age', y = 'Daily Time Spent on Site')

fig = sns.lmplot(data=df,x="Age",y="Daily Time Spent on Site",hue="Clicked on Ad", markers=['o','x'], scatter_kws={'s': 10})
sns.set(style="darkgrid")
avg_time = df['Daily Time Spent on Site'].mean()
plt.axhline(avg_time, color='red', linestyle='--', label=f'Média: {avg_time:.2f} min')
plt.figure(figsize=(13, 9))
plt.show(fig)

