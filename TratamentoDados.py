#Tratamento de dados baseados nas análises estatísticas realizadas

#Bibliotecas utilizadas
import pandas as pd

#Função para pegar os limites extremos (outliers)
def get_limits(data_variable):
    q1=data_variable.quantile(0.25)
    q3=data_variable.quantile(0.75)
    iqr=q3-q1
    lim_sup=q3+1.5*iqr
    lim_inf=q1-1.5*iqr
    return (lim_inf,lim_sup)

#Criar o Dataframe com os dados completos 
df = pd.read_csv('C:/Users/lgrodrigues/Documents/Projeto Final/Projeto-Final---Grupo-4/Dataset/advertising_full.csv')


#Inicio do tratamento de dados
df = df[(df['Age']>18) & (df['Age']<60)]  # Filtro na aplicado na dataframe pela coluna 'Age' para trazer somente as idade entre 18 e 60 anos.
#Selecionar dados positivos 
df = df[df['Daily Time Spent on Site']>=0]
df= df[df['Area Income']>=0]

limites_AreaIncome = get_limits(df['Area Income'])
lim_inf = limites_AreaIncome[0]
lim_sup = limites_AreaIncome[1]
lim_inf, lim_sup

df = df.drop((df[df['Area Income'] > lim_sup].index))

df = df[df['Daily Internet Usage']>=0]


#Salvar como dataframe
df.to_csv("Dataset/advertising_tratado.csv")