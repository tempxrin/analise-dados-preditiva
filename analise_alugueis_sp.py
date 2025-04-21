# Importando bibliotecas essenciais para análise de dados, visualização e modelagem
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Carregando a base de dados de aluguéis de São Paulo
df_alugueis_sp = pd.read_csv('base-alugueis-sp.csv')

# Removendo um registro específico (linha 6095) considerado fora do padrão
df_alugueis_sp = df_alugueis_sp.drop(6095, axis=0)

# Convertendo a variável categórica 'type' em valores numéricos para facilitar a análise
df_alugueis_sp['type'] = df_alugueis_sp['type'].replace('Apartamento', 0).replace('Casa', 1).replace('Casa em condomínio', 2).replace('Studio e kitnet', 3)

# Exibindo estatísticas descritivas dos dados, com arredondamento de 2 casas decimais
df_alugueis_sp.describe().round(2)

# Calculando a matriz de correlação entre as variáveis numéricas
df_alugueis_sp.corr().round(2)

# Visualizando a distribuição dos valores de aluguel (rent) em um boxplot
ax = sns.boxplot(data=df_alugueis_sp, y='rent')
ax.figure.set_size_inches(10,8)
ax.set_title('Aluguel (R$)')

# Relacionando o valor do aluguel com o número de vagas de garagem
ax = sns.boxplot(data=df_alugueis_sp, y='rent', x='garage')
ax.figure.set_size_inches(10,8)
ax.set_title('Aluguel (R$) por Vagas de Garagem')

# Relacionando o valor do aluguel com o número de quartos
ax = sns.boxplot(data=df_alugueis_sp, y='rent', x='bedrooms')
ax.figure.set_size_inches(10,8)
ax.set_title('Aluguel (R$) por Número de Quartos')

# Relacionando o valor do aluguel com o tipo de imóvel
ax = sns.boxplot(data=df_alugueis_sp, y='rent', x='type')
ax.figure.set_size_inches(10,8)
ax.set_title('Aluguel (R$) por Tipo de Imóvel')

# Definindo as variáveis independentes (X) e a variável alvo (y)
y = df_alugueis_sp.rent
X = df_alugueis_sp[['area', 'bedrooms', 'garage', 'type']]

# Dividindo os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)

# Criando e treinando o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_previsto = modelo.predict(X_test)

# Avaliando o desempenho do modelo com o R² para os conjuntos de treino e teste
print('R² de Treino = {}'.format(modelo.score(X_train, y_train).round(2)))
print('R² de Teste = %s' % metrics.r2_score(y_test, y_previsto).round(2))

# Simulador: estimando o valor do aluguel com base em valores fictícios de entrada
area = 100
garagem = 2
quartos = 1
type = 2
entrada = [[area, garagem, quartos, type]]

# Exibindo o valor previsto para o aluguel
print('R$ {0:.2f}'.format(modelo.predict(entrada)[0]))
