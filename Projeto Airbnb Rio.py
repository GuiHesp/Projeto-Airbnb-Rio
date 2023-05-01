#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados

# In[ ]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[ ]:


meses = {'jan': 1, 'fev': 2, 'mar': 3,'abr': 4, 'mai':5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)

display(base_airbnb)


# ### Consolidar Base de Dados

# Colunas para excluir:   

# In[ ]:


print(base_airbnb['experiences_offered'].value_counts())


# In[ ]:


print((base_airbnb['host_listings_count']==base_airbnb['host_total_listings_count']).value_counts())


# In[ ]:


print(base_airbnb['square_feet'].isnull().sum())


# In[ ]:


print(list(base_airbnb.columns))

base_airbnb.head(1000).to_csv('primeiros registros.csv', sep=';')


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir

# In[ ]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']
base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
display(base_airbnb)


# ### Tratar Valores Faltando

# In[ ]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

print(base_airbnb.isnull().sum())


# In[ ]:


base_airbnb = base_airbnb.dropna()

print(base_airbnb.shape)
print(base_airbnb.isnull().sum())


# ### Verificar Tipos de Dados em cada coluna

# In[ ]:


print(base_airbnb.dtypes)
print('-'*50)
print(base_airbnb.iloc[0])


# In[ ]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$','')
base_airbnb['price'] = base_airbnb['price'].str.replace(',','')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)


# In[ ]:


print(base_airbnb.dtypes)


# ### Análise Exploratória e Tratar Outliers

# -Vamos basicamente olhar feature por feature:
#     
#     1. Ver a correlação entre as features e decidir se manteremos todas as features que temos.
#     
#     2. Excluir outliers(usaremos como regra, valores abaixo de Q1-1,5xAmplitude e valores acima de Q3+1,5xAmplitude). Amplitude = Q3 - Q1
#     
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir.
#     
# -Vamos começas pelas colunas de preço(resultado final que queremos)e de extra_people(tbm valor monetário).Esses são os valores numéricos contínuos.
# 
# -Depois vamos analisar as colunas de valores numericos discretos(accomodates, bedrooms,...)
# 
# -Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(base_airbnb.corr(), annot=True , cmap='Greens')

#print(base_airbnb.corr())


# ## Definição de funções para análise de Outliers
# 
# Vamos definir algumas funções para ajudar na análise de outliers das colunas

# In[ ]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] > lim_inf) & (df[nome_coluna] < lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[ ]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# In[ ]:


base_airbnb['host_listings_count'].value_counts()


# ### Price

# In[ ]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas apartamentos de altíssimo luxo, assim podemos excluir estes outliers.

# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(f'{linhas_removidas} linhas removidas.')


# In[ ]:


histograma(base_airbnb['price'])
print(base_airbnb.shape)


# ### Extra People

# In[ ]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{linhas_removidas} linhas removidas.')

histograma(base_airbnb['extra_people'])
print(base_airbnb.shape)


# ### host_listings_count

# In[ ]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# Podemos excluir os outliers, porque para o objetivo do projeto o hosts com mais de 6 imoveis no airbnb não seja o público alvo do objetivo do projeto.

# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas.')


# ### accommodates

# In[ ]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{linhas_removidas} linhas removidas.')


# ### bathrooms

# In[ ]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{linhas_removidas} linhas removidas.')


# ### bedrooms

# In[ ]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{linhas_removidas} linhas removidas.')


# ### beds

# In[ ]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print(f'{linhas_removidas} linhas removidas.')


# ### guests_included 

# In[ ]:


#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Vamos remover esta feature da análise, parece q os usuários do airbnb usam muito o valor padrão do airbnb como 1 o guests included, isso pode levar o modelo a considerar uma feature que não é verdade para a definição de preço.

# In[ ]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape


# ### minimum_nights 

# In[ ]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas.')


# ### maximum_nights 

# In[ ]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# In[ ]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# ### number_of_reviews

# In[ ]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# In[ ]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# ## Tamanho de Colunas de Valores de Texto

# ### property_type

# In[ ]:


print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[ ]:


tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []
for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'
    
print(base_airbnb['property_type'].value_counts())


# In[ ]:


print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### room_type

# In[ ]:


print(base_airbnb['room_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# bed_type 

# In[ ]:


print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[ ]:


tabela_tipos_cama = base_airbnb['bed_type'].value_counts()

colunas_agrupar_cama = []
for tipo in tabela_tipos_cama.index:
    if tabela_tipos_cama[tipo] < 10000:
        colunas_agrupar_cama.append(tipo)

print(colunas_agrupar_cama)

for tipo in colunas_agrupar_cama:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'
    
print(base_airbnb['bed_type'].value_counts())


# In[ ]:


print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# cancellation_policy

# In[ ]:


print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[ ]:


tabela_tipos_cancellation = base_airbnb['cancellation_policy'].value_counts()

colunas_agrupar_cancellation = []
for tipo in tabela_tipos_cancellation.index:
    if tabela_tipos_cancellation[tipo] < 10000:
        colunas_agrupar_cancellation.append(tipo)

print(colunas_agrupar_cancellation)

for tipo in colunas_agrupar_cancellation:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'Strict'
    
print(base_airbnb['cancellation_policy'].value_counts())


# In[ ]:


print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# amenities
# 
# -Como temos uma diversidade mto grande de amenities, sendo que as vezes as msm amenities podem ser escritas de formas diferentes, vamos avaliar a qtdade de amenities como o parametro para o nosso modelo.

# In[ ]:


print(base_airbnb['amenities'].iloc[2].split(','))
print(len(base_airbnb['amenities'].iloc[2].split(',')))

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[ ]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# In[ ]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# In[ ]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{linhas_removidas} linhas removidas.')


# In[ ]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# ### Visualização de Mapa das Propriedades

# In[ ]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5, zoom=10, mapbox_style='stamen-terrain')

mapa.show()


# ### Encoding
# 
# -Ajustar as features para facilitar o trabalho do modelo futuro(features de categoria,true e false, etc...
# 
# -Features de Valores True and False, T por 1 F por 0.
# -Features de Categoria(features em que os valores da coluna são textos), vamos utilizar o método de encoding de variáveis dummies.

# In[ ]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_codificado = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_codificado.loc[base_airbnb_codificado[coluna]=='t' ,coluna] = 1
    base_airbnb_codificado.loc[base_airbnb_codificado[coluna]=='f' ,coluna] = 0


# In[ ]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_codificado = pd.get_dummies(data=base_airbnb_codificado, columns=colunas_categorias)
display(base_airbnb_codificado.head())


# ### Modelo de Previsão

# - Métricas de avaliação

# In[ ]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}: \nR²:{r2:.2%} \nRSME:{RSME:.2f}'
    


# - Escolha dos modelos a serem testados:
#     1. RandomForest
#     2. LinearRegression
#     3. Extra Tree

# In[ ]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_codificado['price']
x = base_airbnb_codificado.drop('price', axis=1)


# - Separar os dados em treino e testes + Treino do Modelo

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(x_train, y_train)
    #testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))

 


# ### Análise do Melhor Modelo

# In[ ]:


for nome_modelo, modelo in modelos.items():
    #testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Modelo escolhido como Melhor Modelo: ExtraTreesRegressor
#     - Esse foi o modelo com o maior valor d r² e ao msm tempo o menor valor de RSME. Outro fator a velocidade é próxima entre o modelo de RandomForest que teve resultados próximos, vamos escolher o modelo extratrees.
#     - O modelo de LinearRegression nao obteve um resultado satisfatorio com valores de r² e RSME
#     
# - Resultados das Métricas de Avaliação do Modelo Vencedor:<br>
# Modelo ExtraTrees:<br>
# R²:97.49%<br>
# RSME:41.70

# ### Ajustes e Melhorias no Melhor Modelo

# In[ ]:


print(modelo_et.feature_importances_)
print(x_train.columns)
importancia_features = pd.DataFrame(modelo_et.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# ### Ajustes finais do modelo
#     - is_business_travel_ready ñ parece impactar o modelo. Por isso iremos deleta-la.

# In[ ]:


base_airbnb_codificado = base_airbnb_codificado.drop('is_business_travel_ready', axis=1)

y = base_airbnb_codificado['price']
x = base_airbnb_codificado.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(x_train, y_train)
previsao = modelo_et.predict(x_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# - bed_type

# In[ ]:


base_teste = base_airbnb_codificado.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
x = base_teste.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(x_train, y_train)
previsao = modelo_et.predict(x_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# ### Deploy do Projeto

# In[ ]:


x['price'] = y
x.to_csv('dados.csv')


# In[ ]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')

