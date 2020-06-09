#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import pandas as pd
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from feature_engineering import Feature_engineering
from training_model import ModelTraining
from infer_model import ModelInference
from recomender import Recomendation

from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
import collections
from collections import Counter



# ------------------------------ EXPORTAÇÃO DA TABELA -------------------------------



def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Tabela_Download.csv">Download</a>'
    return href


# -------------------------- REDUÇÃO DE TAMANHO TROCANDO INT E FLOAT64 POR FLOAT16 -----------------------

def reduce_mem_usage(props):
         
    for col in props.columns:
        if props[col].dtype != object and props[col].dtype != bool:  # Exclude strings and bool
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
                        
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 16 bit
            else:
                props[col] = props[col].astype(np.float16)
    return props



# ----------------------------------SIDEBAR -------------------------------------------------------------

def main():

    st.sidebar.header("Similarity Recommender")
    n_sprites = st.sidebar.radio(
        "Escolha uma opção", options=["Exemplo com nossos dados", "Crie recomendações com seus dados", "Pré processamento de dados","Análise exploratória"], index=0
    )


    st.sidebar.markdown('Desenvolvido por: Lucas V. Rocha')
    st.sidebar.markdown('Email para contato: lucas.vasconcelos3@gmail.com')
    st.sidebar.markdown('Portfólio: https://github.com/lucasvascrocha')
   
                         
    
# ------------------------------ COMUM EXEMPLO COM NOSSOS DADOS------------------------------------------
        
    if n_sprites == "Exemplo com nossos dados":
        
        st.image('https://media.giphy.com/media/BmmfETghGOPrW/giphy.gif', width=600)
        st.title('Sistema de recomendação')
        st.subheader('Desenvolvido para recomendação de clientes por similaridade')
        st.write('O objetivo deste aplicativo é encontrar, em uma lista de diversos consumidores, os mais similares aos clientes já fidelizados de uma dada empresa. Com isso, o funcionamento da aplicação depende de um formato de dados de entrada específico, não sendo um modelo genérico.')

        # TREINAMENTO MODELO EX NOSSOS DADOS  
        if st.button("Criar o modelo com dados de teste"):
            with st.spinner("Criando o modelo...aguarde a inteligência artificial trabalhar, isto pode levar alguns minutos - Aqui é utilizado uma tabela com os dados de todos os alvos, consumidores aos quais se busca identificar alguma similaridade."):

                df_choiced = pd.read_csv('df_entrada_modelo.csv')
                st.markdown('Esta é a distribuição e agrupamento de todos os dados do modelo criado, quanto mais uniformes se formarem os grupos e mais separados entre si melhor será o desempenho do modelo em reconhecer as diferenças.')
                modelo, group = ModelTraining().model_training(df_choiced)
                plt.tight_layout()
                st.pyplot()                                                                       
                    
                st.markdown('Este gráfico representa o número de grupos formados pela quantidade de indivíduos de cada grupo, quanto mais uniforme for a distribuição dos indivíduos nos grupos melhor será a previsão do modelo.')
                st.markdown('Distribuição dos grupos')
                sns.countplot(group[0])
                st.pyplot()

                #  TREINO DOS DADOS DE TESTE
   
                st.subheader('Treinamento dos dados')
            
                df = pd.read_csv('estaticos_portfolio1.csv')
                
                st.subheader('Treinamento dos dados')
                pred1 = ModelInference().predict(df)
                plt.tight_layout()
                st.markdown('Distribuição dos dados e agrupamento de seus clientes')
                st.pyplot()
                st.markdown('Distribuição de grupos de sua empresa, os grupos com maior frequência são os que mais representam seus clientes fidelizados, estes grupos serão utilizdos para recomendar os consumidores da lista alvo.')

                sns.countplot(pred1['grupo'])
                st.pyplot()
                st.subheader('avalião da similaridade e arquivo com os nomes das empresas recomendadas')
                st.markdown('A porcentagem de recomendação da lista total se refere a proporção de clientes alvos recomendados de uma lista com todos os alvos e a proporção de similaridade se refere a similaridade dos clientes recomendados com os já fidelizados')
                features_recomendadas = Recomendation().leads(pred1, group)
                rec= features_recomendadas['id']
                st.markdown('**Índices recomendados**')
                st.dataframe(rec)                          
                st.markdown(get_table_download_link(rec), unsafe_allow_html=True)                            
               

        # TREINAMENTO COM UPLOAD         

        st.subheader('Faça upload dos dados caso seja o usuário para qual este modelo foi criado')
        file  = st.file_uploader('Entre com sua tabela de clientes (.csv)', type = 'csv')
        if file is not None:
            st.subheader('Analisando os dados')
            df = pd.read_csv(file)   
            df_choiced = pd.read_csv('df_entrada_modelo.csv')
            st.markdown('Esta é a distribuição e agrupamento de todos os dados do modelo criado, quanto mais uniformes se formarem os grupos e mais separados entre si melhor será o desempenho do modelo em reconhecer as diferenças.')
            modelo, group = ModelTraining().model_training(df_choiced)
            plt.tight_layout()
            st.pyplot()

            st.markdown('Este gráfico representa o número de grupos formados pela quantidade de indivíduos de cada grupo, quanto mais uniforme for a distribuição dos indivíduos nos grupos melhor será a previsão do modelo.')
            st.markdown('Distribuição dos grupos')
            sns.countplot(group[0])
            st.pyplot()            
                        
            st.subheader('Treinamento dos dados')
            pred1 = ModelInference().predict(df)
            plt.tight_layout()
            st.markdown('Distribuição dos dados e agrupamento de seus clientes')
            st.pyplot()
            st.markdown('Distribuição de grupos de sua empresa, os grupos com maior frequência são os que mais representam seus clientes fidelizados, estes grupos serão utilizdos para recomendar os consumidores da lista alvo.')

            sns.countplot(pred1['grupo'])
            st.pyplot()
            
            st.subheader('avalião da similaridade e arquivo com os nomes das empresas recomendadas')
            st.markdown('A porcentagem de recomendação da lista total se refere a proporção de clientes alvos recomendados de uma lista com todos os alvos e a proporção de similaridade se refere a similaridade dos clientes recomendados com os já fidelizados')
            features_recomendadas = Recomendation().leads(pred1, group)
            rec= features_recomendadas['id']
            st.markdown('**Índices recomendados**')
            st.dataframe(rec)                          
            st.markdown(get_table_download_link(rec), unsafe_allow_html=True)
            


# ------------------------------ FIM DO EXEMPLO ----------------------------                            

# ------------------------------ INÍCIO ANÁLISE EXPLORATÓRIA ---------------------------- 
            

    if n_sprites == "Análise exploratória":
        st.image('https://media.giphy.com/media/d1E2HXeuONnx5YfC/giphy.gif', width=200)                           
        st.title('App para Análise exploratória de dados')
        st.subheader('Inclui funções para corte de linhas e colunas')
        file  = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')
        if file is not None:
            st.subheader('Analisando os dados')
            df = pd.read_csv(file)
            st.markdown('**Número de linhas:**')
            st.markdown(df.shape[0])
            st.markdown('**Número de colunas:**')
            st.markdown(df.shape[1])
            st.markdown('**Visualizando o dataframe**')
            number = st.slider('Escolha o numero de linhas que deseja vizualizar', min_value=1, max_value=df.shape[0])
            st.dataframe(df.head(number))
            st.markdown('**Nome das colunas:**')
            st.markdown(list(df.columns))

            # OPÇÃO DE CORTE E MANIPULAÇÃO DA TABELA

            check = st.checkbox('Caso precise cortar linhas ou colunas do DF original clique aqui')
            if check:
                st.subheader('Seleção da parte da tabela que deseja separar')   
                row_init = int(st.number_input(label='Número da linha onde deseja iniciar o corte - Pressione enter e avançe para o próximo campo'))
                st.markdown(row_init)
                row_end = int(st.number_input(label='Número da linha onde deseja finalizar o corte - Pressione enter e avançe para o próximo campo'))
                st.markdown(row_end)
                df = df[row_init:row_end+1]
                st.dataframe(df)
                check = st.checkbox('Clique aqui para cortar colunas')
                if check:
                    col_cut = st.multiselect("Selecione as colunas que deseja cortar", df.columns.tolist())
                    st.write(col_cut)
                    df.drop(col_cut,axis=1, inplace=True)
                    st.dataframe(df)
                st.subheader('Faça download da tabela manipulada abaixo : ')
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)

           # EXPLORAÇÃO DA EXPLORAÇÃO 

            st.markdown('**Resumo dos Dados**')
            select_analise = st.radio('Escolha uma análise abaixo :', ('head', 'info', 'describe', 'faltantes'))
            if select_analise == 'head':
                st.dataframe(df.head())
            if select_analise == 'info':
                st.dataframe({'Dtype': df.dtypes, 'Non-Null Count': df.count()})
            if select_analise == 'describe':
                st.dataframe(df.describe())
            if select_analise == 'faltantes':
                st.dataframe(pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes, 'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})) 

            st.subheader('**Análise Exploratória**')
            sns.set_style('whitegrid')
            plt.tight_layout()
            plt.figure(figsize=(12,8))
            if st.checkbox("Correlação entre colunas"):
                st.markdown('Mapa geral de correlação')
                sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
                st.pyplot()
                col_corr = st.multiselect("Selecione a coluna que deseja correlacionar", df.columns.tolist())
                if col_corr:
                    st.markdown('Correlação por coluna')
                    st.dataframe(df.corr()[col_corr])
            if st.checkbox("Pair plot"):
                sns.pairplot(df)
                st.pyplot()
            if st.checkbox("Histograma"):
                st.markdown('selecione uma coluna para ver seu histograma')
                option = st.selectbox('Selecione uma coluna',list(df.columns),key="wid1")
                num_bins = st.slider('Escolha o numero de bins', min_value=5, max_value=df.shape[0])
                plt.hist(sorted(df[option]),bins=num_bins)
                st.pyplot()
            if st.checkbox("Bar Plot"):
                st.markdown('Selecione as colunas que deseja plotar')
                col1 = st.selectbox('Variável x',list(df.columns),key="wid2")
                col2 = st.selectbox('Variável y',list(df.columns), key="wid3")
                sns.barplot(df[col1],df[col2])
                st.pyplot()
            if st.checkbox("Scatter Plot"):
                st.markdown('Selecione as colunas que deseja plotar')
                col1 = st.selectbox('Variável x',list(df.columns),key="wid4")
                col2 = st.selectbox('Variável y',list(df.columns),key="wid5")
                plt.scatter(df[col1],df[col2])
                st.pyplot()
            if st.checkbox("Box Plot"):
                st.markdown('Selecione as colunas que deseja plotar')
                col1 = st.selectbox('Variável x',list(df.columns),key="wid6")
                col2 = st.selectbox('Variável y',list(df.columns),key="wid7")
                sns.boxplot(df[col1],df[col2])
                st.pyplot()
            if st.checkbox("Count Plot"):
                st.markdown('Selecione a coluna que deseja plotar')
                col1 = st.selectbox('Variável x',list(df.columns),key="wid8")
                sns.countplot(df[col1])
                st.pyplot()
            if st.checkbox("Swarm Plot"):
                st.markdown('Selecione as colunas que deseja plotar')
                col1 = st.selectbox('Variável x',list(df.columns),key="wid9")
                col2 = st.selectbox('Variável y',list(df.columns),key="wid10")
                sns.swarmplot(df[col1],df[col2])
                st.pyplot()  
                         
# ------------------------------ FIM ANÁLISE EXPLORATÓRIA ----------------------------

# ------------------------------ INÍCIO PRÉ PROCESSAMENTO ----------------------------

    if n_sprites == "Pré processamento de dados":
        st.subheader('Pré-processamento de Dados em Python')
        st.image('https://media.giphy.com/media/KyBX9ektgXWve/giphy.gif', width=200)
        file  = st.file_uploader('Escolha a base de dados que deseja processar (.csv)', type = 'csv')
        if file is not None:
            st.subheader('Analisando os dados')
            df = pd.read_csv(file)
            st.markdown('**Número de linhas:**')
            st.markdown(df.shape[0])
            st.markdown('**Número de colunas:**')
            st.markdown(df.shape[1])
            st.markdown('**Visualizando o dataframe**')
            number = st.slider('Escolha o numero de colunas que deseja visualizar', min_value=1, max_value=20)
            st.dataframe(df.head(number))
            st.markdown('**Nome das colunas:**')
            st.markdown(list(df.columns))
            exploracao = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes, 'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})
            st.markdown('**Contagem dos tipos de dados:**')
            st.write(exploracao.tipos.value_counts())
            st.markdown('**Nomes das colunas do tipo int64:**')
            st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']))
            st.markdown('**Nomes das colunas do tipo float64:**')
            st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))
            st.markdown('**Nomes das colunas do tipo object:**')
            st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))
            st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
            st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])                                    
            
            st.subheader('Inputaçao de dados númericos :')
            percentual = st.slider('Escolha o limite de percentual de dados faltantes nas colunas, só serão imputados dados onde as colunas tenham um percentual de dados faltantes abaixo do limite escolhido', min_value=0, max_value=100)
            lista_colunas = list(exploracao[exploracao['NA %']  < percentual]['nomes'])
            select_method = st.radio('Escolha um metodo abaixo :', ('Média', 'Mediana'))
            st.markdown('Você selecionou : ' +str(select_method))
            
            if select_method == 'Média':
                df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
                exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                           'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
                st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
                st.subheader('Dados Inputados faça download abaixo : ')
                
            if select_method == 'Mediana':
                df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
                exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                           'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
                st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
                        
            # OPÇÃO DE CORTE E MANIPULAÇÃO DA TABELA 

            check = st.checkbox('Caso precise cortar linhas ou colunas do DF original clique aqui')
            if check:
                st.subheader('Caso queira recortar algumas linhas da tabela')   
                row_init = int(st.number_input(label='Número da linha onde deseja iniciar o corte - Pressione enter e avançe para o próximo campo'))
                st.markdown(row_init)
                row_end = int(st.number_input(label='Número da linha onde deseja finalizar o corte - Pressione enter e avançe para o próximo campo'))
                st.markdown(row_end)
                df = df[row_init:row_end+1]
                st.dataframe(df)
                check = st.checkbox('Clique aqui para cortar colunas')
                if check:
                    col_cut = st.multiselect("Selecione as colunas que deseja cortar", df.columns.tolist())
                    st.write(col_cut)
                    df.drop(col_cut,axis=1, inplace=True)
                    st.dataframe(df)
            
            if st.checkbox("Clique aqui para manipular dados faltantes de linhas e colunas"):
                st.markdown('**Resumo dos Dados**')
                exploracao = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes,
                               'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})
                st.dataframe(exploracao)
                num_cut_1 = st.slider('Escola o limite de NA % para o 1º corte, todos as colunas com NA % > que este valor serão cortadas', min_value=0, max_value=100,value=0)
                lista_colunas = list(exploracao[exploracao['NA %']  < num_cut_1]['nomes'])
                df = df[lista_colunas]
                explor_cut = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes,
                               'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})
                st.dataframe(explor_cut)
                num_cut_2 = st.slider('Escola o limite de NA % para excluir linhas, todos as linhas com NA % > que este valor serão cortadas', min_value=0, max_value=100, value =0)
                lista_linhas = list(exploracao[exploracao['NA %']  < num_cut_2]['nomes'])
                df.dropna(subset=lista_linhas, inplace=True)
                explor_cut = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes,
                               'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})
                st.dataframe(explor_cut)                                            
                    
            st.subheader('Faça download da tabela manipulada abaixo : ')
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

# ------------------------------ FIM PRÉ PROCESSAMENTO ----------------------------

# ------------------------------ INÍCIO CRIE RECOMENDAÇÃO COM SEUS DADOS--------------


    if n_sprites == "Crie recomendações com seus dados":
        st.subheader('Crie recomendações com seus dados')
        st.image('https://media.giphy.com/media/dyuc5DfSUg1RGg8P3p/giphy.gif', width=600)
        st.markdown('**É necessário que o arquivo não contenha dados faltantes, para isso vá na aba de pré processamento e confira se sua tabela preenche este pré-requisito.**')
        file  = st.file_uploader('Entre com a tabela completa para a criação do modelo - aqui entra-se com todos os alvos dos quais deseja-se avaliar a similaridade(.csv)', type = 'csv')
        if file is not None:
            st.subheader('Analisando os dados')
            df = pd.read_csv(file)
            df = reduce_mem_usage(df)
            st.markdown('**Número de linhas:**')
            st.markdown(df.shape[0])
            st.markdown('**Número de colunas:**')
            st.markdown(df.shape[1])
            st.markdown('**Visualizando o dataframe**')        
            st.dataframe(df.head())
            check = st.checkbox('Clique aqui para cortar colunas - retire a coluna de identificação')
            if check:
                col_cut = st.multiselect("Selecione a coluna de identificação", df.columns.tolist())
                st.write(col_cut)
                idDf = df[col_cut]
                df.drop(col_cut,axis=1, inplace=True)
                st.dataframe(df)
                st.dataframe(idDf)

            # início de parametrização da criação do modelo

            if st.checkbox("Tabela sem coluna de identificação? clique aqui"):
                
                explor_cut = pd.DataFrame({'nomes': df.columns,
                                   'tipos': df.dtypes, 'Nunique': df.nunique()})

                list_name_encoder = (list(explor_cut[explor_cut['tipos'] == 'object']['nomes']) + list(
                    explor_cut[explor_cut['tipos'] == 'bool']['nomes']))
                list_name_encoder = list_name_encoder[:]
                df = df[list_name_encoder]

                encoder_dict = collections.defaultdict(LabelEncoder)
                df = df.apply(lambda x: encoder_dict[x.name].fit_transform(x))
                                
                QtlTransf = QuantileTransformer()
                X = QtlTransf.fit_transform(df)

                pca_95 = PCA(0.95)
                X = pca_95.fit_transform(X)
                st.markdown('**Distribuição dos dados, escolha o número de grupos de acordo com o arranjo dos dados**')
                plt.scatter(X[:, 0], X[:, 1])
                st.pyplot()
                
                n_grupos = st.slider('Escola o número de grupos que deseja formar', min_value=1, max_value=20,value=1)

                if st.checkbox("Criar modelo"):    
                    modelo = KMeans(n_clusters=n_grupos, max_iter=10, n_init=30, random_state=42)
                    modelo.fit(X)

                    model = {'scaler': QtlTransf,
                             'pca': pca_95,
                             'modelo': modelo}

                    print(model)
                    dump(model, 'modelo.pkl')

                    pred = modelo.predict(X)
                    plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s=100, color='red', label='cluster1')
                    plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=100, color='blue', label='cluster2')
                    plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s=100, color='yellow', label='cluster3')
                    plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s=100, color='green', label='cluster4')
                    plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s=100, color='brown', label='cluster5')
                    plt.scatter(X[pred == 5, 0], X[pred == 5, 1], s=100, color='gray', label='cluster6')

                    plt.scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[:, 1], s=100, color='black',
                                label='centroid')
                    plt.legend()
                    st.markdown('**Distribuição dos dados agrupados**')
                    st.pyplot()

                    # Data frame com previsão e índices
                    group = pd.DataFrame(pred)
                    idDf = pd.read_csv('group.csv')
                    idDf = idDf['indices'].copy()
                    group['indices'] = idDf
                    st.markdown('**Tabela com índice e grupo de cada alvo**')
                    st.dataframe(group)
                    st.subheader('Faça download da tabela com os índices e grupos formados : ')
                    st.markdown(get_table_download_link(group), unsafe_allow_html=True)
                    sns.countplot(group[0])
                    st.markdown('**Distribuição dos grupos formados, frequência de indivíduos em cada grupo**')
                    st.pyplot()
                    
                    
                    #entrada do teste
                    st.markdown('**Previsão - nesta etapa é possível prever os grupos de alguma tabela seguindo os parâmetro criados acima**')
                    file2  = st.file_uploader('Aqui entra-se com os dados de base, dados de referência para serem utilizados como parâmetro na busca por alvos, exemplo: seus clientes fidelizados. Devem conter as mesmas colunas (Features) da tabela utilizada para treino e não pode haver dados faltantes (.csv)', type = 'csv')
                    df1 = pd.read_csv(file2)
                    st.dataframe(df1)
                    if st.checkbox('Clique aqui para cortar colunas - retire a coluna de identificação!'):
                        col_cut = st.multiselect("Selecione a coluna de identificação", df1.columns.tolist(),key="teste")
                        st.write(col_cut)
                        idDf = df1[col_cut]
                        df1.drop(col_cut,axis=1, inplace=True)
                        st.dataframe(df1)
                        st.dataframe(idDf)
                    if st.checkbox("Tabela sem coluna de identificação? clique aqui!"):

                        st.subheader('Analisando os dados')
                        df1 = df1[df.columns]
                        df1 = df1.apply(lambda x: encoder_dict[x.name].transform(x))

                        # pré processamento importado do treino
                        X = model['scaler'].transform(df1)
                        X = model['pca'].transform(X)
                        pred = model['modelo'].predict(X)
                        df1['grupos'] = pred.copy()
                        sns.countplot(df1['grupos'])
                        st.markdown('**Frequência de grupos dos seus dados de referência**')
                        st.markdown('Os grupos com maior frequência serão utilizados para fazer a recomendação do alvo ')
                        st.pyplot()
                        
                        #recomendação
                        st.markdown('**Recomendação de alvos similares**')
                        recomender_1 = Counter(df1['grupos'])

                        if recomender_1[0] > (len(pred) / 2):
                            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
                        else:
                            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
                            leads_port1 = leads_port1.append(group[group[0] == recomender_1.most_common()[1][0]]['indices'])

                        st.markdown(f'Foi recomendado: {round(len(leads_port1) / len(group) * 100, 2)} % da lista total de alvos que são similares a seus dados')
                        st.markdown('**Lista de recomendados**')
                        st.dataframe(leads_port1)
                        st.subheader('Faça download do índice dos alvos recomendados : ')
                        st.markdown(get_table_download_link(leads_port1), unsafe_allow_html=True)
                        
                        
                        
                            
                            
                        
                      
            
            


    

   
        
       
        
if __name__ == '__main__':
    main()


# In[10]:





# In[2]:





# In[3]:





# In[ ]:




