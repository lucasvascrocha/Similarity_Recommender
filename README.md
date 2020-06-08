## CONTEXTO DO SOFTWARE

**Aplicativo capaz de fazer recomendação por similaridade**

O objetivo deste aplicativo é encontrar, em uma lista de diversos consumidores, os mais similares aos clientes já fidelizados de uma dada empresa. Com isso, o funcionamento da aplicação depende de um formato de dados de entrada específico, não sendo um modelo genérico.

Há a possibilidade de fazer teste do aplicativo com nossos dados

acesso ao deploy pelo link:
http://similarity-recommender.herokuapp.com/

## Desenvolvimento

O aplicativo foi desenvolvido na linguagem de programação Python através do App Streamlit e seu deploy feito para a plataforma de nuvem Heroku.

O App streamlit é capaz de criar um aplicativo com uma interface genérica e dinâmica através de uma linguagem simples que suporta python dentre outras linguagens.

O problema foi resolvido através de aprendizagem não supervisionada pelo modelo kmeans

A métrica de avaliação utilizada foi a % de filtragem de recomendação e % de similaridade dos consumidores recomendados com
os clientes já fidelizados, de forma a recomendar uma lista de tamanho considerável (<30% da lista com todos os alvos)  sendo ela o mais similar possível(> 70% de similaridade).

**Objetivo**

O objetivo do App é criar uma lista selecionada de consumidores alvos com características mais similares possíveis com os clientes de uma dada empresa já fidelizados. Com isso a busca de novos clientes será otimizada, além da possibilidade de direcionamento de ações para determinados perfis de cliente.

**Canvas**
<p align="center"> 
<img src="https://github.com/lucasvascrocha/teste/blob/master/Canvas%20rec%20by%20similarity.gif">
</p>

**Principais tecnologias utilizadas**

- pandas==1.0.4
- numpy==1.18.5
- joblib==0.15.1
- streamlit==0.61.0
- seaborn==0.10.1
- matplotlib==3.2.1
- scikit_learn==0.23.1

