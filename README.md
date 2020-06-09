## CONTEXTO DO SOFTWARE

**Aplicativo capaz de fazer recomendação por similaridade**

O objetivo deste aplicativo é encontrar, em uma lista de diversos consumidores, os mais similares aos clientes já fidelizados de uma dada empresa. Com isso a aquisição de novos clientes será otimizada, possibilitando o direcionamento de ações para um determinado perfil de cliente. 

O funcionamento da aplicação depende de um formato de dados de entrada específico, não sendo um modelo genérico.
Há a possibilidade de fazer teste do aplicativo com nossos dados.

acesso ao deploy pelo link:
http://similarity-recommender.herokuapp.com/

## Desenvolvimento

O aplicativo foi desenvolvido na linguagem de programação Python através do App Streamlit e seu deploy feito para a plataforma de nuvem Heroku. O App Streamlit é capaz de criar um aplicativo com uma interface genérica e dinâmica através de uma linguagem simples que suporta python dentre outras linguagens.

O problema foi resolvido através de aprendizagem não supervisionada pelo modelo K-means.
A métrica de avaliação utilizada foi a proporção de novos clientes recomendados e porcentagem de similaridade destes com
os clientes já fidelizados, de forma a recomendar uma lista de tamanho considerável (< 30% da lista com todos os alvos)  sendo ela o mais similar possível (> 70% de similaridade).

O desenvolvimento do código ocorreu na forma de criação de classes utilizando orientação à objeto afim de facilitar o entendimento e melhorar o desempenho do código, visto que a performance é um quesito importante dado o tamanho dos dados utilizados para treinamento do modelo. Por este motivo todos os arquivos csv foram ignorados para este repositório.

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

**Imagens do aplicativo**

Tela inicial
<p align="center"> 
<img src="https://github.com/lucasvascrocha/teste/blob/master/imagens/inicial.png">
</p>

Previsão
<p align="center"> 
<img src="https://github.com/lucasvascrocha/teste/blob/master/imagens/recomend.png">
</p>

