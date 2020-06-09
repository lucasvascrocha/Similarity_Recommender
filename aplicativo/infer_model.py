import matplotlib.pyplot as plt
from joblib import dump, load
from feature_engineering import Feature_engineering




class ModelInference:
    def __init__(self):
        self.modelo = None

    def predict(self, df):
        """
        Retorna um array com as predições
        Entrada é o data frame que se deseja prever
        """

        print('Carregando o modelo')
        self.modelo = load('modelo.pkl')

        df = Feature_engineering().change_columns(df)

        lista_encoder = ['id', 'de_natureza_juridica', 'sg_uf', 'natureza_juridica_macro', 'de_ramo', 'setor',
                         'idade_emp_cat', 'fl_rm', 'nm_divisao', 'nm_segmento',
                         'sg_uf_matriz', 'de_saude_tributaria', 'de_saude_rescencia',
                         'de_nivel_atividade', 'de_natureza_juridica_change', 'de_ramo_change',
                         'setor_change', 'idade_change', 'nm_divisao_change',
                         'nm_segmento_change', 'de_nivel_atividade_change']

        labelels_df1 = df[lista_encoder]

        labelels_df1 = labelels_df1.dropna()

        labeled_df1 = labelels_df1.drop('id', axis=1)

        labeled_df1 = Feature_engineering().predic_encoder(labeled_df1)

        labeled_df1 = labeled_df1[
            ['nm_segmento_change', 'de_ramo_change', 'de_natureza_juridica_change', 'nm_divisao_change',
             'de_nivel_atividade_change', 'idade_change']]

        X = self.modelo['scaler'].transform(labeled_df1)
        X = self.modelo['pca'].transform(X)
        pred = self.modelo['modelo'].predict(X)

        labelels_df1 = labelels_df1.reset_index(drop=True)
        labelels_df1['grupo'] = pred

        print(plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s=100, color='red', label='cluster1'))
        print(plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=100, color='blue', label='cluster2'))
        print(plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s=100, color='yellow', label='cluster3'))
        print(plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s=100, color='green', label='cluster4'))
        print(plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s=100, color='brown', label='cluster5'))
        print(plt.scatter(X[pred == 5, 0], X[pred == 5, 1], s=100, color='gray', label='cluster6'))

        plt.scatter(self.modelo['modelo'].cluster_centers_[:, 0], self.modelo['modelo'].cluster_centers_[:, 1], s=100,
                    color='black', label='centroid')
        plt.legend()

        return labelels_df1[
            ['id', 'grupo', 'nm_segmento_change', 'de_ramo_change', 'de_natureza_juridica_change', 'nm_divisao_change',
             'de_nivel_atividade_change', 'idade_change']]

