from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd


class ModelTraining:
    """
    Treinamento do modelo
    """

    def __init__(self):
        pass

    def model_training(self, df):
        """
        Cria modelo e salva os fits, entrada é um df já pré-processado
        2 saídas, o modelo e um data frame com os ids e grupos gerados
        """

        QtlTransf = QuantileTransformer()
        X = QtlTransf.fit_transform(df)

        pca_95 = PCA(0.95)
        X = pca_95.fit_transform(X)

        modelo = KMeans(n_clusters=6, max_iter=10, n_init=30, random_state=42)
        modelo.fit(X)

        model = {'scaler': QtlTransf,
                 'pca': pca_95,
                 'modelo': modelo}

        print(model)
        dump(model, 'modelo.pkl')

        pred = modelo.predict(X)
        print(plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s=100, color='red', label='cluster1'))
        print(plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=100, color='blue', label='cluster2'))
        print(plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s=100, color='yellow', label='cluster3'))
        print(plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s=100, color='green', label='cluster4'))
        print(plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s=100, color='brown', label='cluster5'))
        print(plt.scatter(X[pred == 5, 0], X[pred == 5, 1], s=100, color='gray', label='cluster6'))

        plt.scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[:, 1], s=100, color='black',
                    label='centroid')
        plt.legend()

        group = pd.DataFrame(pred)
        idDf = pd.read_csv('group.csv')
        idDf = idDf['indices'].copy()
        group['indices'] = idDf

        return model, group
