import pandas as pd
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Feat_select:


    def __init__(self):

        pass

    def rank(self, port1, port2, port3):
        """
        Entrada dos 3 data frames de empresas conhecidas
        Retorna uma lista com combinações de features selecionadas por Random Forest Rank
        necessita da entrada de 3 df distintos que representem algum grupo pré estabelecido

        """

        port1['grupo'] = 1
        port2['grupo'] = 2
        port3['grupo'] = 3
        todos = pd.DataFrame(port1)
        todos = todos.append(port2)
        todos = todos.append(port3)

        X = todos.drop(['grupo'], axis=1)
        y = todos['grupo']
        # estimators
        rf = RandomForestClassifier()
        rf = rf.fit(X, y)
        rfe = RFE(rf, n_features_to_select=1, verbose=2)
        rfe = rfe.fit(X, y)
        rank = pd.DataFrame({'features': X.columns})
        rank['RF rank'] = rfe.ranking_

        rfr = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
        rfr.fit(X, y)
        rank['RFR'] = (rfr.feature_importances_ * 100)

        linreg = LinearRegression(normalize=True)
        linreg.fit(X, y)
        rank['linreg'] = (linreg.coef_.round(3) * 10)

        model = LogisticRegression(solver='liblinear')
        rfe = RFE(model, 3)
        rfe = rfe.fit(X, y)
        rank['logreg'] = rfe.ranking_

        etc = ExtraTreesClassifier()
        etc.fit(X, y)
        rank['etc'] = (etc.feature_importances_.round(3) * 100)

        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(X, y)
        set_printoptions(precision=3)
        rank['f_score'] = fit.scores_
        print(rank.sort_values('RF rank', ascending=True))

        # opções de listas de features selecionadas para cada estimador
        lista_comb_feat_RFR = []
        lista_comb_feat_RFrank = []
        lista_comb_feat_linreg = []
        lista_comb_feat_logreg = []
        lista_comb_feat_etc = []
        lista_comb_feat_f_score = []
        for x in range(2, 11):
            lista_comb_feat_RFR.append(rank.sort_values('RFR', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_RFrank.append(rank.sort_values('RF rank', ascending=True).head(x)['features'].tolist())
            lista_comb_feat_linreg.append(rank.sort_values('linreg', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_logreg.append(rank.sort_values('logreg', ascending=True).head(x)['features'].tolist())
            lista_comb_feat_etc.append(rank.sort_values('etc', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_f_score.append(rank.sort_values('f_score', ascending=False).head(x)['features'].tolist())

        return lista_comb_feat_RFrank

    def test_feat(self, lista_features, port1, port2, port3, id1, id2, id3):
        """
        Testa uma lista de features para o modelo kmeans
        testes de 4 a 8 grupos para cada uma das combinações de features passadas na lista
        Quantile transformer > PCA a 95% > Resulta na % de recomendação da lista total
        e na % de similaridade
        """
        df_labeled = pd.read_csv('../output/df_labeled.csv')
        id_df = pd.read_csv('../output/group.csv')
        id_df = id_df['indices'].copy()

        for i in range(4, 9):

            for x in range(0, len(lista_features)):
                rep_df = df_labeled[lista_features[x]]
                rep_port1 = port1[lista_features[x]]
                rep_port2 = port2[lista_features[x]]
                rep_port3 = port3[lista_features[x]]

                # QuantileTransformerpossui um output_distribution, parâmetro adicional que permite corresponder uma distribuição gaussiana em vez de uma distribuição uniforme. Observe que esse transformador não paramétrico apresenta artefatos de saturação para valores extremos.
                QtTrsfm = QuantileTransformer()
                X = QtTrsfm.fit_transform(rep_df)
                X1 = QtTrsfm.transform(rep_port1)
                X2 = QtTrsfm.transform(rep_port2)
                X3 = QtTrsfm.transform(rep_port3)

                pca = PCA(0.95)
                pca.fit(X)

                new_feat = pca.transform(X)
                new_feat1 = pca.transform(X1)
                new_feat2 = pca.transform(X2)
                new_feat3 = pca.transform(X3)

                X = new_feat.copy()
                X1 = new_feat1.copy()
                X2 = new_feat2.copy()
                X3 = new_feat3.copy()

                k_means = KMeans(n_clusters=i, max_iter=10, n_init=30, random_state=42)
                k_means.fit(X)

                pred = k_means.predict(X)
                group = pd.DataFrame(pred)
                indice_principal = id_df.reset_index(drop=True)
                group['indices'] = indice_principal

                pred1 = k_means.predict(X1)
                pred2 = k_means.predict(X2)
                pred3 = k_means.predict(X3)

                recomender_1 = Counter(pred1)
                recomender_2 = Counter(pred2)
                recomender_3 = Counter(pred3)

                
                # retorna o grupo mais frequente de cada empresa
                leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
                leads_port1 = leads_port1.append(group[group[0] == recomender_1.most_common()[1][0]]['indices'])

                leads_port2 = group[group[0] == recomender_2.most_common(1)[0][0]]['indices']

                leads_port3 = group[group[0] == recomender_3.most_common(1)[0][0]]['indices']

                lista_resultados_RFrank = []
                
                # avaliação da % do tamanho da recomendação, quantos % do total da lista foi recomendado
                rec1 = round(len(leads_port1) / len(group) * 100, 2)
                rec2 = round(len(leads_port2) / len(group) * 100, 2)
                rec3 = round(len(leads_port3) / len(group) * 100, 2)
                
                # avaliaçã da similaridade, confere quantos % da recomendação já é cliente
                sim1 = round(len(id1.loc[id1.isin(leads_port1)]) / len(id1) * 100, 2)
                sim2 = round(len(id2.loc[id2.isin(leads_port2)]) / len(id2) * 100, 2)
                sim3 = round(len(id3.loc[id3.isin(leads_port3)]) / len(id3) * 100, 2)

                lista_resultados_RFrank.append(
                    f'média de recomendação :{round((rec1 + rec2 + rec3) / 3, 2)}%- média similar :{round((sim1 + sim2 + sim3) / 3, 2)}% Nº grupos :{i}- comb feat :{x}')

                print(lista_resultados_RFrank)
                print('--------------------------------------------------------------------------------')


