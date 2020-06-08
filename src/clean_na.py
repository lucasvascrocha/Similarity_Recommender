import pandas as pd



class Clean_na:
    """
    Pré processamento dos dados
    """

    def __init__(self):
        pass

    def drop_col(self, x, df):
        """
            x = valor de corte para colunas, os valores maiores que x terão as colunas dropadas o restante que ainda apresentar Na terá a linha dropada
            df = data frame
        """
        exploracao = pd.DataFrame({'nomes': df.columns, 'tipos': df.dtypes,
                                   'NA #': df.isna().sum(), 'NA %': (df.isna().sum() / df.shape[0]) * 100})

        lista_colunas = list(exploracao[exploracao['NA %'] <= x]['nomes'])
        df_drop_colunas = df[lista_colunas]
        df_drop_colunas = df_drop_colunas.dropna()

        return df_drop_colunas

    def drop_low_var(self, x, df):
        """
        Dropa as features com baixa variação
        x = limit de porcentagem para o corte valores de variação <= x terão as features dropadas
        adicionar x como número inteiro de 0 a 100
        """

        expl_var = pd.DataFrame({'nomes': df.var().index, 'var': df.var().values, 'tipo': df[df.var().index].dtypes})
        lista_drop_colunas = list(expl_var.loc[(expl_var.tipo == bool) + (expl_var.tipo == object)]['nomes'])
        df.drop(lista_drop_colunas, axis=1, inplace=True)

        return df
