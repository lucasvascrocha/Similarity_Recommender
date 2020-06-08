import pandas as pd
import numpy as np


class DataSource:
    """
    Importa data frames já aplicando redução de tamanho no df principal
    etapa_treino = True, gerará 3 data frames
    """

    def __init__(self):
        self.path_train = '../data/estaticos_market.csv'
        self.path_test1 = '../data/estaticos_portfolio1.csv'
        self.path_test2 = '../data/estaticos_portfolio2.csv'
        self.path_test3 = '../data/estaticos_portfolio3.csv'

    def read_data(self, etapa_treino=True):
        """
            if True retorna o df treino
            
        """

        if etapa_treino:
            df = pd.read_csv(self.path_train)
            df = DataSource().reduce_mem_usage(df)

            return df

    def reduce_mem_usage(self, props):
        """
        Reduz tamanho do data frame transformando seus tipos
        props = df que deseja reduzir

        """

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


class Busca:
    """
    Busca id no data frame
    """

    def __init__(self):

        pass


    def busca_id(self):
        """
        Retorna 6 saídas, df1, 2 e 3 que estão no df principal já pre processado
        e seus rescpectivos ids
        """

        df = pd.read_csv('../output/group.csv')
        df_pronto = pd.read_csv('../output/df_labeled.csv')
        df_pronto['id'] = df['indices'].copy()

        port1 = pd.read_csv('../data/estaticos_portfolio1.csv')
        id1 = port1['id'].copy()
        port1 = df_pronto.loc[df_pronto['id'].isin(port1['id'])]
        port1 = port1.drop('id', axis=1)

        port2 = pd.read_csv('../data/estaticos_portfolio2.csv')
        id2 = port2['id'].copy()
        port2 = df_pronto.loc[df_pronto['id'].isin(port2['id'])]
        port2 = port2.drop('id', axis=1)

        port3 = pd.read_csv('../data/estaticos_portfolio3.csv')
        id3 = port3['id'].copy()
        port3 = df_pronto.loc[df_pronto['id'].isin(port3['id'])]
        port3 = port3.drop('id', axis=1)

        return port1, port2, port3, id1, id2, id3
