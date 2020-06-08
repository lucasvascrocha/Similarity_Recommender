import pandas as pd
from collections import Counter


class Recomendation:
    def __init__(self):
        pass

    def leads(self, pred, group):
        """
        Retorna um df com o índice das recomendações
        Entradas:
        pred = o data frame predito com os grupos e índices 
        group = df gerado pelo modelo com o grupo e índices de todos os dados
        """
        # contagem da frequência dos grupos
        recomender_1 = Counter(pred['grupo'])
        
        # se apenas um grupo contemplar mais de 50% dos dados então somente ele servirá de recomendador
        if recomender_1[0] > (len(pred) / 2):
            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
        # caso haja uma distribuição maior de grupos em uma empresa usa-se os 2 grupos maiores para recomendar
        else:
            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
            leads_port1 = leads_port1.append(group[group[0] == recomender_1.most_common()[1][0]]['indices'])

        id1 = pred['id']

        print(f'Foi recomendado: {round(len(leads_port1) / len(group) * 100, 2)} % da lista total que são: {round(len(id1.loc[id1.isin(leads_port1)]) / len(id1) * 100, 2)} % similares a seus clientes')

        feat_recomendadas = pd.read_csv('../output/df_comparar_recomendacao.csv')
        feat_recomendadas = feat_recomendadas.loc[feat_recomendadas['id'].isin(leads_port1)]

        # comparação de similaridade entre características dos clientes já fidelizados com as dos recomendados

        print('--------------------------------------------------------------------------------')
        print('nm_segmento_change')
        print(
            f'seus clientes: {Counter(pred["nm_segmento_change"]).most_common()[0][0]} = {round(Counter(pred["nm_segmento_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}% e {Counter(pred["nm_segmento_change"]).most_common()[1][0]} = {round(Counter(pred["nm_segmento_change"]).most_common()[1][1] / (len(pred)) * 100, 0)}% ')
        print(
            f'recomendados: {Counter(feat_recomendadas["nm_segmento_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["nm_segmento_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% e {Counter(feat_recomendadas["nm_segmento_change"]).most_common()[1][0]} = {round(Counter(feat_recomendadas["nm_segmento_change"]).most_common()[1][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        print('--------------------------------------------------------------------------------')
        print('de_ramo_change')
        print(
            f'seus clientes: {Counter(pred["de_ramo_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        print(
            f'recomendados: {Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        print('--------------------------------------------------------------------------------')
        print('de_natureza_juridica_change')
        print(
            f'seus clientes: {Counter(pred["de_natureza_juridica_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        print(
            f'recomendados: {Counter(feat_recomendadas["de_natureza_juridica_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        print('--------------------------------------------------------------------------------')
        print('nm_divisao_change')
        print(
            f'seus clientes: {Counter(pred["nm_divisao_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        print(
            f'recomendados: {Counter(feat_recomendadas["nm_divisao_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        print('--------------------------------------------------------------------------------')
        print('de_nivel_atividade_change')
        print(
            f'seus clientes: {Counter(pred["de_nivel_atividade_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        print(
            f'recomendados: {Counter(feat_recomendadas["de_nivel_atividade_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        print('--------------------------------------------------------------------------------')
        print('idade_change')
        print(
            f'seus clientes: {Counter(pred["idade_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        print(
            f'recomendados: {Counter(feat_recomendadas["idade_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')

        return feat_recomendadas
