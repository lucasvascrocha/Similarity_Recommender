import pandas as pd
from collections import Counter
import streamlit as st


class Recomendation:
    def __init__(self):
        pass

    def leads(self, pred, group):
        """
        Retorna um df com o índice e características das recomendações
        Entrada é o data frame predito com os grupos e índices
        E o data frame gerado pelo modelo com o grupo e índices de todos os dados
        """

        recomender_1 = Counter(pred['grupo'])

        if recomender_1[0] > (len(pred) / 2):
            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
        else:
            leads_port1 = group[group[0] == recomender_1.most_common(1)[0][0]]['indices']
            leads_port1 = leads_port1.append(group[group[0] == recomender_1.most_common()[1][0]]['indices'])

        id1 = pred['id']

        st.markdown(f'Foi recomendado: {round(len(leads_port1) / len(group) * 100, 2)} % da lista total que são: {round(len(id1.loc[id1.isin(leads_port1)]) / len(id1) * 100, 2)} % similares a seus clientes')

        feat_recomendadas = pd.read_csv('df_comparar_recomendacao.csv')
        feat_recomendadas = feat_recomendadas.loc[feat_recomendadas['id'].isin(leads_port1)]

        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('nm_segmento_change')
        st.markdown(
            f'seus clientes: {Counter(pred["nm_segmento_change"]).most_common()[0][0]} = {round(Counter(pred["nm_segmento_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}% e {Counter(pred["nm_segmento_change"]).most_common()[1][0]} = {round(Counter(pred["nm_segmento_change"]).most_common()[1][1] / (len(pred)) * 100, 0)}% ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["nm_segmento_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["nm_segmento_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% e {Counter(feat_recomendadas["nm_segmento_change"]).most_common()[1][0]} = {round(Counter(feat_recomendadas["nm_segmento_change"]).most_common()[1][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('de_ramo_change')
        st.markdown(
            f'seus clientes: {Counter(pred["de_ramo_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('de_natureza_juridica_change')
        st.markdown(
            f'seus clientes: {Counter(pred["de_natureza_juridica_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["de_natureza_juridica_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('nm_divisao_change')
        st.markdown(
            f'seus clientes: {Counter(pred["nm_divisao_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["nm_divisao_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('de_nivel_atividade_change')
        st.markdown(
            f'seus clientes: {Counter(pred["de_nivel_atividade_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["de_nivel_atividade_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')
        st.markdown('--------------------------------------------------------------------------------')
        st.markdown('idade_change')
        st.markdown(
            f'seus clientes: {Counter(pred["idade_change"]).most_common()[0][0]} = {round(Counter(pred["de_ramo_change"]).most_common()[0][1] / (len(pred)) * 100, 0)}%  ')
        st.markdown(
            f'recomendados: {Counter(feat_recomendadas["idade_change"]).most_common()[0][0]} = {round(Counter(feat_recomendadas["de_ramo_change"]).most_common()[0][1] / (len(feat_recomendadas)) * 100, 0)}% ')

        return feat_recomendadas
