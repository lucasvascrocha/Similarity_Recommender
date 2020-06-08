import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections


class Feature_engineering:
    """
    Feature engineering
    """

    def __init__(self):
        pass

    def change_columns(self, df):
        """
        Entre com o data frame principal ou de teste ou de previsão para alterar as colunas

        Em todas as features que estavam desbalanceadas foi feito um rebalanceamento manual
        através de junção de features de menor frequência ou redução de features dominantes

        """

        # de natureza jurídica, unindo todos que não são a mais frequente
        df['de_natureza_juridica_change'] = df['de_natureza_juridica'].copy()
        df['de_natureza_juridica_change'].loc[df.de_natureza_juridica_change != 'EMPRESARIO INDIVIDUAL'] = 'Outros'

        # de ramo
        df['de_ramo_change'] = df['de_ramo'].copy()
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS DE ALOJAMENTO/ALIMENTACAO'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS ADMINISTRATIVOS'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[
            df.de_ramo_change == 'SERVICOS PROFISSIONAIS, TECNICOS E CIENTIFICOS'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS DE EDUCACAO'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS FINANCEIROS'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS DE SANEAMENTO BASICO'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[df.de_ramo_change == 'SERVICOS SOCIAIS'] = 'SERVICOS DIVERSOS'
        df['de_ramo_change'].loc[
            (df.de_ramo_change != 'COMERCIO VAREJISTA') & (df.de_ramo_change != 'SERVICOS DIVERSOS')] = 'Outros'

        # setor
        df['setor_change'] = df['setor'].copy()
        df['setor_change'].loc[(df.setor_change != 'COMERCIO') & (df.setor_change != 'SERVIÇO')] = 'Outros'

        # idade empresa
        df['idade_change'] = pd.cut(df['idade_empresa_anos'], bins=[0, 2.002, 4.15, 6.934, 10.71, 19.81,
                                                                    106.44],
                                    labels=['0.24 ⊢ 2.00', '2.00 ⊢ 4.15', '4.15 ⊢ 6.93', '6.93 ⊢ 10.71',
                                            '10.71 ⊢ 19.81', '19.81 ⊢ 106.44'])
        df['idade_change'] = df['idade_change'].astype(object)

        # nm divisão
        df['nm_divisao_change'] = df['nm_divisao'].copy()
        df['nm_divisao_change'].loc[df.nm_divisao_change == 'COMERCIO VAREJISTA'] = 'Comercio Varegista' + '' + df[
            'idade_change']
        demais = ['FABRICACAO DE PRODUTOS DE MINERAIS NAO METALICOS', 'FABRICACAO DE PRODUTOS DE MADEIRA',
                  'IMPRESSAO E REPRODUCAO DE GRAVACOES', 'ATIVIDADES ARTISTICAS CRIATIVAS E DE ESPETACULOS',
                  'ARMAZENAMENTO E ATIVIDADES AUXILIARES DOS TRANSPORTES',
                  'ATIVIDADES AUXILIARES DOS SERVICOS FINANCEIROS SEGUROS PREVIDENCIA COMPLEMENTAR E PLANOS DE SAUDE',
                  'OBRAS DE INFRA ESTRUTURA', 'ATIVIDADES DE SERVICOS FINANCEIROS',
                  'CORREIO E OUTRAS ATIVIDADES DE ENTREGA',
                  'ATIVIDADES DE SEDES DE EMPRESAS E DE CONSULTORIA EM GESTAO EMPRESARIAL', 'TELECOMUNICACOES',
                  'EDICAO E EDICAO INTEGRADA A IMPRESSAO', 'ATIVIDADES DOS SERVICOS DE TECNOLOGIA DA INFORMACAO',
                  'ATIVIDADES DE ATENCAO A SAUDE HUMANA INTEGRADAS COM ASSISTENCIA SOCIAL PRESTADAS EM RESIDENCIAS COLETIVAS E PARTICULARES',
                  'FABRICACAO DE PRODUTOS TEXTEIS', 'TRANSPORTE AQUAVIARIO',
                  'ATIVIDADES DE VIGILANCIA SEGURANCA E INVESTIGACAO',
                  'COLETA TRATAMENTO E DISPOSICAO DE RESIDUOS RECUPERACAO DE MATERIAIS',
                  'ATIVIDADES DE PRESTACAO DE SERVICOS DE INFORMACAO', 'EXTRACAO DE MINERAIS NAO METALICOS',
                  'SERVICOS DOMESTICOS', 'SELECAO AGENCIAMENTO E LOCACAO DE MAO DE OBRA',
                  'ATIVIDADES CINEMATOGRAFICAS PRODUCAO DE VIDEOS E DE PROGRAMAS DE TELEVISAO GRAVACAO DE SOM E EDICAO DE MUSICA',
                  'PRODUCAO FLORESTAL', 'ELETRICIDADE GAS E OUTRAS UTILIDADES', 'FABRICACAO DE PRODUTOS QUIMICOS',
                  'ATIVIDADES DE RADIO E DE TELEVISAO', 'PESCA E AQUICULTURA',
                  'FABRICACAO DE PRODUTOS DE BORRACHA E DE MATERIAL PLASTICO',
                  'PREPARACAO DE COUROS E FABRICACAO DE ARTEFATOS DE COURO ARTIGOS PARA VIAGEM E CALCADOS',
                  'SERVICOS DE ASSISTENCIA SOCIAL SEM ALOJAMENTO', 'CAPTACAO TRATAMENTO E DISTRIBUICAO DE AGUA',
                  'FABRICACAO DE CELULOSE PAPEL E PRODUTOS DE PAPEL',
                  'FABRICACAO DE VEICULOS AUTOMOTORES REBOQUES E CARROCERIAS', 'FABRICACAO DE BEBIDAS',
                  'SEGUROS RESSEGUROS PREVIDENCIA COMPLEMENTAR E PLANOS DE SAUDE',
                  'FABRICACAO DE OUTROS EQUIPAMENTOS DE TRANSPORTE EXCETO VEICULOS AUTOMOTORES',
                  'ATIVIDADES VETERINARIAS', 'FABRICACAO DE EQUIPAMENTOS DE INFORMATICA PRODUTOS ELETRONICOS E OPTICOS',
                  'FABRICACAO DE MAQUINAS E EQUIPAMENTOS', 'ESGOTO E ATIVIDADES RELACIONADAS',
                  'EXTRACAO DE MINERAIS METALICOS', 'TRANSPORTE AEREO', 'METALURGIA',
                  'FABRICACAO DE MAQUINAS APARELHOS E MATERIAIS ELETRICOS',
                  'ATIVIDADES LIGADAS AO PATRIMONIO CULTURAL E AMBIENTAL', 'PESQUISA E DESENVOLVIMENTO CIENTIFICO',
                  'ATIVIDADES DE EXPLORACAO DE JOGOS DE AZAR E APOSTAS', 'ATIVIDADES DE APOIO A EXTRACAO DE MINERAIS',
                  'FABRICACAO DE COQUE DE PRODUTOS DERIVADOS DO PETROLEO E DE BIOCOMBUSTIVEIS',
                  'EXTRACAO DE PETROLEO E GAS NATURAL', 'FABRICACAO DE PRODUTOS FARMOQUIMICOS E FARMACEUTICOS',
                  'DESCONTAMINACAO E OUTROS SERVICOS DE GESTAO DE RESIDUOS',
                  'ORGANISMOS INTERNACIONAIS E OUTRAS INSTITUICOES EXTRATERRITORIAIS', 'FABRICACAO DE PRODUTOS DO FUMO',
                  'EXTRACAO DE CARVAO MINERAL']
        demais2 = ['FABRICACAO DE PRODUTOS ALIMENTICIOS', 'PUBLICIDADE E PESQUISA DE MERCADO',
                   'REPARACAO E MANUTENCAO DE EQUIPAMENTOS DE INFORMATICA E COMUNICACAO E DE OBJETOS PESSOAIS E DOMESTICOS',
                   'CONFECCAO DE ARTIGOS DO VESTUARIO E ACESSORIOS',
                   'SERVICOS PARA EDIFICIOS E ATIVIDADES PAISAGISTICAS',
                   'ATIVIDADES JURIDICAS DE CONTABILIDADE E DE AUDITORIA',
                   'ATIVIDADES ESPORTIVAS E DE RECREACAO E LAZER',
                   'FABRICACAO DE PRODUTOS DE METAL EXCETO MAQUINAS E EQUIPAMENTOS',
                   'ALUGUEIS NAO IMOBILIARIOS E GESTAO DE ATIVOS INTANGIVEIS NAO FINANCEIROS',
                   'ADMINISTRACAO PUBLICA DEFESA E SEGURIDADE SOCIAL', 'ATIVIDADES IMOBILIARIAS', 'ALOJAMENTO',
                   'MANUTENCAO REPARACAO E INSTALACAO DE MAQUINAS E EQUIPAMENTOS',
                   'OUTRAS ATIVIDADES PROFISSIONAIS CIENTIFICAS E TECNICAS',
                   'AGENCIAS DE VIAGENS OPERADORES TURISTICOS E SERVICOS DE RESERVAS', 'FABRICACAO DE MOVEIS',
                   'SERVICOS DE ARQUITETURA E ENGENHARIA TESTES E ANALISES TECNICAS',
                   'AGRICULTURA PECUARIA E SERVICOS RELACIONADOS', 'FABRICACAO DE PRODUTOS DIVERSOS',
                   'FABRICACAO DE PRODUTOS DE MINERAIS NAO METALICOS', 'FABRICACAO DE PRODUTOS DE MADEIRA',
                   'IMPRESSAO E REPRODUCAO DE GRAVACOES', 'ATIVIDADES ARTISTICAS CRIATIVAS E DE ESPETACULOS',
                   'ARMAZENAMENTO E ATIVIDADES AUXILIARES DOS TRANSPORTES',
                   'ATIVIDADES AUXILIARES DOS SERVICOS FINANCEIROS SEGUROS PREVIDENCIA COMPLEMENTAR E PLANOS DE SAUDE',
                   'OBRAS DE INFRA ESTRUTURA', 'ATIVIDADES DE SERVICOS FINANCEIROS',
                   'CORREIO E OUTRAS ATIVIDADES DE ENTREGA',
                   'ATIVIDADES DE SEDES DE EMPRESAS E DE CONSULTORIA EM GESTAO EMPRESARIAL', 'TELECOMUNICACOES',
                   'EDICAO E EDICAO INTEGRADA A IMPRESSAO', 'ATIVIDADES DOS SERVICOS DE TECNOLOGIA DA INFORMACAO',
                   'ATIVIDADES DE ATENCAO A SAUDE HUMANA INTEGRADAS COM ASSISTENCIA SOCIAL PRESTADAS EM RESIDENCIAS COLETIVAS E PARTICULARES',
                   'FABRICACAO DE PRODUTOS TEXTEIS', 'TRANSPORTE AQUAVIARIO',
                   'ATIVIDADES DE VIGILANCIA SEGURANCA E INVESTIGACAO',
                   'COLETA TRATAMENTO E DISPOSICAO DE RESIDUOS RECUPERACAO DE MATERIAIS',
                   'ATIVIDADES DE PRESTACAO DE SERVICOS DE INFORMACAO', 'EXTRACAO DE MINERAIS NAO METALICOS',
                   'SERVICOS DOMESTICOS', 'SELECAO AGENCIAMENTO E LOCACAO DE MAO DE OBRA',
                   'ATIVIDADES CINEMATOGRAFICAS PRODUCAO DE VIDEOS E DE PROGRAMAS DE TELEVISAO GRAVACAO DE SOM E EDICAO DE MUSICA',
                   'PRODUCAO FLORESTAL', 'ELETRICIDADE GAS E OUTRAS UTILIDADES', 'FABRICACAO DE PRODUTOS QUIMICOS',
                   'ATIVIDADES DE RADIO E DE TELEVISAO', 'PESCA E AQUICULTURA',
                   'FABRICACAO DE PRODUTOS DE BORRACHA E DE MATERIAL PLASTICO',
                   'PREPARACAO DE COUROS E FABRICACAO DE ARTEFATOS DE COURO ARTIGOS PARA VIAGEM E CALCADOS',
                   'SERVICOS DE ASSISTENCIA SOCIAL SEM ALOJAMENTO', 'CAPTACAO TRATAMENTO E DISTRIBUICAO DE AGUA',
                   'FABRICACAO DE CELULOSE PAPEL E PRODUTOS DE PAPEL',
                   'FABRICACAO DE VEICULOS AUTOMOTORES REBOQUES E CARROCERIAS', 'FABRICACAO DE BEBIDAS',
                   'SEGUROS RESSEGUROS PREVIDENCIA COMPLEMENTAR E PLANOS DE SAUDE',
                   'FABRICACAO DE OUTROS EQUIPAMENTOS DE TRANSPORTE EXCETO VEICULOS AUTOMOTORES',
                   'ATIVIDADES VETERINARIAS',
                   'FABRICACAO DE EQUIPAMENTOS DE INFORMATICA PRODUTOS ELETRONICOS E OPTICOS',
                   'FABRICACAO DE MAQUINAS E EQUIPAMENTOS', 'ESGOTO E ATIVIDADES RELACIONADAS',
                   'EXTRACAO DE MINERAIS METALICOS', 'TRANSPORTE AEREO', 'METALURGIA',
                   'FABRICACAO DE MAQUINAS APARELHOS E MATERIAIS ELETRICOS',
                   'ATIVIDADES LIGADAS AO PATRIMONIO CULTURAL E AMBIENTAL', 'PESQUISA E DESENVOLVIMENTO CIENTIFICO',
                   'ATIVIDADES DE EXPLORACAO DE JOGOS DE AZAR E APOSTAS', 'ATIVIDADES DE APOIO A EXTRACAO DE MINERAIS',
                   'FABRICACAO DE COQUE DE PRODUTOS DERIVADOS DO PETROLEO E DE BIOCOMBUSTIVEIS',
                   'EXTRACAO DE PETROLEO E GAS NATURAL', 'FABRICACAO DE PRODUTOS FARMOQUIMICOS E FARMACEUTICOS',
                   'DESCONTAMINACAO E OUTROS SERVICOS DE GESTAO DE RESIDUOS',
                   'ORGANISMOS INTERNACIONAIS E OUTRAS INSTITUICOES EXTRATERRITORIAIS',
                   'FABRICACAO DE PRODUTOS DO FUMO', 'EXTRACAO DE CARVAO MINERAL']
        df['nm_divisao_change'].loc[
            df['nm_divisao_change'].str.split().map(lambda x: x[-1]) == 'MOTOCICLETAS'] = 'MOTOCICLETAS'
        df['nm_divisao_change'].loc[(df['nm_divisao_change'].str.split().map(lambda x: x[-1]) == 'CONSTRUCAO') + (
                    df['nm_divisao_change'].str.split().map(
                        lambda x: x[-1]) == 'EDIFICIOS')] = '#Construção e edifícios'
        df['nm_divisao_change'].loc[(df['nm_divisao_change'].str.split().map(lambda x: x[-1]) == 'ALIMENTACAO') + (
                    df['nm_divisao_change'].str.split().map(lambda x: x[-1]) == 'ALIMENTICIOS')] = '#alimentos'
        df['nm_divisao_change'].loc[(df['nm_divisao_change'].str.split().map(lambda x: x[0]) == 'FABRICACAO') + (
                    df['nm_divisao_change'].str.split().map(lambda x: x[0]) == 'PRODUCAO')] = '#fabr e prod'
        df['nm_divisao_change'].loc[(
                                                df.nm_divisao_change == 'REPARACAO E MANUTENCAO DE EQUIPAMENTOS DE INFORMATICA E COMUNICACAO E DE OBJETOS PESSOAIS E DOMESTICOS') + (
                                                df.nm_divisao_change == 'MANUTENCAO REPARACAO E INSTALACAO DE MAQUINAS E EQUIPAMENTOS')] = '#Reparo'
        df['nm_divisao_change'].loc[df['nm_divisao_change'].isin(demais)] = 'Demais'
        df['nm_divisao_change'].loc[df['nm_divisao_change'].isin(demais2)] = 'Demais2'

        # nm segmento
        df['nm_segmento_change'] = df['nm_segmento'].copy()
        df['nm_segmento_change'].loc[
            df.nm_segmento_change == 'COMERCIO; REPARACAO DE VEICULOS AUTOMOTORES E MOTOCICLETAS'] = 'most_comom' + '' + \
                                                                                                     df['idade_change']
        df['nm_segmento_change'].loc[df.nm_segmento_change == 'OUTRAS ATIVIDADES DE SERVICOS'] = '#Outras' + '' + df[
            'idade_change']
        reduce = [('ARTES CULTURA ESPORTE E RECREACAO', 5380), ('INFORMACAO E COMUNICACAO', 5299),
                  ('ATIVIDADES FINANCEIRAS DE SEGUROS E SERVICOS RELACIONADOS', 3219),
                  ('AGRICULTURA PECUARIA PRODUCAO FLORESTAL PESCA E AQUICULTURA', 3180),
                  ('ADMINISTRACAO PUBLICA DEFESA E SEGURIDADE SOCIAL', 2726), ('ATIVIDADES IMOBILIARIAS', 2619),
                  ('AGUA ESGOTO ATIVIDADES DE GESTAO DE RESIDUOS E DESCONTAMINACAO', 1216),
                  ('INDUSTRIAS EXTRATIVAS', 889), ('SERVICOS DOMESTICOS', 645), ('ELETRICIDADE E GAS', 588),
                  ('ORGANISMOS INTERNACIONAIS E OUTRAS INSTITUICOES EXTRATERRITORIAIS', 9)]
        reduce = pd.DataFrame(reduce)[0].tolist()
        df['nm_segmento_change'].loc[df['nm_segmento_change'].isin(reduce)] = '#Outros'

        # de nivel atividade
        df['de_nivel_atividade_change'] = df['de_nivel_atividade'].copy()
        df['de_nivel_atividade_change'].loc[df.de_nivel_atividade_change == 'MUITO BAIXA'] = 'BAIXA'

        return df

    def encoder(self, df_colum_changed):

        """
        Aplica label encoder nas features categóricas
        não retorna features numéricas

        """

        explor_cut = pd.DataFrame({'nomes': df_colum_changed.columns,
                                   'tipos': df_colum_changed.dtypes, 'Nunique': df_colum_changed.nunique()})

        list_name_encoder = (list(explor_cut[explor_cut['tipos'] == 'object']['nomes']) + list(
            explor_cut[explor_cut['tipos'] == 'bool']['nomes']))
        list_name_encoder = list_name_encoder[:]
        lista_encoder = df_colum_changed[list_name_encoder]

        encoder_dict = collections.defaultdict(LabelEncoder)
        labeled_df = lista_encoder.apply(lambda x: encoder_dict[x.name].fit_transform(x))

        return labeled_df

    def predic_encoder(self, pred):
        df = pd.read_csv('../output/for_encoder.csv')
        encoder_dict = collections.defaultdict(LabelEncoder)
        labeled_df = df.apply(lambda x: encoder_dict[x.name].fit_transform(x))
        labeled_pred = pred.apply(lambda x: encoder_dict[x.name].transform(x))

        return labeled_pred

