# =============================================================================
# ETAPA 1 — LIMPEZA E SPLIT TEMPORAL
# =============================================================================
# Lê o CSV no formato Orange (pulando as 2 linhas de metadados),
# aplica todas as correções de qualidade, filtra São Paulo capital
# e divide os dados em treino / validação / teste por mês.
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. LEITURA CORRETA DO FORMATO ORANGE ──────────────────────────────────────
# O Orange exporta 3 linhas antes dos dados reais:
#   Linha 0 → nomes das colunas (header real)
#   Linha 1 → tipos das colunas pelo Orange ("string", "continuous"...) ← IGNORAR
#   Linha 2 → papéis das colunas pelo Orange ("meta", "class"...)       ← IGNORAR
#   Linha 3+ → dados reais
df = pd.read_csv(
    'data/raw/SIPCV_SP_limpo.csv',
    sep=',',
    dtype=str,
    encoding='utf-8-sig',
    header=0,       # linha 0 é o header
    skiprows=[1, 2] # pula as linhas de metadados do Orange
)
print(f"Lido: {df.shape[0]} registros × {df.shape[1]} colunas")

# ── 2. CORREÇÃO DE COORDENADAS ────────────────────────────────────────────────
# Problema 1: Orange exportou com vírgula como separador decimal (padrão BR)
#   Exemplo no arquivo: -23,589357  → deve ser: -23.589357
# Problema 2: Registros com sigilo judicial têm texto no lugar das coords
VEDACAO = 'VEDAÇAO DA DIVULGAÇAO DOS DADOS'

# Passo a passo para compatibilidade com pandas 2.x e 3.x:
# 1) Corrige o decimal BR (vírgula → ponto)
# 2) Substitui o texto de vedação por NaN usando where (mais robusto que .replace() com NA)
df['LATITUDE'] = df['LATITUDE'].str.replace(',', '.', regex=False)
df['LATITUDE'] = df['LATITUDE'].where(df['LATITUDE'] != VEDACAO, other=np.nan)

df['LONGITUDE'] = df['LONGITUDE'].str.replace(',', '.', regex=False)
df['LONGITUDE'] = df['LONGITUDE'].where(df['LONGITUDE'] != VEDACAO, other=np.nan)

# Converte para número — qualquer valor inválido remanescente vira NaN
df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

# ── 3. CORREÇÃO DE DATAS ───────────────────────────────────────────────────────
# O arquivo usa o formato brasileiro DD/MM/YYYY — sem dayfirst=True,
# o pandas interpretaria como MM/DD/YYYY e causaria erros silenciosos.
df['DATA_OCORRENCIA_BO'] = pd.to_datetime(
    df['DATA_OCORRENCIA_BO'],
    dayfirst=True,   # interpreta DD/MM/YYYY corretamente
    errors='coerce'
)

# ── 4. CORREÇÃO DE TEXTOS SUJOS ───────────────────────────────────────────────
# Mesmo valor escrito de formas inconsistentes em DESC_PERIODO
periodo_map = {
    'Pela manhA'      : 'Pela manhã',
    'Pelamanhã'       : 'Pela manhã',
    'Demadrugada'     : 'De madrugada',
    'Anoite'          : 'A noite',
    'Atarde'          : 'A tarde',
    'TARDE'           : 'A tarde',
    'Emhoraincerta'   : 'Em hora incerta',
    'EM HORA INCERTA' : 'Em hora incerta',
}
df['DESC_PERIODO'] = df['DESC_PERIODO'].replace(periodo_map)

# Inconsistências em DESCR_TIPOLOCAL
local_map = {
    'ResidEncia'             : 'Residência',
    'CondomInio Residencial' : 'Condomínio Residencial',
    'Area nAo Ocupada'       : 'Area não Ocupada',
    'ComErcio e Serviços'    : 'Comércio e Serviços',
    'Terminal/EstaçAo'       : 'Terminal/Estação',
}
df['DESCR_TIPOLOCAL'] = df['DESCR_TIPOLOCAL'].replace(local_map)

# ── 5. FILTRO: SÃO PAULO CAPITAL ─────────────────────────────────────────────
# O campo CIDADE identifica o município — capital = 'S.PAULO'
sp = df[df['CIDADE'] == 'S.PAULO'].copy()
print(f"Após filtro S.PAULO: {len(sp)} registros")

# ── 6. SELEÇÃO DAS COLUNAS ÚTEIS ─────────────────────────────────────────────
# Colunas demográficas de vítimas (COR_CURTIS, SEXO_PESSOA etc.) são
# excluídas propositalmente — seu uso como features preditoras é
# discriminatório (ver Seção 2.4 da especificação).
COLUNAS_UTEIS = [
    'NUM_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO',
    'NATUREZA APURADA', 'RUBRICA', 'DESC_PERIODO', 'DESCR_TIPOLOCAL',
    'LATITUDE', 'LONGITUDE', 'BAIRRO', 'LOGRADOURO',
    'NOME_DELEGACIA_CIRC', 'NOME_SECCIONAL_CIRC',
    'FLAG_STATUS_CRIME', 'FLAG_FLAGRANTE',
]

# Mantém apenas as colunas que existem no arquivo (tolerante a variações)
colunas_presentes = [c for c in COLUNAS_UTEIS if c in sp.columns]
sp = sp[colunas_presentes].copy()

# ── 7. SPLIT TEMPORAL POR MÊS ────────────────────────────────────────────────
# Dados temporais NUNCA devem ser embaralhados — o split é feito por mês
# para simular predição real: treinar no passado, validar no futuro.
sp['MES'] = sp['DATA_OCORRENCIA_BO'].dt.month
sp['ANO'] = sp['DATA_OCORRENCIA_BO'].dt.year

# Treino: Janeiro, Fevereiro e Março de 2026 (dados históricos)
df_treino    = sp[(sp['ANO'] == 2026) & (sp['MES'].isin([1, 2, 3]))].copy()

# Validação: Abril de 2026 (avaliação durante desenvolvimento)
df_validacao = sp[(sp['ANO'] == 2026) & (sp['MES'] == 4)].copy()

# Teste: Maio de 2026 (avaliação final — não usar durante o desenvolvimento!)
df_teste     = sp[(sp['ANO'] == 2026) & (sp['MES'] == 5)].copy()

# Relatório de qualidade das coordenadas por conjunto
print(f"\n{'Conjunto':<22} {'Total':>8} {'Com coords':>12}")
print("-" * 44)
for nome, sub in [('Treino (Jan-Mar 2026)', df_treino),
                   ('Validação (Abr 2026)',  df_validacao),
                   ('Teste (Mai 2026)',       df_teste)]:
    n_coords = sub['LATITUDE'].notna().sum()
    print(f"{nome:<22} {len(sub):>8} {n_coords:>12}")

# Relatório de coords para todo SP
n_total  = len(sp)
n_coords = sp['LATITUDE'].notna().sum()
n_nulos  = n_total - n_coords
print(f"\nResumo SP capital completo:")
print(f"  Total de registros     : {n_total}")
print(f"  Com coordenadas válidas: {n_coords} ({n_coords/n_total*100:.1f}%)")
print(f"  Sem coordenadas (nulos): {n_nulos} ({n_nulos/n_total*100:.1f}%)")

# ── 8. SALVAMENTO ─────────────────────────────────────────────────────────────
# Separador ponto-e-vírgula para evitar conflito com vírgulas em campos texto
sp.to_csv('data/processed/df_sp_limpo.csv',   index=False, sep=';')
df_treino.to_csv('data/processed/df_treino.csv',     index=False, sep=';')
df_validacao.to_csv('data/processed/df_validacao.csv', index=False, sep=';')
df_teste.to_csv('data/processed/df_teste.csv',       index=False, sep=';')

print("\n[OK] Etapa 1 concluída — arquivos salvos em data/processed/")
