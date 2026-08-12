# src/07_engenharia_temporal.py
# =============================================================================
# ETAPA 7 — ENGENHARIA DE FEATURES TEMPORAIS
# =============================================================================
# Constrói um painel (delegacia x semana) com features de defasagem (lag),
# seguindo a estrutura de dados recomendada no documento de referência
# (célula_id, ocorrencias_7d, ocorrencias_30d, dia_semana, etc.), adaptada
# para a granularidade delegacia x semana dado o volume real de dados.
# =============================================================================
import pandas as pd
import numpy as np
import joblib

# ── 1. CARREGAMENTO ────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/df_sp_limpo.csv', sep=';')
df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df.dropna(subset=['DATA_OCORRENCIA_BO', 'NOME_DELEGACIA_CIRC'])

df['SEMANA'] = df['DATA_OCORRENCIA_BO'].dt.to_period('W').apply(lambda p: p.start_time)

# ── 2. GRADE COMPLETA delegacia x semana (inclui semanas com ZERO ocorrências) ─
delegacias = df['NOME_DELEGACIA_CIRC'].unique()
semanas = pd.date_range(df['SEMANA'].min(), df['SEMANA'].max(), freq='W-MON')
grade_completa = pd.MultiIndex.from_product(
    [delegacias, semanas], names=['NOME_DELEGACIA_CIRC', 'SEMANA']
).to_frame(index=False)

contagem_real = (
    df.groupby(['NOME_DELEGACIA_CIRC', 'SEMANA'])
    .size().reset_index(name='ocorrencias_na_semana')
)
painel = grade_completa.merge(contagem_real, on=['NOME_DELEGACIA_CIRC', 'SEMANA'], how='left')
painel['ocorrencias_na_semana'] = painel['ocorrencias_na_semana'].fillna(0).astype(int)
painel = painel.sort_values(['NOME_DELEGACIA_CIRC', 'SEMANA']).reset_index(drop=True)

# ── 3. FEATURES DE LAG (defasagem) — por delegacia ─────────────────────────
g_ocorrencias = painel.groupby('NOME_DELEGACIA_CIRC')['ocorrencias_na_semana']

painel['lag_1_semana']   = g_ocorrencias.shift(1)
painel['lag_2_semanas']  = g_ocorrencias.shift(2)
painel['media_4_semanas'] = g_ocorrencias.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
painel['media_8_semanas'] = g_ocorrencias.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
painel['tendencia']      = painel['lag_1_semana'] - painel['lag_2_semanas']

# ── 4. FEATURES DE CALENDÁRIO ───────────────────────────────────────────────
painel['MES']        = painel['SEMANA'].dt.month
painel['SEMANA_ANO'] = painel['SEMANA'].dt.isocalendar().week.astype(int)

# ── 5. FEATURE ESPACIAL: SCORE DO KDE JÁ TREINADO ───────────────────────────
kde = joblib.load('models/kde_model.pkl')
centroides = (
    df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    .groupby('NOME_DELEGACIA_CIRC')[['LATITUDE', 'LONGITUDE']]
    .mean().reset_index()
)
coords_rad = np.radians(centroides[['LATITUDE', 'LONGITUDE']].values)
centroides['score_kde'] = np.exp(kde.score_samples(coords_rad))

painel = painel.merge(centroides[['NOME_DELEGACIA_CIRC', 'score_kde']],
                       on='NOME_DELEGACIA_CIRC', how='left')
painel['score_kde'] = painel['score_kde'].fillna(0)  # delegacia sem coords conhecidas

# ── 6. SALVAMENTO ─────────────────────────────────────────────────────────
painel.to_csv('data/processed/painel_delegacia_semana.csv', index=False)
print(f"Painel gerado: {len(painel)} linhas ({len(delegacias)} delegacias x {len(semanas)} semanas)")
print(f"Percentual de semanas com zero ocorrências: "
      f"{100*(painel['ocorrencias_na_semana']==0).mean():.1f}%")
print("\n[OK] Etapa 7 concluída.")
