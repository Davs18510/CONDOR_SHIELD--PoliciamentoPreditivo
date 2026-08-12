# src/10_previsao_multi_horizonte.py
# =============================================================================
# ETAPA 10 — PREVISAO RECURSIVA MULTI-HORIZONTE E GERACAO DE PONTOS PREVISTOS
# =============================================================================
# Usa o XGBoost (Etapa 8) para prever contagens esperadas (lambda) semana a semana
# (rollout autorregressivo) para horizontes de 1 a 6 meses.
# Converte as contagens em probabilidades Poisson e gera pontos espaciais previstos
# de ocorrencia para orientar o mapa e o direcionamento de viaturas.
# =============================================================================
import pandas as pd
import numpy as np
import joblib
import json

painel = pd.read_csv('data/processed/painel_delegacia_semana.csv', parse_dates=['SEMANA'])
modelo = joblib.load('models/xgboost_model.pkl')
df_sp  = pd.read_csv('data/processed/df_sp_limpo.csv', sep=';')

df_sp['LATITUDE']  = pd.to_numeric(df_sp['LATITUDE'],  errors='coerce')
df_sp['LONGITUDE'] = pd.to_numeric(df_sp['LONGITUDE'], errors='coerce')
df_valid = df_sp.dropna(subset=['LATITUDE', 'LONGITUDE', 'NOME_DELEGACIA_CIRC'])

FEATURES = ['lag_1_semana', 'lag_2_semanas', 'media_4_semanas',
            'media_8_semanas', 'tendencia', 'MES', 'SEMANA_ANO', 'score_kde']

HORIZONTES_MESES = [1, 2, 3, 4, 5, 6]
SEMANAS_POR_MES = 4.345
MAX_SEMANAS = int(max(HORIZONTES_MESES) * SEMANAS_POR_MES) + 1

# Tipos de crime e proporção histórica para rotular pontos previstos
tipos_crime = ['TENTATIVA DE HOMICIDIO', 'HOMICIDIO DOLOSO', 'LESAO CORPORAL SEGUIDA DE MORTE', 'LATROCINIO']
prop_tipos = df_valid['NATUREZA APURADA'].value_counts(normalize=True).to_dict()
p_vec = [prop_tipos.get(t, 0.25) for t in tipos_crime]
p_vec = np.array(p_vec) / np.sum(p_vec)

np.random.seed(42)

probabilidades = {}
lambdas_mes    = {}
pred_pts_por_h = {str(h): [] for h in HORIZONTES_MESES}

# Centroides e desvio padrão geográfico histórico de cada delegacia
deleg_stats = (
    df_valid.groupby('NOME_DELEGACIA_CIRC')
    .agg(
        lat_mean=('LATITUDE', 'mean'),
        lon_mean=('LONGITUDE', 'mean'),
        lat_std=('LATITUDE', lambda s: float(np.std(s)) if len(s)>1 else 0.012),
        lon_std=('LONGITUDE', lambda s: float(np.std(s)) if len(s)>1 else 0.012)
    ).reset_index()
)

for deleg in painel['NOME_DELEGACIA_CIRC'].unique():
    hist = painel[painel['NOME_DELEGACIA_CIRC'] == deleg].sort_values('SEMANA')
    if hist.empty:
        continue

    fila4 = hist['ocorrencias_na_semana'].tail(4).tolist()
    fila8 = hist['ocorrencias_na_semana'].tail(8).tolist()
    lag1  = float(hist['ocorrencias_na_semana'].iloc[-1])
    lag2  = float(hist['ocorrencias_na_semana'].iloc[-2]) if len(hist) >= 2 else 0.0
    score_kde = float(hist['score_kde'].iloc[-1])
    data_ref  = hist['SEMANA'].iloc[-1]

    # Stats geográficas da delegacia
    d_stat = deleg_stats[deleg_stats['NOME_DELEGACIA_CIRC'] == deleg]
    if not d_stat.empty:
        lat_m = float(d_stat['lat_mean'].iloc[0])
        lon_m = float(d_stat['lon_mean'].iloc[0])
        lat_s = max(0.005, min(0.025, float(d_stat['lat_std'].iloc[0])))
        lon_s = max(0.005, min(0.025, float(d_stat['lon_std'].iloc[0])))
    else:
        lat_m, lon_m, lat_s, lon_s = -23.55, -46.63, 0.015, 0.015

    lambdas_semanais = []

    for semana_idx in range(1, MAX_SEMANAS + 1):
        data_prevista = data_ref + pd.Timedelta(weeks=semana_idx)
        linha = pd.DataFrame([{
            'lag_1_semana': lag1, 'lag_2_semanas': lag2,
            'media_4_semanas': float(np.mean(fila4)) if fila4 else 0.0,
            'media_8_semanas': float(np.mean(fila8)) if fila8 else 0.0,
            'tendencia': lag1 - lag2,
            'MES': data_prevista.month,
            'SEMANA_ANO': int(data_prevista.isocalendar()[1]),
            'score_kde': score_kde
        }])[FEATURES]

        pred = max(0.0, float(modelo.predict(linha)[0]))
        lambdas_semanais.append(pred)

        lag2 = lag1
        lag1 = pred
        fila4 = (fila4 + [pred])[-4:]
        fila8 = (fila8 + [pred])[-8:]

    probabilidades[deleg] = {}
    lambdas_mes[deleg]    = {}

    for h in HORIZONTES_MESES:
        semana_inicio = int(round((h - 1) * SEMANAS_POR_MES))
        semana_fim    = int(round(h * SEMANAS_POR_MES))
        janela = lambdas_semanais[semana_inicio:semana_fim]
        l_medio = float(np.mean(janela)) if janela else 0.0
        l_soma  = float(np.sum(janela))  if janela else 0.0

        prob_pct = round((1 - np.exp(-l_medio)) * 100, 1)
        probabilidades[deleg][str(h)] = prob_pct
        lambdas_mes[deleg][str(h)]    = round(l_soma, 2)

        # Gera pontos previstos de ocorrência para o mapa
        n_pts_previstos = int(np.round(l_soma))
        if n_pts_previstos > 0:
            for _ in range(n_pts_previstos):
                plat = round(float(np.random.normal(lat_m, lat_s)), 5)
                plon = round(float(np.random.normal(lon_m, lon_s)), 5)
                ptipo = str(np.random.choice(tipos_crime, p=p_vec))
                pred_pts_por_h[str(h)].append({
                    'lat': plat,
                    'lon': plon,
                    'tipo': ptipo,
                    'deleg': deleg,
                    'prob': prob_pct,
                    'lambda': round(l_soma, 1)
                })

dados_saida = {
    'probabilidades': probabilidades,
    'lambdas'       : lambdas_mes,
    'pred_pts'      : pred_pts_por_h
}

with open('data/processed/previsoes_xgboost_horizonte.json', 'w', encoding='utf-8') as f:
    json.dump(dados_saida, f, ensure_ascii=False)

print(f"Previsões salvas: {len(probabilidades)} delegacias x {len(HORIZONTES_MESES)} horizontes")
for h in HORIZONTES_MESES:
    print(f"  Mês {h}: {len(pred_pts_por_h[str(h)])} pontos de crimes previstos gerados")

print("\n[OK] Etapa 10 concluída.")
