# src/08_treino_xgboost.py
# =============================================================================
# ETAPA 8 — TREINO DO MODELO SUPERVISIONADO (XGBoost, Poisson)
# =============================================================================
# Implementa a recomendação do documento de referência (Seção 6): XGBoost
# com objetivo Poisson para prever CONTAGEM de ocorrências, não apenas
# probabilidade binária — mais informativo para priorização de patrulha.
# =============================================================================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

painel = pd.read_csv('data/processed/painel_delegacia_semana.csv', parse_dates=['SEMANA'])

# ── SPLIT CRONOLÓGICO — mesma lógica temporal já usada no projeto (Etapa 1) ──
treino    = painel[(painel['SEMANA'] >= '2026-01-01') & (painel['SEMANA'] < '2026-04-01')]
validacao = painel[(painel['SEMANA'] >= '2026-04-01') & (painel['SEMANA'] < '2026-05-01')]
teste     = painel[(painel['SEMANA'] >= '2026-05-01') & (painel['SEMANA'] < '2026-06-01')]

FEATURES = [
    'lag_1_semana', 'lag_2_semanas', 'media_4_semanas', 'media_8_semanas',
    'tendencia', 'MES', 'SEMANA_ANO', 'score_kde'
]
ALVO = 'ocorrencias_na_semana'

for conj in [treino, validacao, teste]:
    conj[FEATURES] = conj[FEATURES].fillna(0)

X_train, y_train = treino[FEATURES], treino[ALVO]
X_val,   y_val    = validacao[FEATURES], validacao[ALVO]

# ── MODELO — objetivo Poisson, adequado para contagens esparsas ─────────────
modelo = XGBRegressor(
    n_estimators=200,
    max_depth=4,              # profundidade baixa — evita overfit com poucos dados
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='count:poisson',
    eval_metric='poisson-nloglik',
    random_state=42,
    early_stopping_rounds=20
)
modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

joblib.dump(modelo, 'models/xgboost_model.pkl')

# ── IMPORTÂNCIA DAS FEATURES (explicabilidade, Seção 4/8 do doc. referência) ─
importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Importância das features:")
print(importancias.to_string())
print("\n[OK] Modelo salvo em models/xgboost_model.pkl")
print("[OK] Etapa 8 concluída.")
