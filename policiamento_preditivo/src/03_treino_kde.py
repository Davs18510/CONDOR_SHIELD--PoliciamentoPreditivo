# =============================================================================
# ETAPA 3 — TREINAMENTO DO MODELO KDE
# =============================================================================
# Treina um KDE (Kernel Density Estimation) espacial sobre os pontos
# de treino (Jan–Mar 2026) e gera uma grade de densidade sobre São Paulo.
#
# Conceitos técnicos importantes:
#   - metric='haversine': obrigatório para coordenadas geográficas
#   - np.radians(): sklearn haversine exige radianos (não graus)
#   - score_samples(): retorna log-probabilidade → aplicar np.exp()
#   - Ordem: sempre (latitude, longitude), nunca o contrário
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. CARREGAMENTO DOS DADOS DE TREINO ───────────────────────────────────────
df = pd.read_csv('data/processed/df_treino.csv', sep=';')

# Reforça conversão numérica (o CSV pode ter salvo como string)
df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

# Filtra apenas registros com coordenadas válidas (input do KDE)
df_coords = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()].copy()
print(f"Pontos de treino com coordenadas: {len(df_coords)}")
print(f"(Descartados por falta de coords : {len(df) - len(df_coords)})")

# ── 2. CONVERSÃO GRAUS → RADIANOS ────────────────────────────────────────────
# CRÍTICO: sklearn com metric='haversine' exige radianos — sem isso,
# os valores de densidade serão completamente inválidos.
# Shape esperado: (n_amostras, 2) onde colunas são [lat_rad, lon_rad]
coords_rad = np.radians(df_coords[['LATITUDE', 'LONGITUDE']].values)
print(f"Shape da matriz de treino: {coords_rad.shape}")

# ── 3. SELEÇÃO DO MELHOR BANDWIDTH (GridSearchCV) ─────────────────────────────
# O bandwidth controla o "raio de influência" de cada ponto:
#   - Muito pequeno → KDE muito irregular (overfitting)
#   - Muito grande  → KDE muito suavizado (perde padrões locais)
# Em radianos, 0.008 ≈ ~500m de raio no equador.
# GridSearchCV com cross-validation 5-fold seleciona o melhor automaticamente.
print("\nBuscando melhor bandwidth... (pode levar 1–2 minutos)")

bandwidths = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]

grid = GridSearchCV(
    KernelDensity(kernel='gaussian', metric='haversine'),
    param_grid={'bandwidth': bandwidths},
    cv=5,          # 5-fold cross-validation
    n_jobs=-1,     # usa todos os núcleos disponíveis
    verbose=0
)
grid.fit(coords_rad)

melhor_bw = grid.best_params_['bandwidth']
print(f"Melhor bandwidth: {melhor_bw} rad | Score CV: {grid.best_score_:.4f}")
print(f"(Equivalente aproximado: {melhor_bw * 6371:.1f} km de raio)")

# ── 4. TREINO DO MODELO FINAL ─────────────────────────────────────────────────
# Treina o modelo definitivo com todos os dados de treino e o melhor bandwidth
kde = KernelDensity(kernel='gaussian', metric='haversine', bandwidth=melhor_bw)
kde.fit(coords_rad)
print("Modelo KDE treinado com sucesso.")

# ── 5. SERIALIZAÇÃO DO MODELO ─────────────────────────────────────────────────
# Salva o modelo em disco para reutilização na avaliação e visualização
joblib.dump(kde, 'models/kde_model.pkl')
print("[OK] Modelo salvo em models/kde_model.pkl")

# ── 6. GRADE DE PREDIÇÃO (150×150 pontos cobrindo SP capital) ─────────────────
# Bounding box conservador cobrindo todo o município de São Paulo
# Referência: limites oficiais do município (~1500 km²)
LAT_MIN, LAT_MAX = -24.008, -23.357
LON_MIN, LON_MAX = -46.826, -46.365

N_PONTOS = 150  # 150×150 = 22.500 células na grade

lat_vals = np.linspace(LAT_MIN, LAT_MAX, N_PONTOS)
lon_vals = np.linspace(LON_MIN, LON_MAX, N_PONTOS)

# meshgrid cria todas as combinações lat/lon da grade
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals)

# Flattening para formato (n_pontos, 2) exigido pelo sklearn
grade_rad = np.radians(
    np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
)

# score_samples retorna log-probabilidade → np.exp() converte para densidade
print(f"\nCalculando densidade em {len(grade_rad):,} pontos da grade...")
log_densidade = kde.score_samples(grade_rad)
densidade = np.exp(log_densidade)  # log-prob → prob real (sempre positiva)

# Monta DataFrame com as coordenadas e densidades de cada célula
df_grade = pd.DataFrame({
    'lat'       : lat_grid.ravel(),
    'lon'       : lon_grid.ravel(),
    'densidade' : densidade
})

df_grade.to_csv('data/processed/grade_densidade.csv', index=False)

print(f"Grade gerada: {len(df_grade):,} pontos")
print(f"Densidade mínima : {densidade.min():.8f}")
print(f"Densidade máxima : {densidade.max():.8f}")
print(f"Relação max/min  : {densidade.max()/densidade.min():.1f}x")
print("\n[OK] Etapa 3 concluída.")
