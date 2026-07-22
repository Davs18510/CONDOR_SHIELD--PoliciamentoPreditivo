# =============================================================================
# ETAPA 4 — AVALIAÇÃO DO MODELO
# =============================================================================
# Mede o quão bem o KDE (treinado em Jan–Mar) prevê onde os crimes
# ocorrerão em Abril (validação) e Maio (teste).
#
# Métricas:
#   - Precision@N (P@N): % de ocorrências reais que caem nos N% de células
#     com maior densidade prevista
#   - PAI (Predictive Accuracy Index): P@N ÷ proporção da área marcada
#     PAI > 1 → melhor que aleatório | PAI = 3 → 3× mais eficiente
#
# Baseline: KDE treinado apenas em Março (último mês do treino)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. CARREGAMENTO ───────────────────────────────────────────────────────────
kde      = joblib.load('models/kde_model.pkl')
df_grade = pd.read_csv('data/processed/grade_densidade.csv')

df_val    = pd.read_csv('data/processed/df_validacao.csv', sep=';')
df_teste  = pd.read_csv('data/processed/df_teste.csv',    sep=';')
df_treino = pd.read_csv('data/processed/df_treino.csv',   sep=';')

# Reconverte coordenadas para float após releitura do CSV
for d in [df_val, df_teste, df_treino]:
    d['LATITUDE']  = pd.to_numeric(d['LATITUDE'],  errors='coerce')
    d['LONGITUDE'] = pd.to_numeric(d['LONGITUDE'], errors='coerce')
    d['MES']       = pd.to_datetime(
        d['DATA_OCORRENCIA_BO'], errors='coerce'
    ).dt.month

# Filtra apenas registros com coordenadas (sem coords não dá pra avaliar)
val_coords   = df_val[df_val['LATITUDE'].notna()].copy()
test_coords  = df_teste[df_teste['LATITUDE'].notna()].copy()
marco_coords = df_treino[
    (df_treino['MES'] == 3) & df_treino['LATITUDE'].notna()
].copy()

print(f"Ocorrências com coords — Validação (Abr): {len(val_coords)}")
print(f"Ocorrências com coords — Teste (Mai)    : {len(test_coords)}")
print(f"Ocorrências com coords — Março (baseline): {len(marco_coords)}")

# ── 2. FUNÇÃO DE AVALIAÇÃO P@N e PAI ─────────────────────────────────────────
def calcular_metricas(df_grade, ocorrencias, n_pct, coluna_densidade='densidade'):
    """
    Calcula Precision@N e PAI para um conjunto de ocorrências e uma grade de densidade.

    Parâmetros:
        df_grade         : DataFrame com colunas lat, lon e <coluna_densidade>
        ocorrencias      : DataFrame com colunas LATITUDE, LONGITUDE
        n_pct            : fração da área considerada hotspot (ex: 0.10 = top 10%)
        coluna_densidade : nome da coluna de densidade na grade

    Retorna:
        dicionário com as métricas calculadas
    """
    # Define o limiar: apenas as células no percentil superior são "hotspot"
    limiar = df_grade[coluna_densidade].quantile(1 - n_pct)
    hotspots = df_grade[df_grade[coluna_densidade] >= limiar]

    # Proporção da área total marcada como hotspot
    proporcao_area = len(hotspots) / len(df_grade)

    # Para cada ocorrência real, identifica a célula mais próxima na grade
    # e verifica se essa célula é um hotspot previsto
    n_dentro = 0
    for _, row in ocorrencias.iterrows():
        # Distância euclidiana simples em graus (~válida para áreas pequenas)
        dist = np.sqrt(
            (df_grade['lat'] - row['LATITUDE']) ** 2 +
            (df_grade['lon'] - row['LONGITUDE']) ** 2
        )
        # Célula mais próxima
        idx_mais_proximo = dist.idxmin()

        # Verifica se a ocorrência "cai" em um hotspot previsto
        if df_grade.loc[idx_mais_proximo, coluna_densidade] >= limiar:
            n_dentro += 1

    total = len(ocorrencias)
    precision = n_dentro / total if total > 0 else 0

    # PAI: quanto mais eficiente que aleatório é a previsão
    # PAI = 1 → igual a aleatório | PAI > 1 → melhor que aleatório
    pai = precision / proporcao_area if proporcao_area > 0 else 0

    return {
        'total_ocorrencias' : total,
        'em_hotspots'       : n_dentro,
        'precision_at_n'    : round(precision, 4),
        'proporcao_area'    : round(proporcao_area, 4),
        'pai'               : round(pai, 4)
    }

# ── 3. BASELINE — KDE treinado só em Março (último mês do treino) ────────────
# O baseline responde: "Simplesmente repetir o padrão de Março já funciona?"
# Se o KDE completo (Jan–Mar) não superar o baseline, é necessário documentar.
print("\nTreinando modelo baseline (somente Março)...")
coords_marco_rad = np.radians(marco_coords[['LATITUDE', 'LONGITUDE']].values)
kde_base = KernelDensity(kernel='gaussian', metric='haversine', bandwidth=0.008)
kde_base.fit(coords_marco_rad)

# Calcula densidade do baseline para todos os pontos da grade
df_grade['densidade_baseline'] = np.exp(
    kde_base.score_samples(
        np.radians(df_grade[['lat', 'lon']].values)
    )
)
print("Baseline treinado.")

# ── 4. AVALIAÇÃO COM MÚLTIPLOS LIMIARES ──────────────────────────────────────
# Testa com 3 limiares diferentes: top 10%, 20% e 30% da área
resultados = []

print(f"\n{'N%':>5} | {'Conjunto':<16} | {'Modelo':<14} | "
      f"{'P@N':>7} | {'PAI':>6} | {'Captadas':>10}")
print("=" * 72)

for n_pct in [0.10, 0.20, 0.30]:
    for nome_conj, conj_coords in [
        ('Validação Abr', val_coords),
        ('Teste Mai',     test_coords)
    ]:
        # ── KDE principal (treinado em Jan, Fev e Mar) ──
        r_kde = calcular_metricas(df_grade, conj_coords, n_pct, 'densidade')
        r_kde.update({
            'modelo'   : 'KDE Jan-Mar',
            'conjunto' : nome_conj,
            'n_pct'    : n_pct
        })

        # ── Baseline (treinado apenas em Mar) ──
        r_base = calcular_metricas(df_grade, conj_coords, n_pct, 'densidade_baseline')
        r_base.update({
            'modelo'   : 'Baseline Março',
            'conjunto' : nome_conj,
            'n_pct'    : n_pct
        })

        resultados.extend([r_kde, r_base])

        # Impressão formatada
        captadas_kde  = f"{r_kde['em_hotspots']}/{r_kde['total_ocorrencias']}"
        captadas_base = f"{r_base['em_hotspots']}/{r_base['total_ocorrencias']}"

        print(f"{int(n_pct*100):>4}% | {nome_conj:<16} | {'KDE Jan-Mar':<14} | "
              f"{r_kde['precision_at_n']:>7.3f} | {r_kde['pai']:>6.2f} | "
              f"{captadas_kde:>10}")
        print(f"{int(n_pct*100):>4}% | {nome_conj:<16} | {'Baseline Março':<14} | "
              f"{r_base['precision_at_n']:>7.3f} | {r_base['pai']:>6.2f} | "
              f"{captadas_base:>10}")
        print("-" * 72)

# ── 5. SALVAMENTO ─────────────────────────────────────────────────────────────
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('data/processed/metricas_avaliacao.csv', index=False)
print("\n[OK] Métricas salvas em data/processed/metricas_avaliacao.csv")

# Salva grade com coluna de baseline para uso na visualização
df_grade.to_csv('data/processed/grade_densidade.csv', index=False)

print("[OK] Etapa 4 concluída.")
