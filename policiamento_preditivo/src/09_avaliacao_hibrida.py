# src/09_avaliacao_hibrida.py
import pandas as pd
import numpy as np
import joblib

painel = pd.read_csv('data/processed/painel_delegacia_semana.csv', parse_dates=['SEMANA'])
modelo = joblib.load('models/xgboost_model.pkl')

FEATURES = ['lag_1_semana', 'lag_2_semanas', 'media_4_semanas',
            'media_8_semanas', 'tendencia', 'MES', 'SEMANA_ANO', 'score_kde']

def avaliar_conjunto(nome_periodo, data_ini, data_fim):
    conj = painel[(painel['SEMANA'] >= data_ini) & (painel['SEMANA'] < data_fim)].copy()
    conj[FEATURES] = conj[FEATURES].fillna(0)
    conj['pred_xgb'] = modelo.predict(conj[FEATURES])

    resultados = []
    for n_pct in [0.10, 0.20, 0.30]:
        por_delegacia = conj.groupby('NOME_DELEGACIA_CIRC').agg(
            real=('ocorrencias_na_semana', 'sum'),
            previsto_xgb=('pred_xgb', 'sum'),
            previsto_kde=('score_kde', 'mean')
        ).reset_index()

        n_top = max(1, int(len(por_delegacia) * n_pct))

        for coluna_score, nome_modelo in [('previsto_xgb', 'XGBoost híbrido'),
                                            ('previsto_kde', 'KDE isolado')]:
            top = por_delegacia.nlargest(n_top, coluna_score)
            capturado = top['real'].sum()
            total_real = por_delegacia['real'].sum()
            precision = capturado / total_real if total_real > 0 else 0
            pai = precision / n_pct
            resultados.append({
                'periodo': nome_periodo, 'modelo': nome_modelo, 'n_pct': n_pct,
                'precision_at_n': round(precision, 4), 'pai': round(pai, 4),
                'ocorrencias_capturadas': int(capturado), 'total_real': int(total_real)
            })
    return resultados

todos_resultados = (
    avaliar_conjunto('Validação Abr', '2026-04-01', '2026-05-01') +
    avaliar_conjunto('Teste Mai', '2026-05-01', '2026-06-01')
)

df_resultados = pd.DataFrame(todos_resultados)
df_resultados.to_csv('data/processed/metricas_hibrido.csv', index=False)

print(df_resultados.to_string(index=False))
print("\n[OK] Métricas salvas em data/processed/metricas_hibrido.csv")
print("[OK] Etapa 9 concluída.")
print("\nCompare este resultado com data/processed/metricas_avaliacao.csv (KDE original).")
print("O modelo híbrido só deve substituir o KDE isolado se o PAI for consistentemente maior.")
