# =============================================================================
# ETAPA 5 — VISUALIZAÇÃO
# =============================================================================
# Gera dois mapas HTML interativos com Folium:
#   Mapa 1: HeatMap de densidade KDE (risco previsto) sobre São Paulo
#   Mapa 2: Hotspots previstos (top 20%) + ocorrências reais de Abril
# =============================================================================

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# Configurações do mapa — centro geográfico de São Paulo e tema neutro
LAT_CENTRO  = -23.5505
LON_CENTRO  = -46.6333
ZOOM_INICIAL = 12
TILE         = 'CartoDB positron'  # fundo claro, ideal para heatmaps coloridos

# ── 1. CARREGAMENTO DOS DADOS ─────────────────────────────────────────────────
df_grade = pd.read_csv('data/processed/grade_densidade.csv')

df_val = pd.read_csv('data/processed/df_validacao.csv', sep=';')
df_val['LATITUDE']  = pd.to_numeric(df_val['LATITUDE'],  errors='coerce')
df_val['LONGITUDE'] = pd.to_numeric(df_val['LONGITUDE'], errors='coerce')
val_coords = df_val[df_val['LATITUDE'].notna()].copy()

df_treino = pd.read_csv('data/processed/df_treino.csv', sep=';')
df_treino['LATITUDE']  = pd.to_numeric(df_treino['LATITUDE'],  errors='coerce')
df_treino['LONGITUDE'] = pd.to_numeric(df_treino['LONGITUDE'], errors='coerce')
treino_coords = df_treino[df_treino['LATITUDE'].notna()].copy()

print(f"Grade de densidade: {len(df_grade):,} células")
print(f"Ocorrências de validação (Abr) com coords: {len(val_coords)}")

# ── 2. MAPA 1 — HEATMAP DE RISCO (KDE Jan–Mar) ───────────────────────────────
print("\nGerando Mapa 1 — Heatmap de risco...")
mapa1 = folium.Map(
    location=[LAT_CENTRO, LON_CENTRO],
    zoom_start=ZOOM_INICIAL,
    tiles=TILE
)

# Normaliza a densidade para o intervalo [0, 1] (exigido pelo HeatMap)
d_norm = df_grade['densidade'] / df_grade['densidade'].max()

HeatMap(
    data=list(zip(df_grade['lat'], df_grade['lon'], d_norm)),
    min_opacity=0.35,
    radius=10,
    blur=12,
    # Escala de cor: azul (baixo risco) -> verde -> amarelo -> vermelho (alto risco)
    gradient={0.2: 'blue', 0.45: 'lime', 0.65: 'yellow', 1.0: 'red'},
    name='Densidade de risco (KDE Jan-Mar 2026)'
).add_to(mapa1)

# Pontos de treino (cinza, pequenos) como referência dos dados usados
camada_treino = folium.FeatureGroup(name='Pontos de treino (Jan–Mar 2026)', show=False)
for _, row in treino_coords.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=3,
        color='#555555',
        fill=True,
        fill_color='#555555',
        fill_opacity=0.4,
        popup=folium.Popup(
            f"<b>{row.get('NATUREZA APURADA', '')}</b><br>"
            f"Data: {row.get('DATA_OCORRENCIA_BO', '')}<br>"
            f"Local: {row.get('DESCR_TIPOLOCAL', '')}<br>"
            f"Delegacia: {row.get('NOME_DELEGACIA_CIRC', '')}",
            max_width=240
        )
    ).add_to(camada_treino)
camada_treino.add_to(mapa1)

# Legenda HTML embutida no mapa
legenda_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 1000;
    background: rgba(255, 255, 255, 0.95);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: Arial, sans-serif;
    font-size: 12px;
    line-height: 1.6;
">
    <b style="font-size:13px">Risco Previsto — KDE</b><br>
    <i style="font-size:10px;color:#666">Treino: Jan a Mar 2026 | SIPCV/SSP-SP</i><br><br>
    <span style="color:#1E90FF">&#9632;</span> Baixo risco<br>
    <span style="color:#32CD32">&#9632;</span> Risco moderado<br>
    <span style="color:#FFD700">&#9632;</span> Alto risco<br>
    <span style="color:#FF0000">&#9632;</span> Muito alto risco<br><br>
    <i style="font-size:10px;color:#888">Crimes contra a vida (SIPCV)<br>
    Excl. acidentes de trânsito</i>
</div>
"""
mapa1.get_root().html.add_child(folium.Element(legenda_html))

folium.LayerControl().add_to(mapa1)
mapa1.save('outputs/mapa_risco_sp.html')
print("[OK] Mapa 1 salvo: outputs/mapa_risco_sp.html")

# ── 3. MAPA 2 — COMPARAÇÃO PREVISÃO vs. OCORRÊNCIAS REAIS ────────────────────
print("\nGerando Mapa 2 — Comparação previsão vs. real...")
mapa2 = folium.Map(
    location=[LAT_CENTRO, LON_CENTRO],
    zoom_start=ZOOM_INICIAL,
    tiles=TILE
)

# Camada 1: Hotspots previstos (top 20% de densidade) — laranja/vermelho
limiar_20pct = df_grade['densidade'].quantile(0.80)
hotspots     = df_grade[df_grade['densidade'] >= limiar_20pct]
densidade_hs = hotspots['densidade'] / hotspots['densidade'].max()

HeatMap(
    data=list(zip(hotspots['lat'], hotspots['lon'], densidade_hs)),
    min_opacity=0.3,
    radius=12,
    blur=15,
    gradient={0.4: '#FFA500', 1.0: '#FF0000'},
    name='Hotspots previstos (top 20% KDE Jan-Mar 2026)'
).add_to(mapa2)

# Camada 2: Ocorrências reais de Abril — pontos azuis clicáveis
camada_real = folium.FeatureGroup(name='Ocorrências reais — Abril 2026')
for _, row in val_coords.iterrows():
    natureza = row.get('NATUREZA APURADA', 'N/D')
    local    = row.get('DESCR_TIPOLOCAL', 'N/D')
    periodo  = row.get('DESC_PERIODO', 'N/D')
    deleg    = row.get('NOME_DELEGACIA_CIRC', 'N/D')
    data_str = str(row.get('DATA_OCORRENCIA_BO', 'N/D'))[:10]

    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=6,
        color='#003DA5',
        fill=True,
        fill_color='#003DA5',
        fill_opacity=0.75,
        popup=folium.Popup(
            f"<b style='color:#003DA5'>{natureza}</b><br>"
            f"<b>Data:</b> {data_str}<br>"
            f"<b>Local:</b> {local}<br>"
            f"<b>Período:</b> {periodo}<br>"
            f"<b>Delegacia:</b> {deleg}",
            max_width=260
        ),
        tooltip=natureza
    ).add_to(camada_real)
camada_real.add_to(mapa2)

# Legenda do mapa 2
legenda2_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 1000;
    background: rgba(255, 255, 255, 0.95);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: Arial, sans-serif;
    font-size: 12px;
    line-height: 1.8;
">
    <b style="font-size:13px">Validação do Modelo KDE</b><br>
    <i style="font-size:10px;color:#666">Treino: Jan–Mar 2026 | Validação: Abr 2026</i><br><br>
    <span style="color:#FFA500">&#9632;</span> Hotspot previsto (top 20%)<br>
    <span style="color:#FF0000">&#9632;</span> Hotspot crítico previsto<br>
    <span style="color:#003DA5">&#11044;</span> Ocorrência real (Abr 2026)<br><br>
    <i style="font-size:10px;color:#888">Clique nos pontos azuis para detalhes</i>
</div>
"""
mapa2.get_root().html.add_child(folium.Element(legenda2_html))

folium.LayerControl().add_to(mapa2)
mapa2.save('outputs/mapa_validacao_sp.html')
print("[OK] Mapa 2 salvo: outputs/mapa_validacao_sp.html")

print("\n[OK] Etapa 5 concluída.")
print("  Abra os arquivos .html em qualquer navegador para visualizar os mapas.")
print("  outputs/mapa_risco_sp.html      -> Mapa principal de risco")
print("  outputs/mapa_validacao_sp.html  -> Comparação previsão vs. real")
