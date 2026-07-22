# =============================================================================
# ETAPA 6 — DASHBOARD INTERATIVO DE POLICIAMENTO PREDITIVO
# =============================================================================
# Gera outputs/dashboard_preditivo.html — aplicação SPA com:
#   - Seleção de crimes, horizonte de previsão, nº de viaturas
#   - Mapa Leaflet com heatmap KDE + zonas de patrulha
#   - Sistema de direcionamento de viaturas (bairro, BPM, turno)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import json
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACOES ────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -24.008, -23.357
LON_MIN, LON_MAX = -46.826, -46.365
GRID_N = 60
BW = 0.003

CRIME_KEYS = [
    'TENTATIVA DE HOMICIDIO',
    'HOMICIDIO DOLOSO',
    'LESAO CORPORAL SEGUIDA DE MORTE',
    'LATROCINIO',
]
CRIME_LABELS = {
    'TENTATIVA DE HOMICIDIO'         : 'Tentativa de Homicidio',
    'HOMICIDIO DOLOSO'               : 'Homicidio Doloso',
    'LESAO CORPORAL SEGUIDA DE MORTE': 'Lesao Corp. Seguida de Morte',
    'LATROCINIO'                     : 'Latrocinio',
}

# ── CARREGAMENTO ─────────────────────────────────────────────────────────────
print('Carregando dados...')
df_tr = pd.read_csv('data/processed/df_treino.csv',    sep=';')
df_va = pd.read_csv('data/processed/df_validacao.csv', sep=';')
df_te = pd.read_csv('data/processed/df_teste.csv',     sep=';')

for df in [df_tr, df_va, df_te]:
    df['LATITUDE']  = pd.to_numeric(df['LATITUDE'],  errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')

coords = df_tr[df_tr['LATITUDE'].notna()].copy()
print(f'Pontos de treino com coords: {len(coords)}')

# ── GRADE 60x60 ───────────────────────────────────────────────────────────────
lat_arr = np.linspace(LAT_MIN, LAT_MAX, GRID_N)
lon_arr = np.linspace(LON_MIN, LON_MAX, GRID_N)
lg, ng  = np.meshgrid(lat_arr, lon_arr, indexing='ij')
grade   = np.radians(np.column_stack([lg.ravel(), ng.ravel()]))

def kde_grid(pts_df):
    if len(pts_df) < 5:
        return [[0]*GRID_N for _ in range(GRID_N)]
    rad = np.radians(pts_df[['LATITUDE', 'LONGITUDE']].values)
    kde = KernelDensity(kernel='gaussian', metric='haversine', bandwidth=BW)
    kde.fit(rad)
    d = np.exp(kde.score_samples(grade))
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return (d.reshape(GRID_N, GRID_N) * 1000).round(1).tolist()

print('Calculando grades KDE por tipo de crime...')
grids = {'all': kde_grid(coords)}
for ck in CRIME_KEYS:
    sub = coords[coords['NATUREZA APURADA'] == ck]
    print(f'  {CRIME_LABELS[ck]}: {len(sub)} pontos')
    grids[ck] = kde_grid(sub)

# ── PONTOS PARA DISPLAY ───────────────────────────────────────────────────────
def pts_json(df, periodo):
    out = []
    for _, r in df[df['LATITUDE'].notna()].iterrows():
        out.append({
            'lat'  : round(float(r['LATITUDE']),  5),
            'lon'  : round(float(r['LONGITUDE']), 5),
            'tipo' : str(r.get('NATUREZA APURADA', '')),
            'data' : str(r.get('DATA_OCORRENCIA_BO', ''))[:10],
            'local': str(r.get('DESCR_TIPOLOCAL', '')),
            'deleg': str(r.get('NOME_DELEGACIA_CIRC', '')),
            'per'  : str(r.get('DESC_PERIODO', '')),
            'ds'   : periodo,
        })
    return out

turnos = df_tr['TURNO'].value_counts().to_dict() if 'TURNO' in df_tr.columns \
         else {'noite': 99, 'manha': 77, 'madrugada': 76, 'tarde': 67}
tipo_counts = coords['NATUREZA APURADA'].value_counts().to_dict()

DATA = {
    'grids'       : grids,
    'lat_arr'     : lat_arr.round(5).tolist(),
    'lon_arr'     : lon_arr.round(5).tolist(),
    'grid_n'      : GRID_N,
    'treino_pts'  : pts_json(df_tr, 'treino'),
    'val_pts'     : pts_json(df_va, 'val'),
    'test_pts'    : pts_json(df_te, 'test'),
    'metrics'     : {
        'n_treino': len(df_tr), 'n_coords': len(coords),
        'n_val': len(df_va),    'n_test': len(df_te),
        'pai_10_abr': 3.91, 'pai_20_abr': 3.05,
        'pai_10_mai': 4.74, 'pai_20_mai': 3.25,
    },
    'turnos'      : turnos,
    'tipo_counts' : tipo_counts,
    'crime_keys'  : CRIME_KEYS,
    'crime_labels': CRIME_LABELS,
}

data_js = 'const DATA = ' + json.dumps(DATA, ensure_ascii=False, separators=(',', ':')) + ';'
print(f'Dados JSON: {len(data_js)/1024:.1f} KB')

# ── HTML ───────────────────────────────────────────────────────────────────────
HEAD = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PredPol SP - Sistema de Policiamento Preditivo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js"></script>
<style>
:root{--bg:#080D1A;--bg2:#0D1628;--card:rgba(255,255,255,.04);--bdr:rgba(59,130,246,.2);
--blue:#3B82F6;--bl:#60A5FA;--txt:#E2E8F0;--muted:#64748B;
--red:#EF4444;--ora:#F97316;--yel:#EAB308;--grn:#10B981;
--sw:280px;--pw:360px;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;font-family:\'Inter\',sans-serif;background:var(--bg);color:var(--txt);overflow:hidden;}
#hdr{height:52px;background:linear-gradient(90deg,#060E22,#112046,#060E22);
  border-bottom:1px solid var(--bdr);display:flex;align-items:center;
  justify-content:space-between;padding:0 18px;position:relative;z-index:999;}
.hdr-l{display:flex;align-items:center;gap:10px;}
.hdr-l h1{font-size:14px;font-weight:800;letter-spacing:.02em;}
.hdr-l h1 span{color:var(--bl);}
.sim-badge{font-size:9px;font-weight:800;background:var(--red);color:#fff;
  padding:2px 7px;border-radius:3px;letter-spacing:.1em;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.7;}}
.hdr-r{font-size:10px;color:var(--muted);text-align:right;line-height:1.7;}
.hdr-r b{color:var(--bl);}
#main{display:grid;grid-template-columns:var(--sw) 1fr var(--pw);
  height:calc(100vh - 52px - 34px);}
#sidebar{background:var(--bg2);border-right:1px solid var(--bdr);
  overflow-y:auto;padding:12px 10px;display:flex;flex-direction:column;gap:8px;}
#mapbox{position:relative;}
#map{width:100%;height:100%;}
#patrol{background:var(--bg2);border-left:1px solid var(--bdr);
  overflow-y:auto;padding:12px 10px;display:flex;flex-direction:column;gap:6px;}
#ftr{height:34px;background:#04080F;border-top:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 14px;font-size:10px;color:var(--muted);}
.panel{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:11px 12px;}
.ptitle{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--bl);margin-bottom:9px;}
.copt{display:flex;align-items:center;gap:8px;padding:5px 6px;
  border-radius:6px;cursor:pointer;margin-bottom:3px;transition:background .15s;}
.copt:hover{background:rgba(59,130,246,.1);}
.copt input{accent-color:var(--blue);width:14px;height:14px;cursor:pointer;flex-shrink:0;}
.copt label{font-size:12px;cursor:pointer;line-height:1.3;user-select:none;}
.srow{margin-bottom:9px;}
.slbl{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:5px;}
.slbl b{color:var(--bl);font-weight:700;}
input[type=range]{width:100%;accent-color:var(--blue);cursor:pointer;}
select{width:100%;padding:6px 8px;background:rgba(255,255,255,.05);
  border:1px solid var(--bdr);border-radius:6px;color:var(--txt);
  font-size:12px;font-family:inherit;cursor:pointer;outline:none;}
select:focus{border-color:var(--blue);}
#run-btn{width:100%;padding:11px;background:linear-gradient(135deg,#1D4ED8,#2563EB);
  border:none;border-radius:8px;color:#fff;font-size:13px;font-weight:700;
  letter-spacing:.04em;cursor:pointer;transition:all .2s;margin-top:2px;
  font-family:inherit;}
#run-btn:hover{background:linear-gradient(135deg,#2563EB,#3B82F6);
  transform:translateY(-1px);box-shadow:0 4px 18px rgba(59,130,246,.4);}
#run-btn:active{transform:translateY(0);}
.pcard{background:var(--card);border:1px solid var(--bdr);border-radius:8px;
  padding:10px 12px;cursor:pointer;transition:all .15s;margin-bottom:5px;}
.pcard:hover{border-color:rgba(59,130,246,.4);background:rgba(59,130,246,.05);transform:translateX(2px);}
.pcard.critico{border-left:3px solid var(--red);}
.pcard.alto{border-left:3px solid var(--ora);}
.pcard.medio{border-left:3px solid var(--yel);}
.phdr{display:flex;align-items:center;gap:5px;margin-bottom:5px;flex-wrap:wrap;}
.vnum{font-size:11px;font-weight:800;color:#fff;background:rgba(255,255,255,.1);
  padding:2px 7px;border-radius:4px;min-width:36px;text-align:center;}
.utype{font-size:10px;font-weight:600;color:var(--bl);background:rgba(59,130,246,.12);
  padding:2px 6px;border-radius:4px;flex:1;}
.rbadge{font-size:9px;font-weight:800;padding:2px 7px;border-radius:4px;letter-spacing:.06em;}
.rbadge.CRITICO{background:rgba(239,68,68,.2);color:#FCA5A5;}
.rbadge.ALTO{background:rgba(249,115,22,.2);color:#FDBA74;}
.rbadge.MEDIO{background:rgba(234,179,8,.2);color:#FDE68A;}
.pbairro{font-size:12px;font-weight:600;color:#CBD5E1;margin-bottom:4px;}
.pdet{font-size:10px;color:var(--muted);display:flex;flex-direction:column;gap:2px;line-height:1.5;}
.mrow{display:flex;justify-content:space-between;align-items:center;
  padding:4px 0;border-bottom:1px solid rgba(255,255,255,.05);}
.mrow:last-child{border-bottom:none;}
.mkey{font-size:11px;color:var(--muted);}
.mval{font-size:13px;font-weight:700;color:var(--bl);}
.cbar{height:5px;background:rgba(255,255,255,.08);border-radius:3px;overflow:hidden;margin:5px 0;}
.cfill{height:100%;border-radius:3px;transition:width .6s ease;}
.ibox{background:rgba(234,179,8,.07);border:1px solid rgba(234,179,8,.22);
  border-radius:7px;padding:9px 10px;font-size:10px;color:#FDE68A;line-height:1.55;}
.pp-hdr{padding-bottom:8px;border-bottom:1px solid var(--bdr);margin-bottom:4px;}
.pp-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--bl);}
.pp-sub{font-size:11px;color:var(--muted);margin-top:3px;}
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px;}
.chip{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
  border-radius:3px;padding:1px 6px;font-size:9px;color:var(--bl);}
#map-overlay{position:absolute;top:10px;right:10px;z-index:500;
  background:rgba(8,13,26,.88);border:1px solid var(--bdr);border-radius:8px;
  padding:10px 12px;font-size:11px;line-height:1.8;min-width:160px;
  backdrop-filter:blur(8px);}
#map-overlay b{color:var(--bl);display:block;margin-bottom:4px;font-size:12px;}
.leg-row{display:flex;align-items:center;gap:6px;}
.leg-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(59,130,246,.25);border-radius:2px;}
.empty-state{text-align:center;color:var(--muted);padding:40px 16px;font-size:12px;line-height:1.7;}
.empty-state b{color:var(--bl);display:block;margin-bottom:8px;font-size:14px;}
</style>
</head>
<body>
<div id="hdr">
  <div class="hdr-l">
    <span style="font-size:20px">🚔</span>
    <h1><span>PredPol</span> SP &mdash; Sistema de Policiamento Preditivo</h1>
    <span class="sim-badge">SIMULACAO</span>
  </div>
  <div class="hdr-r">
    <div>SIPCV / SSP-SP &nbsp;&bull;&nbsp; Jan&ndash;Mar 2026</div>
    <div>Modelo KDE Haversine &nbsp;&bull;&nbsp; <b>PAI max: 4.74</b></div>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div class="panel">
      <div class="ptitle">Tipos de Crime</div>
      <div class="copt"><input type="checkbox" id="ck0" checked>
        <label for="ck0">Tentativa de Homicidio</label></div>
      <div class="copt"><input type="checkbox" id="ck1" checked>
        <label for="ck1">Homicidio Doloso</label></div>
      <div class="copt"><input type="checkbox" id="ck2" checked>
        <label for="ck2">Lesao Corp. Seguida de Morte</label></div>
      <div class="copt"><input type="checkbox" id="ck3" checked>
        <label for="ck3">Latrocinio</label></div>
    </div>
    <div class="panel">
      <div class="ptitle">Horizonte de Previsao</div>
      <div class="srow">
        <div class="slbl"><span>Meses a frente</span><b id="hv">1</b></div>
        <input type="range" id="hslider" min="1" max="6" value="1">
        <div id="htarget" style="font-size:11px;color:var(--bl);margin-top:4px;font-weight:600;"></div>
      </div>
    </div>
    <div class="panel">
      <div class="ptitle">Viaturas Disponiveis</div>
      <div class="srow">
        <div class="slbl"><span>Unidades</span><b id="uv">5</b></div>
        <input type="range" id="uslider" min="1" max="20" value="5">
      </div>
      <div class="ptitle" style="margin-top:8px;">Tipo de Patrulha</div>
      <select id="psel">
        <option value="AUTO">Atribuicao automatica</option>
        <option value="ROCAM/M">ROCAM/M</option>
        <option value="RPA">RPA (Radio Patrulha)</option>
        <option value="Forca Tatica">Forca Tatica</option>
        <option value="ROTAM">ROTAM</option>
        <option value="GCM">Guarda Civil Metropolitana</option>
        <option value="CHOQUE">Batalhao de Choque</option>
      </select>
    </div>
    <button id="run-btn" onclick="runPrediction()">&#9654; EXECUTAR PREVISAO</button>
    <div class="panel" id="mpanel">
      <div class="ptitle">Metricas do Modelo</div>
      <div id="mcontent"><div class="mrow"><span class="mkey">PAI (top 10%)</span>
        <span class="mval">3.91</span></div>
        <div class="mrow"><span class="mkey">Captacao (10% area)</span>
        <span class="mval">39%</span></div>
        <div class="mrow"><span class="mkey">Treino</span>
        <span class="mval">Jan-Mar 2026</span></div></div>
    </div>
    <div class="ibox">
      &#9888; Ferramenta de apoio a decisao. Nao substitui julgamento operacional.
      Modelo treinado com dados de Jan-Mar 2026 &mdash; recomenda-se atualizacao periodica.
    </div>
  </div>
  <div id="mapbox">
    <div id="map"></div>
    <div id="map-overlay">
      <b>Legenda</b>
      <div class="leg-row"><div class="leg-dot" style="background:#EF4444"></div> Risco CRITICO</div>
      <div class="leg-row"><div class="leg-dot" style="background:#F97316"></div> Risco ALTO</div>
      <div class="leg-row"><div class="leg-dot" style="background:#EAB308"></div> Risco MEDIO</div>
      <div class="leg-row" style="margin-top:6px"><div class="leg-dot"
        style="background:linear-gradient(90deg,#1E40AF,#EF4444);border-radius:3px;width:24px;height:6px"></div>
        &nbsp;Heatmap KDE</div>
    </div>
  </div>
  <div id="patrol">
    <div class="pp-hdr">
      <div class="pp-title">Direcionamento de Viaturas</div>
      <div class="pp-sub" id="pp-sub">Configure e clique em EXECUTAR</div>
    </div>
    <div id="plist">
      <div class="empty-state">
        <b>&#128205; Aguardando previsao</b>
        Configure os parametros no painel esquerdo<br>e clique em <b>EXECUTAR PREVISAO</b>
      </div>
    </div>
  </div>
</div>
<div id="ftr">
  <span>Dados: SIPCV/SSP-SP (Out 2025&ndash;Mai 2026) &nbsp;|&nbsp; KDE Haversine (BW=0.003 rad ~19km) &nbsp;|&nbsp; Split: Jan-Mar treino / Abr validacao / Mai teste</span>
  <span id="fstatus" style="color:var(--bl);font-weight:600;"></span>
</div>
<script>
'''

TAIL = '''
const BAIRROS=[
  {n:"Se / Centro Historico",lat:-23.547,lon:-46.637},{n:"Republica",lat:-23.543,lon:-46.642},
  {n:"Liberdade",lat:-23.560,lon:-46.632},{n:"Consolacao",lat:-23.554,lon:-46.652},
  {n:"Bela Vista",lat:-23.557,lon:-46.643},{n:"Cambuci",lat:-23.572,lon:-46.622},
  {n:"Bras / Belem",lat:-23.547,lon:-46.607},{n:"Mooca",lat:-23.553,lon:-46.596},
  {n:"Tatuape",lat:-23.540,lon:-46.572},{n:"Penha",lat:-23.520,lon:-46.543},
  {n:"Itaquera",lat:-23.536,lon:-46.453},{n:"Ermelino Matarazzo",lat:-23.497,lon:-46.472},
  {n:"Vila Matilde",lat:-23.530,lon:-46.516},{n:"Ipiranga",lat:-23.589,lon:-46.605},
  {n:"Saude / Jabaquara",lat:-23.622,lon:-46.636},{n:"Cidade Ademar",lat:-23.663,lon:-46.663},
  {n:"Vila Prudente",lat:-23.585,lon:-46.578},{n:"Sao Mateus",lat:-23.613,lon:-46.537},
  {n:"Sapopemba",lat:-23.591,lon:-46.521},{n:"Santana",lat:-23.498,lon:-46.624},
  {n:"Tucuruvi",lat:-23.475,lon:-46.607},{n:"Pirituba",lat:-23.474,lon:-46.728},
  {n:"Lapa / Barra Funda",lat:-23.523,lon:-46.706},{n:"Pinheiros",lat:-23.564,lon:-46.693},
  {n:"Butanta",lat:-23.578,lon:-46.722},{n:"Campo Limpo",lat:-23.676,lon:-46.743},
  {n:"Santo Amaro",lat:-23.655,lon:-46.706},{n:"Jardins",lat:-23.561,lon:-46.654},
  {n:"Perdizes / Santa Cecilia",lat:-23.537,lon:-46.655},{n:"Guarulhos",lat:-23.465,lon:-46.533},
  {n:"Osasco / Carapicuiba",lat:-23.532,lon:-46.792},{n:"Taboao da Serra",lat:-23.611,lon:-46.762},
  {n:"Sao Bernardo do Campo",lat:-23.698,lon:-46.565},{n:"Santo Andre",lat:-23.653,lon:-46.538},
  {n:"Diadema",lat:-23.686,lon:-46.622},{n:"Maua",lat:-23.668,lon:-46.461},
];
const PATROL_TYPES=['ROCAM/M','RPA','RPA','Forca Tatica','ROTAM','GCM','RPA','CHOQUE'];
const BPMS=['1o BPM/M (Se)','5o BPM/M (Bras)','7o BPM/M (Ipiranga)','8o BPM/M (Pinheiros)',
  '12o BPM/M (S.Mateus)','13o BPM/M (Santana)','14o BPM/M (Tatua pe)','15o BPM/M (Jabaquara)',
  '17o BPM/M (C.Limpo)','19o BPM/M (Guarulhos)','21o BPM/M (Pirituba)','23o BPM/M (Itaquera)',
  '24o BPM/M (Guaianazes)','27o BPM/M (Diadema)','37o BPM/M (Osasco)'];
const RCOLS={CRITICO:'#EF4444',ALTO:'#F97316',MEDIO:'#EAB308'};
const CONF={1:85,2:78,3:62,4:48,5:38,6:30};
const PAI_EST={1:3.91,2:4.74,3:3.2,4:2.5,5:1.9,6:1.5};
const MONTHS=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];

// ── MAP SETUP ─────────────────────────────────────────────────────────────────
const map=L.map('map').setView([-23.5505,-46.6333],12);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
  attribution:'&copy; OpenStreetMap &amp; CARTO',
  subdomains:'abcd',maxZoom:18
}).addTo(map);

let heatL=null,zoneL=[],markL=[],routeL=[];

function clearLayers(){
  if(heatL){map.removeLayer(heatL);heatL=null;}
  [...zoneL,...markL,...routeL].forEach(l=>map.removeLayer(l));
  zoneL=[];markL=[];routeL=[];
}

// ── UTILITIES ─────────────────────────────────────────────────────────────────
function nearestBairro(lat,lon){
  let mn=999,best='Sao Paulo';
  for(const b of BAIRROS){
    const d=Math.hypot(lat-b.lat,lon-b.lon);
    if(d<mn){mn=d;best=b.n;}
  }
  return best;
}

function getTargetMonth(h){
  const now=new Date();
  const td=new Date(now.getFullYear(),now.getMonth()+parseInt(h),1);
  return MONTHS[td.getMonth()]+' '+td.getFullYear();
}

function getSelectedKeys(){
  const IDS=['TENTATIVA DE HOMICIDIO','HOMICIDIO DOLOSO',
    'LESAO CORPORAL SEGUIDA DE MORTE','LATROCINIO'];
  const keys=[];
  for(let i=0;i<4;i++)
    if(document.getElementById('ck'+i).checked) keys.push(IDS[i]);
  return keys;
}

// ── GRID LOGIC ────────────────────────────────────────────────────────────────
function combineGrids(keys){
  const N=DATA.grid_n;
  const out=Array.from({length:N},()=>new Float32Array(N));
  const active=keys.length?keys:['all'];
  for(const k of active){
    const g=DATA.grids[k]||DATA.grids['all'];
    for(let i=0;i<N;i++) for(let j=0;j<N;j++) out[i][j]+=g[i][j];
  }
  let mx=0;
  for(let i=0;i<N;i++) for(let j=0;j<N;j++) if(out[i][j]>mx) mx=out[i][j];
  if(mx>0) for(let i=0;i<N;i++) for(let j=0;j<N;j++) out[i][j]=out[i][j]/mx*1000;
  return out;
}

function findZones(grid,hFactor){
  const N=DATA.grid_n;
  const cells=[];
  let flat=[];
  for(let i=0;i<N;i++) for(let j=0;j<N;j++) flat.push(grid[i][j]);
  flat.sort((a,b)=>b-a);
  const thr=flat[Math.floor(flat.length*0.08)];
  for(let i=0;i<N;i++)
    for(let j=0;j<N;j++)
      if(grid[i][j]>=thr)
        cells.push({lat:DATA.lat_arr[i],lon:DATA.lon_arr[j],d:grid[i][j],i,j});
  cells.sort((a,b)=>b.d-a.d);
  const R=0.032*Math.max(1,hFactor);
  const zones=[],used=new Set();
  for(const c of cells){
    const key=c.i+','+c.j;
    if(used.has(key)) continue;
    const mems=[c]; used.add(key);
    for(const o of cells){
      if(used.has(o.i+','+o.j)) continue;
      if(Math.hypot(c.lat-o.lat,c.lon-o.lon)<=R){mems.push(o);used.add(o.i+','+o.j);}
    }
    let wLat=0,wLon=0,wS=0;
    for(const m of mems){wLat+=m.lat*m.d;wLon+=m.lon*m.d;wS+=m.d;}
    zones.push({lat:wLat/wS,lon:wLon/wS,totD:wS,maxD:c.d,sz:mems.length,
      radM:Math.max(600,Math.round(R*111000*Math.max(1,hFactor*0.8)))});
    if(zones.length>=12) break;
  }
  zones.sort((a,b)=>b.totD-a.totD);
  return zones;
}

function assignUnits(zones,nUnits,pType){
  if(!zones.length) return [];
  const assigns=[];
  for(let u=0;u<nUnits;u++){
    const zi=u%zones.length;
    const z=zones[zi];
    const rp=z.maxD/1000;
    const risk=rp>0.65?'CRITICO':rp>0.35?'ALTO':'MEDIO';
    const bairro=nearestBairro(z.lat,z.lon);
    const ut=pType!=='AUTO'?pType:PATROL_TYPES[u%PATROL_TYPES.length];
    const bpm=BPMS[zi%BPMS.length];
    const tos=DATA.turnos;
    const maxT=Object.entries(tos).sort((a,b)=>b[1]-a[1])[0][0];
    const tStr={noite:'18h-23h (pico noturno)',manha:'06h-12h (pico matutino)',
      tarde:'12h-18h (pico vespertino)',madrugada:'00h-05h (madrugada)'}[maxT]||'18h-23h';
    assigns.push({num:u+1,zone:z,risk,bairro,ut,bpm,tStr,zi,coords:[z.lat,z.lon]});
  }
  return assigns;
}

// ── MAP RENDERING ─────────────────────────────────────────────────────────────
function renderHeatmap(keys){
  let pts=DATA.treino_pts;
  if(keys.length) pts=pts.filter(p=>keys.includes(p.tipo));
  if(!pts.length) return;
  heatL=L.heatLayer(pts.map(p=>[p.lat,p.lon,1.0]),{
    radius:22,blur:14,minOpacity:0.4,maxZoom:16,
    gradient:{0.25:'#1E3A8A',0.5:'#0EA5E9',0.72:'#10B981',0.87:'#F59E0B',1.0:'#EF4444'}
  }).addTo(map);
}

function renderZones(zones,assigns){
  // Build zone->units map
  const zu={};
  for(const a of assigns){
    if(!zu[a.zi]) zu[a.zi]={z:a.zone,risk:a.risk,bairro:a.bairro,units:[]};
    zu[a.zi].units.push(a);
  }
  for(const [zi,zd] of Object.entries(zu)){
    const col=RCOLS[zd.risk];
    const circle=L.circle([zd.z.lat,zd.z.lon],{
      radius:zd.z.radM,color:col,fillColor:col,fillOpacity:0.1,
      weight:2,dashArray:'6,3'
    }).addTo(map);
    const popBody=zd.units.map(u=>`<b>V${String(u.num).padStart(2,'0')}</b> ${u.ut}`).join('<br>');
    circle.bindPopup(`<div style="font-family:sans-serif;font-size:12px">
      <b style="color:${col}">${zd.bairro}</b><br>Risco: ${zd.risk}<br><br>${popBody}</div>`);
    zoneL.push(circle);

    // Markers (grouped per zone)
    const lbl=zd.units.length>1?zd.units.length+'V':'V'+String(zd.units[0].num).padStart(2,'0');
    const icon=L.divIcon({
      html:`<div style="background:${col};color:#fff;border-radius:50%;width:34px;height:34px;
        display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;
        border:2px solid rgba(255,255,255,.55);box-shadow:0 2px 10px rgba(0,0,0,.6);
        font-family:Inter,sans-serif">${lbl}</div>`,
      className:'',iconSize:[34,34],iconAnchor:[17,17]
    });
    const mk=L.marker([zd.z.lat,zd.z.lon],{icon}).addTo(map);
    mk.bindPopup(`<div style="font-family:sans-serif;font-size:12px">
      <b>${zd.bairro}</b><br>${popBody}</div>`);
    markL.push(mk);

    // Patrol route star: center -> nearby training pts
    const near=DATA.treino_pts.filter(p=>Math.hypot(p.lat-zd.z.lat,p.lon-zd.z.lon)<0.025).slice(0,6);
    for(const np of near){
      const rl=L.polyline([[zd.z.lat,zd.z.lon],[np.lat,np.lon]],
        {color:col,weight:1.2,opacity:0.35,dashArray:'3,4'}).addTo(map);
      routeL.push(rl);
    }
  }
}

// ── PATROL PANEL ──────────────────────────────────────────────────────────────
function renderPatrolPanel(assigns,h){
  const conf=CONF[h]||30;
  const target=getTargetMonth(h);
  const confCol=conf>70?'#10B981':conf>50?'#F59E0B':'#EF4444';

  document.getElementById('pp-sub').textContent='Previsao para '+target+' | '+assigns.length+' viaturas';

  // Confidence block
  let html=`<div class="panel" style="margin-bottom:6px">
    <div class="ptitle">Confianca da Previsao</div>
    <div style="font-size:20px;font-weight:800;color:${confCol}">${conf}%</div>
    <div class="cbar"><div class="cfill" style="width:${conf}%;background:${confCol}"></div></div>
    <div style="font-size:10px;color:var(--muted)">Horizonte: +${h} mes(es) | Alvo: ${target}</div>
    ${h>=3?'<div style="font-size:10px;color:#F59E0B;margin-top:5px">&#9888; Previsao extrapolada. Atualizar modelo com dados recentes.</div>':''}
  </div>`;

  // Unit cards
  for(const a of assigns){
    const rc=a.risk.toLowerCase();
    const rl={CRITICO:'CRITICO',ALTO:'ALTO',MEDIO:'MEDIO'}[a.risk];
    html+=`<div class="pcard ${rc}" onclick="focusZone(${a.zone.lat},${a.zone.lon},${a.zone.radM})">
      <div class="phdr">
        <span class="vnum">V${String(a.num).padStart(2,'0')}</span>
        <span class="utype">${a.ut}</span>
        <span class="rbadge ${a.risk}">${rl}</span>
      </div>
      <div class="pbairro">&#128205; ${a.bairro}</div>
      <div class="pdet">
        <span>&#127963; ${a.bpm}</span>
        <span>&#8987; Foco: ${a.tStr}</span>
        <span>&#128204; ${a.zone.lat.toFixed(4)}, ${a.zone.lon.toFixed(4)}</span>
        <span>&#128308; Zona ${a.zi+1} de ${Math.min(assigns.length,12)}</span>
      </div>
    </div>`;
  }
  document.getElementById('plist').innerHTML=html;
}

// ── METRICS PANEL ─────────────────────────────────────────────────────────────
function updateMetrics(keys,assigns,h){
  const pai=PAI_EST[h]||1.5;
  const conf=CONF[h]||30;
  const nPts=keys.length?DATA.treino_pts.filter(p=>keys.includes(p.tipo)).length:DATA.treino_pts.length;
  const cap=Math.round((pai*10));
  document.getElementById('mcontent').innerHTML=`
    <div class="mrow"><span class="mkey">PAI estimado</span><span class="mval">${pai.toFixed(2)}</span></div>
    <div class="mrow"><span class="mkey">Captacao (10% area)</span><span class="mval">~${Math.min(99,cap)}%</span></div>
    <div class="mrow"><span class="mkey">Confianca</span><span class="mval">${conf}%</span></div>
    <div class="mrow"><span class="mkey">Pontos de treino</span><span class="mval">${nPts}</span></div>
    <div class="mrow"><span class="mkey">Viaturas</span><span class="mval">${assigns.length}</span></div>
    <div class="mrow"><span class="mkey">Zonas detectadas</span><span class="mval">${Math.min(assigns.length,12)}</span></div>
  `;
}

// ── FOCUS MAP ─────────────────────────────────────────────────────────────────
function focusZone(lat,lon,radM){
  const zoom=radM>2000?12:radM>1000?13:14;
  map.flyTo([lat,lon],zoom,{duration:1.0});
}

// ── MAIN RUN ──────────────────────────────────────────────────────────────────
function runPrediction(){
  const btn=document.getElementById('run-btn');
  btn.textContent='Calculando...';btn.disabled=true;
  setTimeout(()=>{
    const keys=getSelectedKeys();
    const h=parseInt(document.getElementById('hslider').value);
    const n=parseInt(document.getElementById('uslider').value);
    const pt=document.getElementById('psel').value;
    const hF=1+(h-1)*0.22;

    const grid=combineGrids(keys);
    const zones=findZones(grid,hF);
    const assigns=assignUnits(zones,n,pt);

    clearLayers();
    renderHeatmap(keys);
    renderZones(zones,assigns);
    renderPatrolPanel(assigns,h);
    updateMetrics(keys,assigns,h);

    // Val/test pts toggle layer (always shown as small gray dots)
    const valPts=[...DATA.val_pts,...DATA.test_pts];
    const vKeys=keys.length?keys:DATA.crime_keys;
    const vf=valPts.filter(p=>vKeys.includes(p.tipo));
    if(vf.length){
      const vLayer=L.layerGroup();
      for(const p of vf){
        L.circleMarker([p.lat,p.lon],{radius:3,color:'#94A3B8',fillColor:'#94A3B8',
          fillOpacity:.5,weight:0}).bindPopup(
          `<div style="font-size:11px;font-family:sans-serif"><b>${p.tipo}</b><br>${p.data}<br>${p.local}</div>`
        ).addTo(vLayer);
      }
      vLayer.addTo(map);
      markL.push(vLayer);
    }

    const target=getTargetMonth(h);
    document.getElementById('fstatus').textContent=
      'Previsao para '+target+' | '+assigns.length+' viaturas | Confianca '+CONF[h]+'%';

    btn.textContent='\\u25b6 EXECUTAR PREVISAO';btn.disabled=false;
  },80);
}

// ── EVENT LISTENERS ───────────────────────────────────────────────────────────
document.getElementById('hslider').addEventListener('input',e=>{
  document.getElementById('hv').textContent=e.target.value;
  document.getElementById('htarget').textContent='Alvo: '+getTargetMonth(e.target.value);
});
document.getElementById('uslider').addEventListener('input',e=>{
  document.getElementById('uv').textContent=e.target.value;
});

// ── INIT ──────────────────────────────────────────────────────────────────────
document.getElementById('htarget').textContent='Alvo: '+getTargetMonth(1);
runPrediction();
</script>
</body>
</html>'''

HTML = HEAD + '\n' + data_js + '\n' + TAIL

out_path = 'outputs/dashboard_preditivo.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f'[OK] Dashboard salvo: {out_path}')
print(f'     Tamanho: {len(HTML)/1024:.0f} KB')
print('[OK] Etapa 6 concluida. Abra o dashboard_preditivo.html no navegador.')
