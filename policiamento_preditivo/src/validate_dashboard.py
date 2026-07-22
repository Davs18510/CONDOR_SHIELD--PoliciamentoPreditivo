with open('outputs/dashboard_preditivo.html', encoding='utf-8') as f:
    html = f.read()

checks = [
    ('leaflet.js', 'leaflet.js' in html),
    ('leaflet-heat.js', 'leaflet-heat.js' in html),
    ('CartoDB dark tile', 'dark_all' in html),
    ('const DATA', 'const DATA = ' in html),
    ('runPrediction', 'function runPrediction' in html),
    ('findZones', 'function findZones' in html),
    ('assignUnits', 'function assignUnits' in html),
    ('renderPatrolPanel', 'function renderPatrolPanel' in html),
    ('focusZone', 'function focusZone' in html),
    ('hslider presente', 'hslider' in html),
    ('uslider presente', 'uslider' in html),
    ('psel presente', 'psel' in html),
    ('BAIRROS SP', 'Se / Centro Historico' in html),
    ('BPMS SP', '1o BPM/M' in html),
    ('treino_pts embedded', 'treino_pts' in html),
    ('Heatmap gradient', '1E40AF' in html),
    ('RCOLS', 'RCOLS' in html),
    ('addEventListener', 'addEventListener' in html),
    ('Auto-run init', 'runPrediction()' in html),
    ('Tamanho > 100KB', len(html) > 100000),
]

ok = 0
for name, result in checks:
    tag = '[OK]  ' if result else '[ERRO]'
    if result:
        ok += 1
    print(tag, name)

print()
print('Resultado:', ok, '/', len(checks), 'verificacoes passaram')
print('Tamanho do HTML:', round(len(html)/1024, 1), 'KB')
