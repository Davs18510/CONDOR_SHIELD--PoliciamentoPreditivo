# =============================================================================
# ETAPA 2 — ENGENHARIA DE FEATURES
# =============================================================================
# Cria colunas derivadas para enriquecer a análise e o relatório.
# IMPORTANTE: O KDE usa apenas LATITUDE e LONGITUDE — estas features
# servem para exploração, estatísticas descritivas e interpretação.
# =============================================================================

import pandas as pd


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame processado e adiciona as colunas:
      - TURNO       : período do dia baseado na hora da ocorrência
      - FIM_DE_SEMANA: 1 se sábado ou domingo, 0 caso contrário
      - VIA_PUBLICA : 1 se o local for 'Via Pública'
      - CRIME_DOLOSO: 1 para Homicídio Doloso e Latrocínio
    """
    df = df.copy()

    # Garante que a data está no tipo correto após releitura do CSV
    df['DATA_OCORRENCIA_BO'] = pd.to_datetime(
        df['DATA_OCORRENCIA_BO'], errors='coerce'
    )

    # ── TURNO ─────────────────────────────────────────────────────────────────
    # Extrai apenas a hora (0–23) do campo de hora da ocorrência
    hora = pd.to_datetime(
        df['HORA_OCORRENCIA_BO'], format='%H:%M:%S', errors='coerce'
    ).dt.hour

    def classificar_turno(h):
        """Classifica a hora em um dos quatro turnos do dia."""
        if pd.isna(h):
            return 'desconhecido'
        h = int(h)
        if h < 6:   return 'madrugada'  # 00:00 – 05:59
        if h < 12:  return 'manha'      # 06:00 – 11:59
        if h < 18:  return 'tarde'      # 12:00 – 17:59
        return 'noite'                   # 18:00 – 23:59

    df['TURNO'] = hora.apply(classificar_turno)

    # ── FIM DE SEMANA ─────────────────────────────────────────────────────────
    # dayofweek: 0=segunda ... 4=sexta, 5=sábado, 6=domingo
    df['FIM_DE_SEMANA'] = (df['DATA_OCORRENCIA_BO'].dt.dayofweek >= 5).astype(int)

    # ── VIA PÚBLICA ───────────────────────────────────────────────────────────
    # Identifica ocorrências em espaço público aberto
    df['VIA_PUBLICA'] = df['DESCR_TIPOLOCAL'].isin(['Via Pública']).astype(int)

    # ── CRIME DOLOSO ──────────────────────────────────────────────────────────
    # Agrupa os crimes com intenção homicida direta (diferente de tentativa)
    df['CRIME_DOLOSO'] = df['NATUREZA APURADA'].isin(
        ['HOMICIDIO DOLOSO', 'LATROCINIO']
    ).astype(int)

    return df


if __name__ == '__main__':
    # Aplica a engenharia de features em todos os conjuntos e sobrescreve os CSVs
    conjuntos = [
        ('treino',    'data/processed/df_treino.csv'),
        ('validacao', 'data/processed/df_validacao.csv'),
        ('teste',     'data/processed/df_teste.csv'),
    ]

    for nome, caminho in conjuntos:
        df = pd.read_csv(caminho, sep=';', dtype=str)
        df = criar_features(df)
        df.to_csv(caminho, index=False, sep=';')

        # Estatísticas rápidas para conferência
        n_fds  = df['FIM_DE_SEMANA'].astype(int).sum() if 'FIM_DE_SEMANA' in df.columns else 0
        n_via  = df['VIA_PUBLICA'].astype(int).sum()   if 'VIA_PUBLICA'   in df.columns else 0
        turnos = df['TURNO'].value_counts().to_dict()  if 'TURNO'         in df.columns else {}

        print(f"[OK] Features criadas: {nome} ({len(df)} registros)")
        print(f"   Fins de semana : {n_fds}  |  Via pública: {n_via}")
        print(f"   Turnos         : {turnos}")

    print("\n[OK] Etapa 2 concluída.")
