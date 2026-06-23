"""
monitoring_utils.py
====================
Utilidades de monitoreo para el pipeline MLOps.
Incluye:
  - Cálculo de Data Drift (PSI / KL) en cada etapa del pipeline
  - Cálculo de SHAP values y variación mes a mes
  - Paralelismo CPU (multiprocessing + concurrent.futures)
  - Generación de tablero de control HTML interactivo (E2E dashboard)
"""

import glob
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
PSI_LOW    = 0.10   # sin drift relevante
PSI_MEDIUM = 0.20   # drift moderado
# PSI > 0.20 → drift alto

N_WORKERS = max(2, cpu_count() - 1)   # mínimo 2 workers (requisito del curso)

# ─────────────────────────────────────────────────────────────────────────────
# LEER VARIABLES CRUDAS
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_files(DIR_RAWDATA):
    """Lee y concatena todos los CSV de un directorio."""
    csv_files = glob.glob(f'{DIR_RAWDATA}/*.csv')
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print("Successfully unified dataframes.")
    else:
        df = pd.DataFrame()
        print("No CSV files found or read errors occurred.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE MÉTRICAS DE DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def calculate_drift(dd_metric, df_ref, df_actual, quantils):
    """
    Calcula PSI o KL-divergence entre una distribución de referencia y actual.

    Returns
    -------
    tuple: (dd_value, dd_values, ref_counts, actual_counts, breakpoints)
    """
    breakpoints = np.percentile(df_ref, np.linspace(0, 100, quantils + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    ref_counts    = np.histogram(df_ref,    bins=breakpoints)[0] / len(df_ref)
    actual_counts = np.histogram(df_actual, bins=breakpoints)[0] / len(df_actual)

    ref_counts    = np.where(ref_counts    == 0, 1e-6, ref_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    if dd_metric == 'PSI':
        dd_values = (ref_counts - actual_counts) * np.log(ref_counts / actual_counts)
    elif dd_metric == 'KL':
        dd_values = ref_counts * np.log(ref_counts / actual_counts)
    else:
        raise ValueError(f"Métrica no soportada: {dd_metric}. Use 'PSI' o 'KL'.")

    dd_value = np.sum(dd_values)
    return (dd_value, dd_values, ref_counts, actual_counts, breakpoints)


# ── Worker helper (debe estar en el módulo para que pickle lo serialice) ─────
def _drift_worker(args):
    """Función worker para paralelismo en data_drift."""
    col, dd_metric, ref_vals, actual_vals, quantils = args
    try:
        result = calculate_drift(dd_metric, ref_vals, actual_vals, quantils)
        return col, result[0], None
    except Exception as e:
        return col, np.nan, str(e)


def data_drift(dd_metric, df_actual, df_ref, quantils, n_workers=N_WORKERS):
    """
    Calcula drift (PSI / KL) para todas las columnas numéricas en paralelo.

    Parameters
    ----------
    dd_metric  : 'PSI' o 'KL'
    df_actual  : DataFrame del período actual
    df_ref     : DataFrame de referencia
    quantils   : número de buckets / cuantiles
    n_workers  : número de procesos paralelos (mínimo 2)

    Returns
    -------
    pd.DataFrame con columnas ['feature', 'metric_value', 'alert_level']
    """
    n_workers = max(2, n_workers)

    tasks = []
    for col in df_actual.columns:
        if (pd.api.types.is_numeric_dtype(df_actual[col]) and
                col in df_ref.columns and
                pd.api.types.is_numeric_dtype(df_ref[col])):
            ref_vals    = df_ref[col].dropna().values
            actual_vals = df_actual[col].dropna().values
            if len(ref_vals) > 0 and len(actual_vals) > 0:
                tasks.append((col, dd_metric, ref_vals, actual_vals, quantils))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_drift_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            col, metric_val, err = future.result()
            if err:
                print(f"  [WARN] {col}: {err}")
            results.append({'feature': col, 'metric_value': metric_val})

    df_result = pd.DataFrame(results).set_index('feature')

    # Nivel de alerta
    def _alert(v):
        if pd.isna(v):
            return 'ERROR'
        if v < PSI_LOW:
            return 'OK'
        if v < PSI_MEDIUM:
            return 'WARN'
        return 'ALERT'

    df_result['alert_level'] = df_result['metric_value'].apply(_alert)
    return df_result.sort_values('metric_value', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT POR ETAPA DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_drift(stage_name, df_ref, df_actual,
                         dd_metric='PSI', quantils=10, n_workers=N_WORKERS):
    """
    Calcula el drift para una etapa específica del pipeline.

    Parameters
    ----------
    stage_name : str  – nombre de la etapa ('raw', 'preprocessed', 'score', 'postprocessed')
    df_ref     : DataFrame de referencia
    df_actual  : DataFrame del período actual
    dd_metric  : 'PSI' o 'KL'
    quantils   : número de buckets
    n_workers  : workers paralelos

    Returns
    -------
    dict con keys: stage, metric, timestamp, drift_df (DataFrame)
    """
    print(f"  [Monitoring] Calculando drift en etapa '{stage_name}' con {n_workers} workers...")
    drift_df = data_drift(dd_metric, df_actual, df_ref, quantils, n_workers=n_workers)
    summary = {
        'stage'          : stage_name,
        'metric'         : dd_metric,
        'timestamp'      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_features'     : len(drift_df),
        'n_alerts'       : int((drift_df['alert_level'] == 'ALERT').sum()),
        'n_warns'        : int((drift_df['alert_level'] == 'WARN').sum()),
        'mean_drift'     : float(drift_df['metric_value'].mean()),
        'max_drift'      : float(drift_df['metric_value'].max()),
        'drift_df'       : drift_df,
    }
    print(f"    → Features: {summary['n_features']} | "
          f"ALERT: {summary['n_alerts']} | "
          f"WARN: {summary['n_warns']} | "
          f"Mean PSI: {summary['mean_drift']:.4f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SHAP VALUES
# ─────────────────────────────────────────────────────────────────────────────

def _shap_worker(args):
    """Worker para calcular SHAP en un subset de filas (paralelismo CPU)."""
    model_path, df_chunk_json, ml_name = args
    try:
        import shap, pickle, pandas as pd, numpy as np
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        df_chunk = pd.read_json(df_chunk_json, orient='split')

        if ml_name == 'xgb':
            explainer  = shap.TreeExplainer(model)
        elif ml_name == 'lgbm':
            explainer  = shap.TreeExplainer(model)
        elif ml_name == 'catb':
            explainer  = shap.TreeExplainer(model)
        else:
            explainer  = shap.Explainer(model)

        shap_vals = explainer.shap_values(df_chunk)
        if isinstance(shap_vals, list):      # clasificación binaria LightGBM
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return mean_abs, df_chunk.columns.tolist(), None
    except Exception as e:
        return None, None, str(e)


def compute_shap_importance(model_path, ml_name, df_features,
                             n_workers=N_WORKERS, sample_size=500):
    """
    Calcula la importancia media de SHAP en paralelo partiendo el DataFrame.

    Parameters
    ----------
    model_path  : ruta al archivo .pkl del modelo
    ml_name     : 'xgb', 'lgbm' o 'catb'
    df_features : DataFrame con las variables de entrada (sin target)
    n_workers   : número de procesos paralelos (mínimo 2)
    sample_size : número máximo de filas a usar

    Returns
    -------
    pd.Series con importancia media |SHAP| por feature
    """
    if not SHAP_AVAILABLE:
        print("  [WARN] shap no está instalado. Saltando cálculo SHAP.")
        return pd.Series(dtype=float)

    n_workers = max(2, n_workers)
    df_sample = df_features.sample(min(sample_size, len(df_features)), random_state=42)
    chunks    = np.array_split(df_sample, n_workers)

    tasks = [(model_path, chunk.to_json(orient='split'), ml_name)
             for chunk in chunks if len(chunk) > 0]

    agg_importance = None
    cols = None
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_shap_worker, t) for t in tasks]
        for future in as_completed(futures):
            mean_abs, columns, err = future.result()
            if err:
                print(f"  [WARN] SHAP worker error: {err}")
                continue
            if agg_importance is None:
                agg_importance = mean_abs
                cols = columns
            else:
                agg_importance += mean_abs

    if agg_importance is None:
        return pd.Series(dtype=float)

    shap_series = pd.Series(agg_importance / len(tasks), index=cols)
    return shap_series.sort_values(ascending=False)


def load_shap_history(shap_history_path):
    """Carga el historial de importancias SHAP desde un JSON."""
    if os.path.exists(shap_history_path):
        with open(shap_history_path) as f:
            return json.load(f)
    return {}


def save_shap_history(shap_history, shap_history_path):
    """Guarda el historial de importancias SHAP en un JSON."""
    os.makedirs(os.path.dirname(shap_history_path), exist_ok=True)
    with open(shap_history_path, 'w') as f:
        json.dump(shap_history, f, indent=2)


def compute_shap_variation(shap_history):
    """
    Calcula la variación de importancia SHAP entre el período actual y el anterior.

    Parameters
    ----------
    shap_history : dict {partition_str: {feature: importance, ...}, ...}

    Returns
    -------
    pd.DataFrame con columnas [feature, current, previous, variation_pct]
    """
    if len(shap_history) < 2:
        return pd.DataFrame()

    partitions = sorted(shap_history.keys())
    current_key  = partitions[-1]
    previous_key = partitions[-2]

    current  = pd.Series(shap_history[current_key])
    previous = pd.Series(shap_history[previous_key])

    df_var = pd.DataFrame({'current': current, 'previous': previous}).dropna()
    df_var['variation_pct'] = ((df_var['current'] - df_var['previous'])
                                / (df_var['previous'].abs() + 1e-9)) * 100
    return df_var.sort_values('variation_pct', key=abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR RESULTADOS DE MONITOREO
# ─────────────────────────────────────────────────────────────────────────────

def save_monitoring_results(stage_summaries, shap_series, shap_variation,
                             partition, output_dir):
    """
    Persiste los resultados de monitoreo en CSV y JSON para auditoría.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Drift por etapa
    for summary in stage_summaries:
        stage = summary['stage']
        csv_path = f"{output_dir}/drift_{stage}_{partition}.csv"
        summary['drift_df'].to_csv(csv_path)

    # SHAP importance
    if shap_series is not None and len(shap_series) > 0:
        shap_path = f"{output_dir}/shap_importance_{partition}.csv"
        shap_series.to_frame('shap_importance').to_csv(shap_path)

    # SHAP variation
    if shap_variation is not None and len(shap_variation) > 0:
        var_path = f"{output_dir}/shap_variation_{partition}.csv"
        shap_variation.to_csv(var_path)

    print(f"  [Monitoring] Resultados guardados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLERO DE CONTROL HTML (E2E Dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def _color_alert(level):
    return {'OK': '#27ae60', 'WARN': '#f39c12', 'ALERT': '#e74c3c'}.get(level, '#95a5a6')


def _drift_badge(level):
    color = _color_alert(level)
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:10px;font-size:0.8em">{level}</span>')


def generate_dashboard(stage_summaries, shap_series, shap_variation,
                        shap_history, partition, output_dir):
    """
    Genera un tablero de control HTML interactivo (E2E Monitoring Dashboard).

    Secciones
    ---------
    1. Resumen ejecutivo con semáforos por etapa
    2. Tabla de drift por etapa (top features)
    3. Score distribution drift (etapas 'score' y 'postprocessed')
    4. SHAP importance actual y variación mes a mes
    5. Evolución histórica de importancia SHAP (top 10)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Paleta y estilos ────────────────────────────────────────────────────
    css = """
    <style>
      body { font-family: 'Segoe UI', sans-serif; background:#f0f2f5; margin:0; padding:0; }
      .header { background:linear-gradient(135deg,#1a237e,#283593); color:white;
                padding:28px 40px; }
      .header h1 { margin:0; font-size:1.8em; }
      .header p  { margin:4px 0 0; opacity:0.8; font-size:0.95em; }
      .container { max-width:1200px; margin:0 auto; padding:24px 20px; }
      .card { background:white; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08);
              margin-bottom:24px; padding:24px; }
      .card h2 { margin:0 0 16px; color:#1a237e; font-size:1.15em;
                 border-bottom:2px solid #e8eaf6; padding-bottom:8px; }
      .semaphore-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
                        gap:16px; }
      .semaphore { border-radius:10px; padding:18px 20px; color:white; }
      .semaphore .stage  { font-size:0.85em; opacity:0.9; text-transform:uppercase; }
      .semaphore .value  { font-size:2em; font-weight:700; margin:4px 0; }
      .semaphore .label  { font-size:0.8em; opacity:0.85; }
      table { width:100%; border-collapse:collapse; font-size:0.88em; }
      th { background:#e8eaf6; color:#1a237e; padding:9px 12px; text-align:left; }
      td { padding:8px 12px; border-bottom:1px solid #f0f0f0; }
      tr:hover td { background:#fafafa; }
      .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
      img.chart { width:100%; border-radius:6px; }
      @media(max-width:700px){ .chart-row{grid-template-columns:1fr;} }
      .footer { text-align:center; color:#888; font-size:0.8em; padding:20px; }
    </style>
    """

    # ── Semáforos por etapa ─────────────────────────────────────────────────
    semaphore_html = '<div class="semaphore-grid">'
    for s in stage_summaries:
        bg = _color_alert('OK' if s['n_alerts'] == 0 and s['n_warns'] == 0
                          else ('WARN' if s['n_alerts'] == 0 else 'ALERT'))
        status_txt = ('✓ Sin drift' if s['n_alerts'] == 0 and s['n_warns'] == 0
                      else (f"⚠ {s['n_warns']} warns" if s['n_alerts'] == 0
                            else f"✗ {s['n_alerts']} alertas"))
        semaphore_html += f"""
        <div class="semaphore" style="background:{bg}">
          <div class="stage">{s['stage']}</div>
          <div class="value">{s['mean_drift']:.4f}</div>
          <div class="label">PSI medio · {status_txt}</div>
        </div>"""
    semaphore_html += '</div>'

    # ── Tablas de drift por etapa ───────────────────────────────────────────
    drift_tables_html = ''
    for s in stage_summaries:
        df_top = s['drift_df'].head(20)
        rows = ''
        for feat, row in df_top.iterrows():
            rows += (f'<tr><td>{feat}</td>'
                     f'<td>{row["metric_value"]:.5f}</td>'
                     f'<td>{_drift_badge(row["alert_level"])}</td></tr>')
        drift_tables_html += f"""
        <div class="card">
          <h2>Drift — Etapa: {s['stage'].upper()}
              &nbsp;<small style="font-weight:normal;color:#666">
              (PSI medio: {s['mean_drift']:.4f} · Alertas: {s['n_alerts']})</small></h2>
          <table><thead><tr><th>Feature</th><th>PSI</th><th>Estado</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>"""

    # ── Gráficas ────────────────────────────────────────────────────────────
    charts_b64 = {}

    # --- Score distribution por etapa score/postprocessed ------------------
    score_stages = [s for s in stage_summaries
                    if s['stage'] in ('score', 'postprocessed', 'score_puro', 'score_posprocesado')]

    if score_stages:
        fig, axes = plt.subplots(1, len(score_stages), figsize=(6 * len(score_stages), 4))
        if len(score_stages) == 1:
            axes = [axes]
        for ax, s in zip(axes, score_stages):
            df_d = s['drift_df']
            ax.barh(df_d.head(15).index[::-1],
                    df_d.head(15)['metric_value'].values[::-1],
                    color=[_color_alert(v) for v in df_d.head(15)['alert_level'].values[::-1]])
            ax.axvline(PSI_LOW,    color='orange', linestyle='--', lw=1, label='0.10')
            ax.axvline(PSI_MEDIUM, color='red',    linestyle='--', lw=1, label='0.20')
            ax.set_title(f"PSI — {s['stage']}", fontsize=11)
            ax.set_xlabel('PSI')
            ax.legend(fontsize=8)
        plt.tight_layout()
        chart_path = f"{output_dir}/chart_score_drift_{partition}.png"
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()
        charts_b64['score_drift'] = chart_path

    # --- SHAP importance actual --------------------------------------------
    shap_chart_path = None
    if shap_series is not None and len(shap_series) > 0:
        top_shap = shap_series.head(20)
        fig, ax  = plt.subplots(figsize=(8, 5))
        ax.barh(top_shap.index[::-1], top_shap.values[::-1], color='#3f51b5')
        ax.set_title(f'SHAP Mean |Importance| — Partición {partition}', fontsize=11)
        ax.set_xlabel('Mean |SHAP value|')
        plt.tight_layout()
        shap_chart_path = f"{output_dir}/chart_shap_{partition}.png"
        plt.savefig(shap_chart_path, dpi=120, bbox_inches='tight')
        plt.close()

    # --- SHAP variation chart ----------------------------------------------
    shap_var_chart_path = None
    if shap_variation is not None and len(shap_variation) > 0:
        top_var = shap_variation.head(20)
        colors  = ['#e74c3c' if v > 20 else ('#f39c12' if v > 10 else '#27ae60')
                   for v in top_var['variation_pct'].abs()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_var.index[::-1], top_var['variation_pct'].values[::-1], color=colors[::-1])
        ax.axvline(0,   color='black', lw=0.8)
        ax.axvline(20,  color='red',   linestyle='--', lw=1, label='+20%')
        ax.axvline(-20, color='red',   linestyle='--', lw=1, label='-20%')
        ax.set_title('Variación % SHAP vs mes anterior', fontsize=11)
        ax.set_xlabel('Variación (%)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        shap_var_chart_path = f"{output_dir}/chart_shap_variation_{partition}.png"
        plt.savefig(shap_var_chart_path, dpi=120, bbox_inches='tight')
        plt.close()

    # --- SHAP histórico (top 10) -------------------------------------------
    shap_hist_chart_path = None
    if shap_history and len(shap_history) >= 2:
        partitions = sorted(shap_history.keys())
        hist_df    = pd.DataFrame({p: shap_history[p] for p in partitions}).T
        top10_cols = hist_df.mean().nlargest(10).index
        hist_df    = hist_df[top10_cols].fillna(0)

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in top10_cols:
            ax.plot(hist_df.index, hist_df[col], marker='o', label=col)
        ax.set_title('Evolución histórica SHAP — Top 10 features', fontsize=11)
        ax.set_xlabel('Partición')
        ax.set_ylabel('Mean |SHAP|')
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        shap_hist_chart_path = f"{output_dir}/chart_shap_history_{partition}.png"
        plt.savefig(shap_hist_chart_path, dpi=120, bbox_inches='tight')
        plt.close()

    # ── SHAP HTML sections ──────────────────────────────────────────────────
    shap_html = ''
    if shap_series is not None and len(shap_series) > 0:
        chart_tag = (f'<img class="chart" src="{os.path.basename(shap_chart_path)}">'
                     if shap_chart_path else '')
        var_tag   = (f'<img class="chart" src="{os.path.basename(shap_var_chart_path)}">'
                     if shap_var_chart_path else '')
        hist_tag  = (f'<img class="chart" src="{os.path.basename(shap_hist_chart_path)}">'
                     if shap_hist_chart_path else '')

        # Variación tabla
        var_table = ''
        if shap_variation is not None and len(shap_variation) > 0:
            rows = ''
            for feat, row in shap_variation.head(20).iterrows():
                color = ('#e74c3c' if abs(row['variation_pct']) > 20
                         else ('#f39c12' if abs(row['variation_pct']) > 10 else '#27ae60'))
                rows += (f'<tr><td>{feat}</td>'
                         f'<td>{row["previous"]:.5f}</td>'
                         f'<td>{row["current"]:.5f}</td>'
                         f'<td style="color:{color};font-weight:600">'
                         f'{row["variation_pct"]:+.1f}%</td></tr>')
            var_table = f"""
            <table><thead>
              <tr><th>Feature</th><th>Mes anterior</th><th>Mes actual</th><th>Variación</th></tr>
            </thead><tbody>{rows}</tbody></table>"""

        shap_html = f"""
        <div class="card">
          <h2>📊 SHAP — Feature Importance Actual (Partición {partition})</h2>
          <div class="chart-row">
            <div>{chart_tag}</div>
            <div>{var_tag}</div>
          </div>
          <br>
          <h2>📈 Variación SHAP mes a mes</h2>
          {var_table}
        </div>
        <div class="card">
          <h2>🕐 Evolución histórica SHAP — Top 10 features</h2>
          {hist_tag}
        </div>"""

    # ── Ensamble final HTML ─────────────────────────────────────────────────
    score_drift_tag = ''
    if charts_b64.get('score_drift'):
        score_drift_tag = f"""
        <div class="card">
          <h2>📉 Score Distribution Drift (PSI por etapa)</h2>
          <img class="chart" src="{os.path.basename(charts_b64['score_drift'])}">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLOps Monitoring Dashboard — Partición {partition}</title>
  {css}
</head>
<body>
  <div class="header">
    <h1>🔍 MLOps Monitoring Dashboard</h1>
    <p>Partición: <strong>{partition}</strong> &nbsp;|&nbsp; Generado: {timestamp}
       &nbsp;|&nbsp; Workers: {N_WORKERS}</p>
  </div>
  <div class="container">

    <div class="card">
      <h2>🚦 Resumen Ejecutivo — Estado del Pipeline E2E</h2>
      {semaphore_html}
    </div>

    {drift_tables_html}
    {score_drift_tag}
    {shap_html}

  </div>
  <div class="footer">MLOps Monitoring · Generado automáticamente por monitoring_utils.py</div>
</body>
</html>"""

    dashboard_path = f"{output_dir}/dashboard_{partition}.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [Monitoring] Dashboard guardado: {dashboard_path}")
    return dashboard_path
