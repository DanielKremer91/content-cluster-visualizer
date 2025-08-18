import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from io import BytesIO
from urllib.parse import urlparse

SKLEARN_OK = True
_import_err = None
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
except Exception as e:
    SKLEARN_OK = False
    _import_err = e
import plotly.express as px
import plotly.graph_objects as go  # für graue Basisschicht & präzise Markersteuerung

# =============================
# Page setup & Branding
# =============================
st.set_page_config(page_title="ONE Semantic Content-Map", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Semantic Content-Map")

if not SKLEARN_OK:
    st.error(
        "💥 Problem beim Laden von scikit-learn / NumPy.\n\n"
        f"**Fehler:** `{_import_err}`\n\n"
        "Bitte prüfe deine Umgebung: `numpy` und `scikit-learn` Versionen müssen zueinander passen "
        "(z. B. numpy==1.26.4, scikit-learn==1.4.2)."
    )
    st.stop()


st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 765px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates
</div>
<hr>
""", unsafe_allow_html=True)

# >>> Download-Buttons rot stylen <<<
st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #e60023 !important;
    color: #ffffff !important;
    border: 1px solid #990014 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(230,0,35,0.25) !important;
}
div.stDownloadButton > button:hover {
    background-color: #cc001f !important;
    border-color: #7a0010 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Hilfe / Tool-Dokumentation (Expander)
# =============================
with st.expander("❓ Hilfe / Tool-Dokumentation", expanded=False):
    st.markdown("""
## Was macht die ONE Semantic Content-Map?

Dieses Tool macht **thematische Strukturen einer Domain sichtbar** und erlaubt dir u. a.
**thematische Ausreißer (Off-Topic-Content)** zu erkennen und **für SEO-Audits relevante Listen zu exportieren**.

### 🔄 Input
- **Pflicht:** *Embedding-Datei* (CSV/Excel) mit **URLs** und **Embedding-Spalte**  
  ↳ Optional: *Segment-Spalte* einfügen (z. B. um nach Verzeichnissen oder URL-Gruppen clustern zu können – Tipp: im Screaming Frog **Segmente** z. B. anhand der URL-/Verzeichnisstruktur definieren und **mit ausgeben lassen**)
- **Optional:** *URL-Performance-Datei* (CSV/Excel, z. B. mit Daten aus der Search Console/SISTRIX/Ahrefs etc.)  
  ↳ Alle **numerischen Spalten** daraus können zur Skalierung der **Bubble-Größe** verwendet werden. Das Tool erkennt die Spalten automatisch und bietet sie im Dropdown-Menü zur Auswahl an.
""")

    # >>> WICHTIG-Box <<<
    st.markdown("""
<div style="margin-top: 0.5rem; background:#fff8e6; border:1px solid #ffd28a; border-radius:8px; padding:10px 12px; color:#000;">
  <strong>❗WICHTIG:</strong> Achte darauf, dass deine CSV echte Spaltentrenner nutzt (z. B. Tab/Komma) und <em>nicht</em> als Ein-Spalten-Datei vorliegt – das passiert bei Screaming-Frog-Exporten schnell. <br>
  Gegebenenfalls ist vor dem Upload noch eine kurze Anpassung der <strong>Input-Dateien</strong> notwendig.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### ⚙️ Wie funktioniert’s?
- **t-SNE** projiziert hochdimensionale Embeddings auf 2D, um **Nachbarschaften** sichtbar zu machen.
- **Clustering:** *K-Means* (feste k; Anzahl der Cluster wählbar), *DBSCAN* (dichtebasiert, Cosinus-Distanz) oder vorhandene *Segments*-Spalte nutzen.    
- **Abstände:** *Euklidisch* misst Luftlinie; *Cosinus* misst **Winkel/Ähnlichkeit** der Vektoren.  
- **Bubble-Größe:** nach beliebiger **numerischer KPI** aus der Performance-Datei darstellbar.
- **Suche:** interaktive **URL-Suche** – Treffer werden farbig markiert, restliche Bubbles ausgegraut.  
- **Centroid:** thematischen **Schwerpunkt** markieren (roter Stern).

### 📤 Output (Ergebnisse)
- **Interaktives t-SNE-Chart** (HTML-Export möglich)  
- **CSV-Exports (optional):**  
  1) **Semantisch ähnliche Paare** (mit Cosinus-Score, Cosinus-Similarity ≥ Schwellenwert; frei definierbar)  
  2) **Low-Relevance-URLs** (Cosinus-Similarity zum Centroid < Schwellwert), um thematische Ausreißer-URLs „schwarz auf weiß“ vorliegen zu haben
""")

    # Optional: zweite Info-Box
    st.markdown("""
<div style="margin-top: 0.5rem; background:#fff8e6; border:1px solid #ffd28a; border-radius:8px; padding:10px 12px; color:#000;">
  <strong>💡 Komische Ergebnisse?</strong> Oft liegt es an der <strong>Embedding-Erzeugung</strong>. Genauigkeit ist entscheidend – Details
  <a href="https://www.linkedin.com/posts/daniel-kremer-b38176264_vektor-embedding-analyse-klingt-smart-wird-activity-7359197501897269249-eLmI?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEDO8dwBl0C_keb4KGiqxRXp2oPlLQjlEsY"
     target="_blank" style="color:#000; text-decoration:underline;">HIER</a>. Ein <strong>Praxisbeispiel</strong> findet ihr
  <a href="https://www.linkedin.com/posts/daniel-kremer-b38176264_%F0%9D%90%85%F0%9D%90%A2%F0%9D%90%A7%F0%9D%90%A0%F0%9D%90%9E%F0%9D%90%AB-%F0%9D%90%B0%F0%9D%90%9E%F0%9D%90%A0-%F0%9D%90%AF%F0%9D%90%A8%F0%9D%90%A7-%F0%9D%90%82%F0%9D%90%A8%F0%9D%90%A7%F0%9D%90%AD%F0%9D%90%9E%F0%9D%90%A7%F0%9D%90%AD-%F0%9D%90%80%F0%9D%90%AE%F0%9D%90%9D%F0%9D%90%A2%F0%9D%90%AD%F0%9D%90%AC-activity-7361780015908171776-3Y-f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEDO8dwBl0C_keb4KGiqxRXp2oPlLQjlEsY"
     target="_blank" style="color:#000; text-decoration:underline;">HIER</a>.
</div>
""", unsafe_allow_html=True)

# =============================
# Utilities
# =============================

def _cleanup_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def robust_read_table(uploaded_file):
    """
    Robustes Einlesen: CSV/Excel mit Encoding- und Delimiter-Fallback.
    - Bevorzugt: Excel, GSC (UTF-16 + Tab), dann Auto-Detect.
    - HARTE SICHERUNG: Wenn nur 1 Spalte rauskommt, prüfen wir Header + erste Datenzeile
      und lesen gezielt mit erkannten Delimitern (z. B. ';', ',', '\\t', '|', ':') neu ein.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    def _read_csv(bytes_data, **kwargs):
        return pd.read_csv(BytesIO(bytes_data), dtype=str, low_memory=False, **kwargs)

    # 1) Excel zuerst
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(BytesIO(raw))
            return _cleanup_headers(df)
        except Exception:
            pass

    # 2) GSC-typisch: UTF-16 + Tab
    try:
        df = _read_csv(raw, sep="\t", encoding="UTF-16")
        if df.shape[1] > 0:
            return _cleanup_headers(df)
    except Exception:
        pass

    # 3) Auto-Detect via python-engine
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "UTF-16", "UTF-16LE", "UTF-16BE"]
    hard_delims = [";", ",", "\t", "|", ":"]
    for enc in encodings:
        try:
            df = _read_csv(raw, sep=None, engine="python", encoding=enc)
            df = _cleanup_headers(df)
            if df.shape[1] == 1:
                header = str(df.columns[0])
                first_row = str(df.iloc[0, 0]) if len(df) else ""
                for d in hard_delims:
                    if d in header or d in first_row:
                        try:
                            df2 = _read_csv(raw, sep=d, encoding=enc)
                            if df2.shape[1] > 1:
                                return _cleanup_headers(df2)
                        except Exception:
                            pass
            if df.shape[1] > 0:
                return df
        except Exception:
            pass

    # 4) Raster: Encodings x feste Delimiter (Fallback)
    for enc in encodings:
        for sep in hard_delims:
            try:
                df = _read_csv(raw, sep=sep, encoding=enc)
                if df.shape[1] > 0:
                    return _cleanup_headers(df)
            except Exception:
                pass

    raise ValueError("❌ Datei konnte nicht eingelesen werden (Encoding/Trennzeichen unbekannt).")

def parse_embedding(value):
    try:
        if pd.isna(value):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                return ast.literal_eval(s)
            return ast.literal_eval(f"[{s.strip(', ')}]")
    except Exception:
        return None

def normalize_embedding_lengths(vectors: pd.Series):
    lengths = vectors.apply(lambda x: len(x) if isinstance(x, list) else 0)
    max_len = int(lengths.max()) if len(lengths) else 0

    def pad_or_trim(emb):
        if not isinstance(emb, list):
            return None
        if len(emb) < max_len:
            return emb + [0.0] * (max_len - len(emb))
        return emb[:max_len]

    return vectors.apply(pad_or_trim), max_len

def normalize_url(u: str) -> str:
    if pd.isna(u):
        return None
    try:
        p = urlparse(str(u).strip())
        netloc = p.netloc.lower()
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        query = (f"?{p.query}" if p.query else "")
        return f"{netloc}{path}{query}"
    except Exception:
        s = str(u).strip()
        s = re.sub(r"#.*$", "", s)
        if s.endswith("/"):
            s = s[:-1]
        return s.lower()

def to_numeric_series(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

def scale_sizes(series, method="log", size_min=2, size_max=10, clip_low=1, clip_high=95):
    s = to_numeric_series(series).fillna(0)
    if len(s) == 0:
        return pd.Series([], dtype=float)
    lo = np.percentile(s, clip_low)
    hi = np.percentile(s, clip_high)
    if hi <= lo:
        lo, hi = s.min(), s.max()
    s = s.clip(lo, hi)
    if method == "log":
        s = np.log1p(s)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.full(len(s), (size_min + size_max) / 2.0))
    s_norm = (s - mn) / (mx - mn)
    diam = size_min + s_norm * (size_max - size_min)
    return pd.Series(diam)

def find_column(possible_names, columns):
    for name in possible_names:
        if name in columns:
            return name
    lower = {str(c).lower(): c for c in columns}
    for name in possible_names:
        n = str(name).lower()
        if n in lower:
            return lower[n]
    return None

def autodetect_embedding_column(df: pd.DataFrame, sample=50):
    """Falls die Kandidatenliste nichts findet, erkennen wir Embedding-Spalten heuristisch."""
    def looks_like_embedding_series(s: pd.Series) -> bool:
        non_null = s.dropna().astype(str).head(sample)
        if non_null.empty:
            return False
        hits = 0
        for v in non_null:
            v = v.strip()
            if (v.startswith("[") and v.endswith("]")) or ("," in v and any(ch.isdigit() for ch in v)):
                if v.count(",") >= 5 or v.count(" ") >= 5:
                    hits += 1
        return hits >= max(3, int(len(non_null) * 0.2))
    for c in df.columns:
        try:
            if looks_like_embedding_series(df[c]):
                return c
        except Exception:
            pass
    return None

# --- Centroid-Logik: Auto/Standard/Unit-Norm ---
def norm_stats(X: np.ndarray):
    norms = np.linalg.norm(X, axis=1)
    p10, p90 = np.percentile(norms, [10, 90])
    mean, std = norms.mean(), norms.std()
    cv = std / (mean + 1e-12)
    ratio = (p90 + 1e-12) / (p10 + 1e-12)
    if ratio < 1.5 and cv < 0.15:
        level = "ok"
    elif ratio < 1.8 and cv < 0.25:
        level = "warn"
    else:
        level = "high"
    return {"cv": cv, "ratio": ratio, "level": level}

def compute_centroid(X: np.ndarray, mode: str):
    # returns (centroid_vector_1d, effective_mode_str)
    if mode.startswith("Auto"):
        stats = norm_stats(X)
        eff = "Unit-Norm" if (stats["level"] in ("warn", "high")) else "Standard"
        c, _ = compute_centroid(X, eff)
        return c, eff
    if mode.startswith("Unit"):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = np.divide(X, np.where(norms == 0, 1.0, norms))
        Xn[~np.isfinite(Xn)] = 0.0
        c = Xn.mean(axis=0)
        cn = np.linalg.norm(c)
        if cn > 0:
            c = c / cn
        return c, "Unit-Norm"
    # Standard
    return X.mean(axis=0), "Standard"

# -------------------------------------------------
# URL-Kandidaten
# -------------------------------------------------
URL_CANDIDATES_BASE = [
    "URL", "Page", "Pages",
    "Adresse", "Address",
    "Seite", "Seiten",
    "URLs",
    "URL-Adresse", "URL Adresse",
]
URL_CANDIDATES_GSC_EXTRA = ["Landing Page", "Seiten-URL", "Seiten URL"]

# =============================
# Uploads
# =============================
st.subheader("1) Embedding-Datei hochladen")
emb_file = st.file_uploader("CSV/Excel mit URLs und Embeddings", type=["csv", "xlsx", "xls"], key="emb")

st.subheader("2) Optional: Performance-/Metrik-Datei hochladen (z. B. GSC, SISTRIX, Ahrefs)")
perf_file = st.file_uploader("Performance-/Metrik-CSV/Excel (optional)", type=["csv", "xlsx", "xls"], key="perf")

if emb_file is None:
    st.info("Bitte zuerst die Embedding-Datei hochladen.")
    st.stop()

# Read embeddings table
try:
    df = robust_read_table(emb_file)
except Exception as e:
    st.error(str(e))
    st.stop()

st.caption(f"Columns detected (Embedding-Datei): {list(df.columns)}")

# --- Spaltenfindung (robust) ---
url_col = find_column(URL_CANDIDATES_BASE, df.columns)
if url_col is None:
    for c in df.columns:
        n = str(c).lower().replace("-", " ")
        if any(tok in n for tok in ["url", "page", "adresse", "address", "seite", "seiten"]):
            url_col = c
            break

embedding_col = find_column([
    "ChatGPT Embedding Erzeugung",
    "ChatGPT Embedding Erzeugung 1",
    "Embedding",
    "Embeddings",
    "Embedding Vector",
    "OpenAI Embedding",
    "Extract embeddings from page content",
], df.columns)
if embedding_col is None:
    for c in df.columns:
        colname = str(c).lower()
        if ("embedding" in colname) or ("embed" in colname) or ("vector" in colname):
            embedding_col = c
            break
if embedding_col is None:
    embedding_col = autodetect_embedding_column(df)

if url_col is None or embedding_col is None:
    st.error(
        "❌ URL- oder Embedding-Spalte nicht gefunden.\n\n"
        f"Gefundene Spalten: {list(df.columns)}\n\n"
        "Erwarte URL-Spalte (z. B. URL/Adresse/Page) und eine Embedding-Spalte (z. B. 'Embedding' oder eine Liste von Zahlen)."
    )
    st.stop()

# ---- Segment/Cluster-Spalte vorab erkennen (für UI) ----
SEGMENT_NAME_CANDIDATES = ["Segmente", "Segment", "Segments", "Cluster"]
def detect_segment_col(df_input):
    seg = find_column(SEGMENT_NAME_CANDIDATES, df_input.columns)
    if seg is None:
        for c in df_input.columns:
            n = str(c).lower().replace("-", " ").replace("_", " ")
            tokens = n.split()
            if any(tok in tokens for tok in ["segment", "segments", "cluster"]):
                return c
    return seg
segment_col_global = detect_segment_col(df)

with st.spinner("Verarbeite Embeddings…"):
    df["embedding_vector"] = df[embedding_col].apply(parse_embedding)
    df_valid = df[df["embedding_vector"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    df_valid["embedding_vector"], dim = normalize_embedding_lengths(df_valid["embedding_vector"])

if len(df_valid) < 5:
    st.error("❌ Zu wenige gültige Embeddings. Mindestens 5 erforderlich.")
    st.stop()

embedding_matrix = np.array(df_valid["embedding_vector"].tolist())
st.caption(f"✅ Gültige Embeddings: {len(df_valid)} · Vektor-Dim: {embedding_matrix.shape[1]}")

# =============================
# Interaktive URL-Suche (nur Darstellung)
# =============================
search_q = st.text_input(
    "🔍 URL-Suche (Teilstring)",
    value="",
    help="Markiert Treffer im Plot. Beeinflusst weder Berechnungen noch Exporte."
)

# Optional: Performance-/Metrik-Datei einlesen + numerische Kandidaten sammeln
perf_df = None
perf_url_col = clicks_col = impressions_col = None
perf_metric_candidates = []
if perf_file is not None:
    try:
        perf_df = robust_read_table(perf_file)
        st.caption(f"Columns detected (Performance-Datei): {list(perf_df.columns)}")

        perf_url_col = find_column(URL_CANDIDATES_BASE + URL_CANDIDATES_GSC_EXTRA, perf_df.columns)
        if perf_url_col is None:
            for c in perf_df.columns:
                n = str(c).lower().replace("-", " ")
                if any(tok in n for tok in ["url", "page", "adresse", "address", "seite", "seiten", "landing"]):
                    perf_url_col = c
                    break

        for c in perf_df.columns:
            name = str(c).strip().lower()
            if clicks_col is None and ("klick" in name or "click" in name):
                clicks_col = c
            if impressions_col is None and ("impress" in name):
                impressions_col = c

        if perf_url_col is None:
            st.warning("⚠️ Konnte URL-Spalte in der Performance-/Metrik-Datei nicht erkennen – Bubbles werden nicht skaliert.")
            perf_df = None
        else:
            for c in perf_df.columns:
                if c == perf_url_col:
                    continue
                s_num = to_numeric_series(perf_df[c])
                valid_ratio = float(s_num.notna().mean()) if len(s_num) else 0.0
                if valid_ratio >= 0.30 and s_num.nunique(dropna=True) > 1:
                    perf_metric_candidates.append(c)

            def sort_key(x):
                xl = str(x).lower()
                if clicks_col and x == clicks_col:
                    return (0, x)
                if impressions_col and x == impressions_col:
                    return (1, x)
                if any(k in xl for k in ["click", "klick"]):
                    return (2, x)
                if "impress" in xl:
                    return (3, x)
                return (4, x)

            perf_metric_candidates = sorted(set(perf_metric_candidates), key=sort_key)

    except Exception as e:
        st.warning(f"Performance-/Metrik-Datei konnte nicht verarbeitet werden: {e}")
        perf_df = None
        perf_metric_candidates = []

# =============================
# Sidebar Controls (dynamisch)
# =============================
st.sidebar.header("Einstellungen")

# Dynamische Optionsliste: 'Segments' nur, wenn Spalte existiert
cluster_options = ["K-Means", "DBSCAN (Cosinus)"]
if segment_col_global:
    cluster_options.insert(1, "Segments")

cluster_method = st.sidebar.selectbox(
    "Clustermethode",
    cluster_options,
    help=("K-Means: feste Clusterzahl • "
          + (f"Segments: nutzt erkannte Spalte „{segment_col_global}“ • " if segment_col_global else "")
          + "DBSCAN: dichtebasiert (Cosinus)")
)
if not segment_col_global:
    st.sidebar.caption("ℹ️ Keine Segment-/Cluster-Spalte erkannt – Option „Segments“ ist deaktiviert.")
else:
    st.sidebar.caption(f"✅ Segment-/Cluster-Spalte erkannt: „{segment_col_global}“")

cluster_k = st.sidebar.slider(
    "Cluster (nur K-Means)",
    min_value=2, max_value=20, value=8, step=1,
    help=("Legt fest, in wie viele Gruppen (Cluster) die Punkte bei der K-Means-Methode unterteilt werden. "
          "Eine höhere Zahl erzeugt kleinere, spezialisiertere Gruppen; eine niedrigere Zahl erzeugt größere, "
          "allgemeinere Cluster. Nur relevant, wenn ‚K-Means‘ gewählt ist.")
)

# Darstellungsmethode (Abstand)
metric_label = st.sidebar.selectbox(
    "Darstellungsmethode (Abstand)",
    ["Euklidisch", "Cosinus"],
    help=("Bestimmt, wie der Abstand bzw. die Ähnlichkeit zwischen Seiten berechnet wird, bevor die t-SNE-Visualisierung erstellt wird.\n\n"
          "Euklidisch: misst die „Luftlinie“ im Embedding-Raum, stabil und Standard.\n\n"
          "Cosinus: misst den Winkel zwischen Vektoren, gut geeignet für semantische Ähnlichkeiten.\n"
          "Die Wahl beeinflusst nur die Darstellung, nicht die eigentlichen Daten.")
)
tsne_metric = "euclidean" if metric_label == "Euklidisch" else "cosine"

# Bubblegröße nach – dynamisch; Auto-Vorauswahl: Keine Skalierung
size_options = ["Keine Skalierung"]
if perf_metric_candidates:
    size_options += perf_metric_candidates

size_by = st.sidebar.selectbox(
    "Bubblegröße nach",
    size_options,
    index=0,
    help=("Welche Spalte aus der Performance-/Metrik-Datei (z. B. GSC/SISTRIX/Ahrefs) bestimmt die Blasengröße? "
          "'Keine Skalierung' = konstant. Es werden nur numerische Spalten angeboten.")
)

# Skalierung + Hilfetext
size_method = st.sidebar.radio(
    "Skalierung",
    ["Logarithmisch (log1p)", "Linear (Min–Max)"],
    index=0,
    help=("Bestimmt, wie die Blasengrößen aus der gewählten Metrik berechnet werden.\n\n"
          "- Logarithmisch (log1p): komprimiert große Werteunterschiede, robust gegen Ausreißer; ideal bei schiefen Verteilungen.\n"
          "- Linear (Min–Max): erhält Proportionen direkt; kann bei Ausreißern sehr große/kleine Bubbles erzeugen.\n\n"
          "Tipp: Nutze ‚Perzentil-Grenze unten/oben (%)‘, um Extremwerte abzuschneiden, und ‚Min-/Max-Größe‘ sowie ‚Bubble-Scale‘, "
          "um die Darstellung feinzujustieren.\n\n"
          "Hinweis: In der Praxis ist „Logarithmisch“ bei SEO/GSC-Daten fast immer die bessere Wahl (Long Tail, schiefe Verteilungen).")
)

# Min-/Max-Größe + Perzentil-Grenzen
size_min = st.sidebar.slider(
    "Min-Größe (px)", 1, 12, 2,
    help=("Kleinster Bubble-Durchmesser in Pixeln nach der Skalierung. "
          "Verhindert, dass sehr kleine Werte ‚verschwinden‘.")
)
size_max = st.sidebar.slider(
    "Max-Größe (px)", 6, 40, 10,
    help=("Größter Bubble-Durchmesser in Pixeln nach der Skalierung. "
          "Zu groß kann zu starker Überlappung führen.")
)
clip_low = st.sidebar.slider(
    "Perzentil-Grenze unten (%)", 0, 20, 1,
    help=("Hebt sehr kleine Werte auf diese Untergrenze an (Perzentil). So „verschwinden“ kleine Bubbles nicht. "
          "Wirkt nur auf die Bubble-Größen, nicht auf t-SNE, Cluster oder Exporte.")
)
clip_high = st.sidebar.slider(
    "Perzentil-Grenze oben (%)", 80, 100, 95,
    help=("Begrenzt sehr große Werte auf diese Obergrenze (Perzentil), damit einzelne Riesen-Bubbles die Darstellung nicht dominieren. "
          "Wirkt nur auf die Bubble-Größen, nicht auf t-SNE, Cluster oder Exporte.")
)

# Centroid-Optionen
show_centroid = st.sidebar.checkbox(
    "Centroid markieren", value=False,
    help="Markiert den thematischen Schwerpunkt der analysierten URLs (Centroid), berechnet als Durchschnitt aller Embeddings."
)
with st.sidebar.expander("Erweitert: Centroid", expanded=False):
    centroid_mode = st.radio(
        "Centroid-Modus",
        ["Auto (empfohlen)", "Standard", "Unit-Norm"],
        index=0,
        help=("Wie der thematische Schwerpunkt (Centroid) berechnet wird.\n"
              "• Auto: nutzt Standard, wechselt bei hoher Normstreuung automatisch zu Unit-Norm.\n"
              "• Standard: arithmetischer Mittelwert der Vektoren.\n"
              "• Unit-Norm: erst jeden Vektor auf Länge 1 normieren, dann mitteln (richtungsbasiert).")
    )
centroid_size = st.sidebar.slider(
    "Centroid-Sterngröße (px)", 10, 40, 22, 1,
    help="Größe des roten Sterns, der den Centroid visualisiert.",
    disabled=not show_centroid
)

# Bubble-Scale und Hintergrundfarbe (vor Export-Überschrift)
if perf_df is not None and (size_by != "Keine Skalierung"):
    bubble_scale = st.sidebar.slider(
        "Bubble-Scale (global)", 0.20, 2.00, 1.00, 0.05,
        help=("Globaler Zoomfaktor für die Blasengrößen: multipliziert alle Durchmesser nach der Berechnung "
              "(Min/Max, Perzentil-Grenzen, Log/Linear). Praktisch zum schnellen Feinjustieren, ohne Min/Max zu ändern.")
    )
else:
    bubble_scale = 1.0  # Standard: kein globales Upscaling/Downscaling

bg_color = st.sidebar.color_picker("Hintergrundfarbe für Bubble-Chart", value="#FFFFFF")

# Kleine Section-Überschrift für Exporte
st.sidebar.markdown("**Weitere Exportmöglichkeiten**")

# Export 1: Paar-Ähnlichkeiten (Cosinus) mit Schwellwert
export_csv = st.sidebar.checkbox(
    "Semantisch ähnliche URLs exportieren", value=False,
    help="Export semantisch ähnlicher URL-Paare mit einer Cosinus Similarity über dem gewählten Schwellenwert als CSV"
)
sim_threshold = st.sidebar.slider(
    "Ähnlichkeitsschwelle (Cosinus)",
    min_value=0.00, max_value=1.00, value=0.00, step=0.01,
    help=("Nur Paare mit Cosinus-Ähnlichkeit ≥ Schwellenwert werden exportiert. "
          "1.00 = sehr ähnlich/identisch, 0.00 = keine Ähnlichkeit."),
    disabled=not export_csv
)

# Export 2: Low-Relevance (Centroid-Ähnlichkeit) mit Schwellwert
export_lowrel_csv = st.sidebar.checkbox(
    "Low-Relevance-URLs exportieren", value=False,
    help=("Low-Relevance URLs (thematische Ausreißer-URLs) als CSV exportieren. "
          "Beispielsweise alle URLs mit einer Cosinus Similarity von unter 0,4 zum Centroid (Durchschnitt aller Embeddings). "
          "Schwellenwert ist flexibel anpassbar.")
)
lowrel_threshold = st.sidebar.slider(
    "Ähnlichkeitsschwelle zum Centroid (Cosinus)",
    min_value=0.00, max_value=1.00, value=0.40, step=0.01,
    help=("Nur Seiten mit Cosinus-Ähnlichkeit zum Centroid unterhalb der Schwelle werden exportiert. "
          "Niedrige Werte = thematisch abweichend."),
    disabled=not export_lowrel_csv
)

# Export-Limits (konfigurierbar)
unlimited_export = st.sidebar.checkbox(
    "Kein Limit für Export", value=False,
    help="Hebt die Zeilenbegrenzung auf. Vorsicht: Sehr große CSVs können Browser/Speicher überlasten."
)
if not unlimited_export:
    max_export_rows = st.sidebar.number_input(
        "Max. Zeilen pro Export", min_value=50_000, max_value=5_000_000, step=50_000, value=250_000,
        help="Begrenzt die Zeilenanzahl in Exporten (Performance & Speicher)."
    )
else:
    max_export_rows = None

recalc = st.sidebar.button("Let's Go / Refresh", type="primary")

# =============================
# Build data (heavy) & cache in session_state
# =============================

def _build_hover_cols(merged, metric_col):
    h = {url_col: True, "Cluster": True}
    for extra in {metric_col}:
        if extra and extra in merged.columns:
            h[extra] = True
    return h

def build_data_and_cache():
    """Schwere Schritte ausführen und Ergebnis in Session-State ablegen."""
    merged = df_valid.copy()

    # Merge Performance-Metriken (alle Kandidaten-Spalten)
    if isinstance(perf_df, pd.DataFrame) and perf_url_col:
        merged["__join"] = merged[url_col].apply(normalize_url)
        perf_local = perf_df.copy()
        perf_local["__join"] = perf_local[perf_url_col].apply(normalize_url)
        keep_cols = ["__join"]
        perf_keep = perf_local[keep_cols + list(set(perf_metric_candidates))].drop_duplicates("__join") \
            if perf_metric_candidates else perf_local[keep_cols].drop_duplicates("__join")
        merged = merged.merge(perf_keep, on="__join", how="left")
        merged.drop(columns=["__join"], inplace=True, errors="ignore")

    # t-SNE (+ optionaler Centroid-Punkt)
    perplexity = int(min(30, max(5, len(merged) // 3)))
    X = np.array(merged["embedding_vector"].tolist())
    use_centroid_flag = bool(show_centroid)
    if use_centroid_flag:
        centroid_vec, centroid_mode_eff = compute_centroid(X, centroid_mode)
        X_tsne = np.vstack([X, centroid_vec[None, :]])
    else:
        centroid_mode_eff = None
        X_tsne = X

    tsne = TSNE(n_components=2, metric=tsne_metric, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(X_tsne)

    merged["tsne_x"] = tsne_result[: len(X), 0]
    merged["tsne_y"] = tsne_result[: len(X), 1]

    # Cluster
    method = cluster_method
    segment_col = segment_col_global

    if method == "K-Means":
        kmeans = KMeans(n_clusters=cluster_k, random_state=42)
        merged["Cluster"] = kmeans.fit_predict(X).astype(str)
    elif method == "DBSCAN (Cosinus)":
        cos_dist = cosine_distances(X)
        dbscan = DBSCAN(eps=0.3, min_samples=5, metric="precomputed")
        merged["Cluster"] = dbscan.fit_predict(cos_dist).astype(str)
    elif method == "Segments":
        if segment_col:
            merged["Cluster"] = merged[segment_col].fillna("Unbekannt").astype(str)
        else:
            merged["Cluster"] = "Kein Segment"
    else:
        merged["Cluster"] = "Kein Segment"

    # Bubblegrößen
    scaled = False
    metric_col = None
    if size_by != "Keine Skalierung":
        metric_col = size_by

    if metric_col and metric_col in merged.columns:
        mth = "log" if size_method.startswith("Log") else "linear"
        merged["__marker_size"] = scale_sizes(
            merged[metric_col],
            method=mth,
            size_min=size_min,
            size_max=size_max,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        scaled = True
    else:
        merged["__marker_size"] = float(size_min)

    if scaled:
        merged["__marker_px"] = (merged["__marker_size"] * float(bubble_scale)).clip(lower=1)
    else:
        merged["__marker_px"] = max(1, int(size_min * float(bubble_scale)))

    # Cache
    st.session_state["merged_cached"] = merged
    st.session_state["scaled_cached"] = scaled
    st.session_state["hover_cols_cached"] = _build_hover_cols(merged, metric_col)
    st.session_state["plot_title_cached"] = "🔍 t-SNE der Seiten-Embeddings (mit Skalierung)" if scaled else "🔍 t-SNE der Seiten-Embeddings"
    st.session_state["bg_color_cached"] = bg_color
    st.session_state["highlight_px_cached"] = max(int(size_min * float(bubble_scale)) + 6, 8)
    st.session_state["url_col_cached"] = url_col
    st.session_state["centroid_in_tsne"] = use_centroid_flag
    if use_centroid_flag:
        st.session_state["centroid_xy"] = (tsne_result[len(X), 0], tsne_result[len(X), 1])
        st.session_state["centroid_mode_eff"] = centroid_mode_eff
    else:
        st.session_state["centroid_xy"] = None
        st.session_state["centroid_mode_eff"] = None

def render_plot_from_cache(q: str):
    """Zeichnet den Plot aus dem Cache neu; bei Suche: Rest grau, Treffer farbig.
       Legende: numerisch/alpha sortiert; Hover-Kästchen übernimmt Bubble-Farbe."""
    merged = st.session_state.get("merged_cached")
    if merged is None:
        st.info("Bitte zuerst Einstellungen wählen und auf **Let's Go / Refresh** klicken.")
        return

    scaled = st.session_state.get("scaled_cached", False)
    hover_cols = st.session_state.get("hover_cols_cached", {url_col: True, "Cluster": True})
    title = st.session_state.get("plot_title_cached", "🔍 t-SNE der Seiten-Embeddings")
    bg = st.session_state.get("bg_color_cached", "#FFFFFF")
    url_c = st.session_state.get("url_col_cached", url_col)
    highlight_px = st.session_state.get("highlight_px_cached", 10)
    centroid_xy = st.session_state.get("centroid_xy", None)
    centroid_mode_eff = st.session_state.get("centroid_mode_eff", None)

    q = (q or "").strip().lower()

    if q:
        # --- Suchmodus: Basisschicht grau, nur Treffer farbig ---
        fig = go.Figure()

        # Basisschicht (alle Punkte grau, Hover aus)
        fig.add_trace(go.Scatter(
            x=merged["tsne_x"], y=merged["tsne_y"], mode="markers", name="Alle",
            marker=dict(
                size=merged["__marker_px"].tolist(),
                color="lightgray",
                opacity=0.35,
                line=dict(width=0.5, color="white")
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # Treffer
        mask = merged[url_c].astype(str).str.lower().str.contains(q, na=False)
        if mask.any():
            hi = merged[mask]
            hover_texts = []
            for _, row in hi.iterrows():
                extras = []
                if "Cluster" in row:
                    extras.append(f"Cluster: {row['Cluster']}")
                hover_texts.append(f"{row[url_c]}<br>" + ("<br>".join(extras) if extras else ""))
            fig.add_trace(go.Scatter(
                x=hi["tsne_x"], y=hi["tsne_y"], mode="markers", name="Treffer",
                marker=dict(
                    size=hi["__marker_px"].tolist(),
                    color="orange",
                    line=dict(width=2, color="black")
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                hoverlabel=dict(bgcolor="orange", font_color="black", bordercolor="black"),
                showlegend=False
            ))
            st.caption(f"✨ {int(mask.sum())} Treffer für „{q}“")
    else:
        # --- Normalmodus: farbige Cluster (sortierte Legende) ---
        merged["Cluster"] = merged["Cluster"].astype(str)
        cluster_labels = merged["Cluster"].unique().tolist()

        def _legend_sort_key(lbl):
            try:
                return (0, float(lbl))  # numerische Labels (inkl. -1) zuerst
            except Exception:
                return (1, str(lbl).lower())  # danach alphabetisch

        cluster_order = [lbl for lbl in sorted(cluster_labels, key=_legend_sort_key)]

        fig = px.scatter(
            merged,
            x="tsne_x",
            y="tsne_y",
            color="Cluster",
            category_orders={"Cluster": cluster_order},
            hover_data=hover_cols,
            template="plotly_white",
            title=title,
        )

        # Größen je Trace setzen, echte Datentraces aus der Legende ausblenden
        # und Hoverlabel-Farbe = Marker-Farbe setzen
        color_by_name = {}
        for tr in fig.data:
            mask = (merged["Cluster"] == tr.name)
            sizes = merged.loc[mask, "__marker_px"].tolist()
            tr.marker.update(size=sizes, sizemode="diameter", opacity=0.55, line=dict(width=0.5, color="white"))
            # Farbe ermitteln
            cval = tr.marker.color
            if isinstance(cval, (list, np.ndarray)) and len(cval) > 0:
                cval = cval[0]
            color_by_name[tr.name] = cval
            tr.hoverlabel = dict(bgcolor=cval, font_color="white", bordercolor="black")
            tr.legendgroup = tr.name
            tr.showlegend = False  # echte Datentraces aus Legende nehmen

        # Dummy-Legendentraces in gewünschter Reihenfolge hinzufügen
        for name in cluster_order:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=True,
                marker=dict(size=12, color=color_by_name.get(name, None), line=dict(width=0.5, color="white")),
                hoverinfo="skip"
            ))

    # Centroid optional
    if centroid_xy is not None:
        cx, cy = centroid_xy
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers", name="Centroid",
            marker=dict(symbol="star", size=int(centroid_size), color="red", line=dict(width=1, color="black")),
            hoverlabel=dict(bgcolor="red", font_color="white", bordercolor="black")
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        height=750,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title="Cluster",
        showlegend=True,
        dragmode="zoom",
        hovermode="closest",
        legend=dict(itemsizing="constant")
    )

    st.subheader("📈 Visualisierung")
    if centroid_mode_eff:
        st.caption(f"Centroid-Modus aktiv: {centroid_mode_eff}")
    st.plotly_chart(fig, use_container_width=True)

    # Download HTML
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label="📥 Interaktive HTML-Datei herunterladen",
        data=html_bytes,
        file_name="tsne_plot.html",
        mime="text/html",
    )

# =============================
# Run (heavy on refresh, light on search)
# =============================
if recalc:
    with st.spinner("Berechne t-SNE & erstelle Plot…"):
        build_data_and_cache()
        render_plot_from_cache(search_q)
else:
    render_plot_from_cache(search_q)

# =============================
# Exporte (unabhängig von Suche!)
# =============================
if export_csv:
    merged_cached = st.session_state.get("merged_cached")
    if merged_cached is not None:
        with st.spinner("Berechne semantische Ähnlichkeiten…"):
            url_list = merged_cached[url_col].astype(str).tolist()
            X_raw = np.array(merged_cached["embedding_vector"].tolist()).astype("float32")
            thr = float(sim_threshold)
            pairs = []

            # --- SKLEARN (exakt, O(N^2)) -----------------------------------
            sim_matrix = cosine_similarity(X_raw)
            n = len(url_list)
            est_pairs = n * (n - 1) // 2
            if unlimited_export and est_pairs > 2_000_000 and thr <= 0.2:
                st.warning(f"Viele Paare erwartet (~{est_pairs:,}). "
                           f"Niedrige Schwelle + kein Limit kann sehr große CSVs erzeugen.")

            # Nur obere Dreiecksmatrix (i<j): eindeutige Paare, keine Selbstpaare
            for i in range(n):
                row = sim_matrix[i, i+1:]
                j_idx = np.where(row >= thr)[0]
                if len(j_idx):
                    for off in j_idx:
                        j = i + 1 + int(off)
                        s = float(sim_matrix[i, j])
                        pairs.append({
                            "URL_A": url_list[i],
                            "URL_B": url_list[j],
                            "Cosinus_Ähnlichkeit": s,
                            "Match-Typ": "Similarity (sklearn)"
                        })

            if not pairs:
                st.warning("Keine Paare über der eingestellten Ähnlichkeitsschwelle gefunden.")
            else:
                # Optionales Limit
                if (max_export_rows is not None) and (len(pairs) > max_export_rows):
                    st.warning(f"Export auf {int(max_export_rows):,} Zeilen begrenzt (von {len(pairs):,}).")
                    pairs = pairs[: int(max_export_rows)]

                sim_df = pd.DataFrame(pairs)
                sim_df = sim_df.sort_values("Cosinus_Ähnlichkeit", ascending=False, kind="stable")

                csv_bytes = sim_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 Cosinus-Ähnlichkeiten als CSV (≥ {thr:.2f}, sklearn)",
                    data=csv_bytes,
                    file_name=f"cosinus_aehnlichkeiten_ge_{thr:.2f}_sklearn.csv",
                    mime="text/csv",
                )
    else:
        st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")

if export_lowrel_csv:
    merged_cached = st.session_state.get("merged_cached")
    if merged_cached is not None:
        with st.spinner("Berechne Centroid-Ähnlichkeiten pro URL…"):
            X = np.array(merged_cached["embedding_vector"].tolist())
            centroid_vec, centroid_mode_eff_export = compute_centroid(X, centroid_mode)
            centroid_sim = cosine_similarity(X, centroid_vec[None, :]).ravel()

            low_thr = float(lowrel_threshold)
            export_df = pd.DataFrame({
                "URL": merged_cached[url_col].astype(str).values,
                "Cosinus_Ähnlichkeit_zum_Centroid": centroid_sim
            })

            if "Cluster" in merged_cached.columns:
                export_df["Cluster"] = merged_cached["Cluster"].astype(str).values

            if size_by != "Keine Skalierung" and size_by in merged_cached.columns:
                export_df[size_by] = merged_cached[size_by].values

            export_df = export_df[export_df["Cosinus_Ähnlichkeit_zum_Centroid"] < low_thr].copy()
            export_df = export_df.sort_values("Cosinus_Ähnlichkeit_zum_Centroid", ascending=True)

            if export_df.empty:
                st.warning("Keine Seiten unterhalb der eingestellten Centroid-Schwelle gefunden.")
            else:
                if (max_export_rows is not None) and (len(export_df) > max_export_rows):
                    st.warning(f"Export auf {int(max_export_rows):,} Zeilen begrenzt (von {len(export_df):,}).")
                    export_df = export_df.head(int(max_export_rows))

                csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 Low-Relevance-URLs als CSV (Centroid < {low_thr:.2f})",
                    data=csv_bytes,
                    file_name=f"low_relevance_urls_centroid_lt_{low_thr:.2f}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")
