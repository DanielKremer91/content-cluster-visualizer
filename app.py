import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from io import BytesIO
from urllib.parse import urlparse
import platform

# Immer detaillierte Fehlermeldungen im UI anzeigen
st.set_option("client.showErrorDetails", True)

# =============================
# Page setup & Branding
# =============================
st.set_page_config(page_title="ONE Semantic Content-Map", layout="wide")

# --- scikit-learn Lazy-Import + Guard ---
SKLEARN_OK = True
_import_err = None
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    SKLEARN_OK = False
    _import_err = e

# --- FAISS (optional) ---
FAISS_OK = True
_faiss_err = None
try:
    import faiss  # wird nur für den schnellen Similarity-Export genutzt
except Exception as e:
    FAISS_OK = False
    _faiss_err = e

import plotly.express as px
import plotly.graph_objects as go  # für graue Basisschicht & präzise Markersteuerung

# Remote-Logo robust laden (kein Crash, wenn Bild nicht geht)
try:
    st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
except Exception as _img_err:
    st.caption(f"Logo konnte nicht geladen werden ({_img_err}). Überspringe Bild…")

st.title("ONE Semantic Content-Map")

# Früher Abbruch mit sauberer Meldung, falls sklearn/NumPy kaputt
if not SKLEARN_OK:
    st.error(
        "💥 Problem beim Laden von scikit-learn / NumPy.\n\n"
        f"**Fehler:** `{_import_err}`\n\n"
        "Bitte prüfe die Umgebung (Versionen in der Sidebar)."
    )
    st.stop()

# Sidebar: Versionen anzeigen (schnelle Diagnose)
VER_PY = platform.python_version()
try:
    import sklearn as _skl
    VER_SKL = _skl.__version__
except Exception:
    VER_SKL = "n/a"
import numpy as _np
VER_NP = _np.__version__
VER_PD = pd.__version__
st.sidebar.info(f"🔧 Python {VER_PY} · NumPy {VER_NP} · pandas {VER_PD} · scikit-learn {VER_SKL}")

if not FAISS_OK:
    st.sidebar.caption(f"ℹ️ FAISS nicht verfügbar: {_faiss_err}")

st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 1.2em; max-width: 765px; margin-bottom: 1.5em; line-height: 1.5;">
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

### ⚙️ Wie funktioniert’s?
- **2D-Projektion:** *t-SNE* (fix) – feine Nachbarschaften, gut für Clustervisualisierung.
- **Clustering:** *K-Means* (feste k), *DBSCAN* (dichtebasiert, Cosinus) oder vorhandene *Segments*-Spalte.
- **Abstände:** *Euklidisch* oder *Cosinus (schnell)* (L2-normalisiert + euklidisch).
- **Bubble-Größe:** nach beliebiger **numerischer KPI** aus der Performance-Datei darstellbar.
- **Suche:** interaktive **URL-Suche** – Treffer farbig, Rest ausgegraut.
- **Centroid:** thematischen Schwerpunkt markieren (roter Stern).

### 📤 Output (Ergebnisse)
- **Interaktive Visualisierung** (HTML-Export)
- **CSV-Exports (optional):**
  1. **Semantisch ähnliche URL-Paare**  
     – Modus *Schwellenwert* (blockweis, exakt) **oder** *Top-N (FAISS, schnell)*  
  2. **Low-Relevance-URLs** (Cosinus-Similarity zum Centroid < Schwellenwert)
""", unsafe_allow_html=True)

# =============================
# Utilities
# =============================

def _cleanup_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

# --------- Schnelles, cachebares Einlesen (Bytes-basiert) ----------
def _read_csv_bytes(bytes_data, **kwargs):
    return pd.read_csv(BytesIO(bytes_data), dtype=str, low_memory=False, **kwargs)

@st.cache_data(show_spinner=False)
def robust_read_table_bytes(raw: bytes, name: str):
    """
    Robustes Einlesen aus Bytes: CSV/Excel mit Encoding- und Delimiter-Fallback.
    """
    name = (name or "").lower()

    # 1) Excel zuerst
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(BytesIO(raw))
            return _cleanup_headers(df)
        except Exception:
            pass

    # 2) GSC-typisch: UTF-16 + Tab
    try:
        df = _read_csv_bytes(raw, sep="\t", encoding="UTF-16")
        if df.shape[1] > 0:
            return _cleanup_headers(df)
    except Exception:
        pass

    # 3) Auto-Detect via python-engine
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "UTF-16", "UTF-16LE", "UTF-16BE"]
    hard_delims = [";", ",", "\t", "|", ":"]
    for enc in encodings:
        try:
            df = _read_csv_bytes(raw, sep=None, engine="python", encoding=enc)
            df = _cleanup_headers(df)
            if df.shape[1] == 1:
                header = str(df.columns[0])
                first_row = str(df.iloc[0, 0]) if len(df) else ""
                for d in hard_delims:
                    if d in header or d in first_row:
                        try:
                            df2 = _read_csv_bytes(raw, sep=d, encoding=enc)
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
                df = _read_csv_bytes(raw, sep=sep, encoding=enc)
                if df.shape[1] > 0:
                    return _cleanup_headers(df)
            except Exception:
                pass

    raise ValueError("❌ Datei konnte nicht eingelesen werden (Encoding/Trennzeichen unbekannt).")

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
    diam = size_min + s_norm * (size_max - size_max if size_max == size_min else (size_max - size_min))
    return pd.Series(diam)

# --------- Embedding-Parsing ----------
def parse_embedding_fast(val):
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, np.ndarray):
        return val.astype(np.float32, copy=False)
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s and s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    arr = np.fromstring(s, sep=",", dtype=np.float32)
    return arr if arr.size else None

def pad_to_maxdim(arrs):
    lengths = [a.size if isinstance(a, np.ndarray) else 0 for a in arrs]
    max_dim = int(max(lengths)) if lengths else 0
    out = []
    for a in arrs:
        if not isinstance(a, np.ndarray):
            out.append(None)
        elif a.size < max_dim:
            out.append(np.pad(a, (0, max_dim - a.size)))
        else:
            out.append(a[:max_dim])
    return out, max_dim

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    X[~np.isfinite(X)] = 0.0
    return X

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
        Xn = l2_normalize_rows(X)
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
# Hauptlogik in Try/Except: zeigt echte Tracebacks statt "Oh no"
# =============================
try:
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

    # --------- Eingelesene Dateien cachen ----------
    try:
        emb_raw = emb_file.getvalue()
        df = robust_read_table_bytes(emb_raw, emb_file.name)
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

    # --------- Embeddings parsen & normalisieren (gecacht) ----------
    @st.cache_data(show_spinner=False)
    def parse_and_normalize_embeddings(df_in: pd.DataFrame, col: str):
        # schneller Parser
        vecs = df_in[col].map(parse_embedding_fast)
        df_tmp = df_in.copy()
        df_tmp["embedding_vector"] = vecs
        valid = df_tmp[df_tmp["embedding_vector"].map(lambda x: isinstance(x, np.ndarray) and x.size > 0)].copy()
        padded, dim = pad_to_maxdim(valid["embedding_vector"].tolist())
        valid["embedding_vector"] = padded
        X = np.stack(valid["embedding_vector"].values).astype(np.float32, copy=False)
        return valid, X, dim

    with st.spinner("Verarbeite Embeddings…"):
        df_valid, embedding_matrix, dim = parse_and_normalize_embeddings(df, embedding_col)

    if len(df_valid) < 5:
        st.error("❌ Zu wenige gültige Embeddings. Mindestens 5 erforderlich.")
        st.stop()

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
            perf_raw = perf_file.getvalue()
            perf_df = robust_read_table_bytes(perf_raw, perf_file.name)
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

    # (Fix) Projektion: immer t-SNE
    proj_method = "t-SNE"

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
        help=("Legt fest, in wie viele Gruppen (Cluster) die Punkte bei der K-Means-Methode unterteilt werden.")
    )

    # Darstellungsmethode (Abstand) – Cosinus (schnell) = Unit-Norm + Euclid
    metric_label = st.sidebar.selectbox(
        "Darstellungsmethode (Abstand)",
        ["Cosinus (schnell)", "Euklidisch"],
        index=0,
        help=("Legt fest, wie die Ähnlichkeit zwischen Seiten gemessen wird.\n\n"
              "• Cosinus (schnell): meist die beste Wahl für Texte.\n"
              "• Euklidisch: klassisches Maß für Abstände.")
    )
    tsne_metric = "euclidean"  # für t-SNE immer euclid (Speed)
    use_cosine_equivalent = metric_label.startswith("Cosinus")

    # Bubblegröße nach – dynamisch; Auto-Vorauswahl: Keine Skalierung
    size_options = ["Keine Skalierung"]
    if perf_metric_candidates:
        size_options += perf_metric_candidates

    size_by = st.sidebar.selectbox(
        "Bubblegröße nach",
        size_options,
        index=0,
        help=("Welche Werte aus welcher Spalte der Performance-/Metrik-Datei bestimmt die Größe der Bubbles?")
    )

    # Skalierung + Hilfetext (NEU)
    size_method = st.sidebar.radio(
        "Skalierung",
        ["Logarithmisch (log1p)", "Linear (Min–Max)"],
        index=0,
        help=("Bestimmt, wie aus der gewählten KPI die Punktdurchmesser berechnet werden.\n\n"
              "• *Logarithmisch (log1p)*: komprimiert Ausreißer – sinnvoll bei stark schiefen Verteilungen.\n"
              "• *Linear (Min–Max)*: skaliert proportional zwischen Minimum und Maximum.")
    )

    # Min-/Max-Größe + Perzentil-Grenzen (NEU: Hilfetexte)
    size_min = st.sidebar.slider("Min-Größe (px)", 1, 12, 2,
                                 help="Legt fest, wie klein die Bubbles im Diagramm mindestens dargestellt werden.")
    size_max = st.sidebar.slider("Max-Größe (px)", 6, 40, 10,
                                 help="Legt fest, wie groß die Bubbles im Diagramm maximal dargestellt werden.")
    clip_low = st.sidebar.slider("Perzentil-Grenze unten (%)", 0, 20, 1,
                                 help="Werte unterhalb dieses Perzentils werden bei der Skalierung abgeschnitten (robust gegen Ausreißer unten). Wirkt sich nur auf die Darstellung der Bubblegrößen aus.")
    clip_high = st.sidebar.slider("Perzentil-Grenze oben (%)", 80, 100, 95,
                                  help="Werte oberhalb dieses Perzentils werden abgeschnitten (robust gegen Ausreißer oben). Wirkt sich nur auf die Darstellung der Bubblegrößen aus.")

    # Centroid-Optionen (NEU: Hilfetext)
    show_centroid = st.sidebar.checkbox("Centroid markieren", value=False,
                                        help="Zeigt den thematischen Mittelpunkt an, indem der Durchschnitt aller Embeddings berechnet und als Centroid eingefügt wird.")
    with st.sidebar.expander("Erweitert: Centroid", expanded=False):
        centroid_mode = st.radio("Centroid-Modus", ["Auto (empfohlen)", "Standard", "Unit-Norm"], index=0,
                                 help=("Steuert, ob der Centroid im Normalraum (Standard) oder auf normierten Vektoren (Unit-Norm) berechnet wird.\n"
                                       "‘Auto’ wählt basierend auf Norm-Statistiken sinnvoll.") )
    centroid_size = st.sidebar.slider("Centroid-Sterngröße (px)", 10, 40, 22, 1, disabled=not show_centroid)

    bg_color = st.sidebar.color_picker("Hintergrundfarbe für Bubble-Chart", value="#FFFFFF")

    # Exporte
    st.sidebar.markdown("**Weitere Exportmöglichkeiten**")

    export_csv = st.sidebar.checkbox(
        "Semantisch ähnliche URLs exportieren", value=False,
        help="CSV mit ähnlichen URL-Paaren (Cosinus-Ähnlichkeit)."
    )
    # Exportmodus: Schwellenwert (exakt) oder FAISS Top-N (schnell)
    export_modes = ["Schwellenwert (exakt)"]
    if FAISS_OK:
        export_modes.append("Top-N (FAISS, schnell)")
    export_mode = st.sidebar.radio("Exportmodus", export_modes, index=0, disabled=not export_csv,
                                   help=("*Schwellenwert*: alle Paare ≥ gewählter Ähnlichkeit.\n"
                                         "*Top-N (FAISS)*: pro URL die N besten Nachbarn (schnell)."))
    sim_threshold = st.sidebar.slider(
        "Ähnlichkeitsschwelle (Cosinus)",
        min_value=0.00, max_value=1.00, value=0.00, step=0.01,
        help=("Es weren URL-Paare mit Cosinus-Ähnlichkeit ≥ Schwellenwert exportiert."),
        disabled=not export_csv
    )
    top_n = None
    if export_csv and (export_mode.startswith("Top-N")):
        top_n = st.sidebar.slider("Top-N pro URL (FAISS)", 1, 50, 5, 1,
                                  help="Wie viele beste Nachbarn pro URL werden berücksichtigt?")

    export_lowrel_csv = st.sidebar.checkbox(
        "Low-Relevance-URLs exportieren", value=False,
        help=("Seiten mit Cosinus-Ähnlichkeit zum Centroid unterhalb der Schwelle.")
    )
    lowrel_threshold = st.sidebar.slider(
        "Schwelle (Centroid-Cosinus)",
        min_value=0.00, max_value=1.00, value=0.40, step=0.01,
        disabled=not export_lowrel_csv,
        help="URLs mit Ähnlichkeit < Schwelle werden als potenziell themenfern exportiert."
    )

    # (NEU) Hilfetext für „Kein Limit beim Export“
    unlimited_export = st.sidebar.checkbox("Kein Limit für Export", value=False,
                                           help=("Bei sehr vielen URLs können extrem große CSVs entstehen.\n"
                                                 "Wenn deaktiviert, wird der Export auf eine maximale Zeilenzahl begrenzt."))
    if not unlimited_export:
        max_export_rows = st.sidebar.number_input(
            "Max. Zeilen pro Export", min_value=50_000, max_value=5_000_000, step=50_000, value=250_000,
            help="Begrenzt die Größe der Export-Dateien, um Speicher/Browser zu schonen."
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

    def pre_reduce_for_proj(X_in: np.ndarray, d=50, normalize=False):
        Xp = l2_normalize_rows(X_in) if normalize else X_in
        if Xp.shape[1] > d:
            Xp = PCA(n_components=d, svd_solver="randomized", random_state=42).fit_transform(Xp)
        return Xp.astype(np.float32, copy=False)

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

        # Embedding-Matrix
        X = embedding_matrix  # float32

        # Optionaler Centroid-Punkt in die Projektion aufnehmen
        use_centroid_flag = bool(show_centroid)
        if use_centroid_flag:
            centroid_vec, centroid_mode_eff = compute_centroid(X, centroid_mode)
            X_proj = np.vstack([X, centroid_vec[None, :]]).astype(np.float32, copy=False)
        else:
            centroid_mode_eff = None
            X_proj = X

        # --------- 2D-Projektion: t-SNE (fix) ----------
        X_for_tsne = pre_reduce_for_proj(X_proj, d=50, normalize=use_cosine_equivalent)
        n_samples_tsne = X_for_tsne.shape[0]
        if n_samples_tsne <= 5:
            st.error("Zu wenige Punkte für die 2D-Projektion.")
            st.stop()
        perplexity = max(5, min(30, n_samples_tsne - 1))

        tsne_method = "exact" if n_samples_tsne <= 3000 else "barnes_hut"
        tsne = TSNE(
            n_components=2,
            metric="euclidean",
            method=tsne_method,
            init="pca",
            learning_rate="auto",
            n_iter=750,
            random_state=42,
            perplexity=perplexity,
            square_distances=True,
        )
        tsne_result = tsne.fit_transform(X_for_tsne)
        merged["tsne_x"] = tsne_result[: X.shape[0], 0]
        merged["tsne_y"] = tsne_result[: X.shape[0], 1]
        n_points_2d = int(tsne_result.shape[0])
        st.caption(f"t-SNE Perplexity: {perplexity} · Punkte im 2D: {n_points_2d}")

        # Cluster
        method = cluster_method
        segment_col = segment_col_global

        if method == "K-Means":
            Xk = l2_normalize_rows(X)
            n_samples = Xk.shape[0]
            if n_samples < 1500:
                km = KMeans(n_clusters=cluster_k, n_init="auto", random_state=42)
            else:
                bs = max(512, min(4096, int(n_samples * 0.01)))
                km = MiniBatchKMeans(
                    n_clusters=cluster_k,
                    batch_size=bs,
                    n_init="auto",
                    max_iter=100,
                    random_state=42,
                    reassignment_ratio=0.01,
                )
            merged["Cluster"] = km.fit_predict(Xk).astype(str)
        elif method == "DBSCAN (Cosinus)":
            cos_dist = cosine_distances(l2_normalize_rows(X))
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
            merged["__marker_px"] = merged["__marker_size"].clip(lower=1)
        else:
            merged["__marker_px"] = max(1, int(size_min))


        # Cache
        st.session_state["merged_cached"] = merged
        st.session_state["scaled_cached"] = scaled
        st.session_state["hover_cols_cached"] = _build_hover_cols(merged, metric_col)
        st.session_state["plot_title_cached"] = (
            "🔍 t-SNE der Seiten-Embeddings" + (" (mit Skalierung)" if scaled else "")
        )
        st.session_state["bg_color_cached"] = bg_color
        st.session_state["highlight_px_cached"] = max(int(size_min) + 6, 8)
        st.session_state["url_col_cached"] = url_col
        st.session_state["centroid_in_proj"] = use_centroid_flag
        st.session_state["centroid_mode_eff"] = centroid_mode_eff
        if use_centroid_flag:
            cx, cy = float(tsne_result[-1, 0]), float(tsne_result[-1, 1])
            st.session_state["centroid_xy"] = (cx, cy)
        else:
            st.session_state["centroid_xy"] = None

    def render_plot_from_cache(q: str):
        """Zeichnet den Plot aus dem Cache neu; bei Suche: Rest grau, Treffer farbig."""
        merged = st.session_state.get("merged_cached")
        if merged is None:
            st.info("Bitte zuerst Einstellungen wählen und auf **Let's Go / Refresh** klicken.")
            return

        scaled = st.session_state.get("scaled_cached", False)
        hover_cols = st.session_state.get("hover_cols_cached", {url_col: True, "Cluster": True})
        title = st.session_state.get("plot_title_cached", "🔍 2D-Projektion der Seiten-Embeddings")
        bg = st.session_state.get("bg_color_cached", "#FFFFFF")
        url_c = st.session_state.get("url_col_cached", url_col)
        highlight_px = st.session_state.get("highlight_px_cached", 10)
        centroid_xy = st.session_state.get("centroid_xy", None)
        centroid_mode_eff = st.session_state.get("centroid_mode_eff", None)

        q = (q or "").strip().lower()

        # Für dynamischen Legendentitel, insbesondere bei Segments
        method = cluster_method
        segment_col = segment_col_global
        # Aussagekräftiger Legenden-Titel je Methode
        if method == "Segments" and segment_col:
            legend_title_text = segment_col  # z.B. "Segmente" oder Spaltenname
        elif method == "K-Means":
            legend_title_text = "Cluster (K-Means)"
        elif method == "DBSCAN (Cosinus)":
            legend_title_text = "Cluster (DBSCAN)"
        else:
            legend_title_text = "Cluster"


        if q:
            # --- Suchmodus: Basisschicht grau, nur Treffer farbig ---
            fig = go.Figure()

            fig.add_trace(go.Scattergl(
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

            mask = merged[url_c].astype(str).str.lower().str.contains(q, na=False)
            if mask.any():
                hi = merged[mask]
                hover_texts = []
                for _, row in hi.iterrows():
                    extras = []
                    if "Cluster" in row:
                        extras.append(f"{legend_title_text}: {row['Cluster']}")
                    hover_texts.append(f"{row[url_c]}<br>" + ("<br>".join(extras) if extras else ""))
                fig.add_trace(go.Scattergl(
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
            # --- Normalmodus: farbige Cluster ---
            merged["Cluster"] = merged["Cluster"].astype(str)
            cluster_labels = merged["Cluster"].unique().tolist()

            def _legend_sort_key(lbl):
                try:
                    return (0, float(lbl))
                except Exception:
                    return (1, str(lbl).lower())

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
                render_mode="webgl",
            )

            color_by_name = {}
            for tr in fig.data:
                mask = (merged["Cluster"] == tr.name)
                sizes = merged.loc[mask, "__marker_px"].tolist()
                tr.marker.update(size=sizes, sizemode="diameter", opacity=0.55, line=dict(width=0.5, color="white"))
                cval = tr.marker.color
                if isinstance(cval, (list, np.ndarray)) and len(cval) > 0:
                    cval = cval[0]
                color_by_name[tr.name] = cval
                tr.hoverlabel = dict(bgcolor=cval, font_color="white", bordercolor="black")
                tr.legendgroup = tr.name
                tr.showlegend = False

            for name in cluster_order:
                fig.add_trace(go.Scattergl(
                    x=[None], y=[None],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=True,
                    marker=dict(size=12, color=color_by_name.get(name, None), line=dict(width=0.5, color="white")),
                    hoverinfo="skip"
                ))

        if centroid_xy is not None:
            cx, cy = centroid_xy
            fig.add_trace(go.Scattergl(
                x=[cx], y=[cy], mode="markers", name="Centroid",
                marker=dict(symbol="star", size=int(centroid_size), color="red", line=dict(width=1, color="black")),
                hoverlabel=dict(bgcolor="red", font_color="white", bordercolor="black")
            ))

        # (NEU) Keine Gitternetzlinien
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        # Keine Gitternetzlinien + Achsen-Beschriftungen
        fig.update_xaxes(showgrid=False, zeroline=False, title_text="t-SNE X")
        fig.update_yaxes(showgrid=False, zeroline=False, title_text="t-SNE Y")

        # Layout + gut lesbare Legende (schwarze Schrift)
        fig.update_layout(
            title=title,
            plot_bgcolor=bg,
            paper_bgcolor=bg,
            height=750,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
            dragmode="zoom",
            hovermode="closest",
            legend=dict(
                title=dict(text=legend_title_text, font=dict(color="black")),
                font=dict(color="black"),                 # <-- Legenden-Schrift schwarz
                bgcolor="rgba(255,255,255,0.9)",          # <-- milchiger weißer Hintergrund (bessere Lesbarkeit)
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                itemsizing="constant",
            ),
        )


        st.subheader("📈 Visualisierung")
        if st.session_state.get("centroid_mode_eff"):
            st.caption(f"Centroid-Modus aktiv: {st.session_state.get('centroid_mode_eff')}")
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
        with st.spinner("Berechne Projektion & erstelle Plot…"):
            build_data_and_cache()
            render_plot_from_cache(search_q)
    else:
        render_plot_from_cache(search_q)

    # =============================
    # Exporte (unabhängig von Suche!)
    # =============================

    # ----- Blockweiser Similarity-Export (exakt, Schwellenwert) -----
    def similar_pairs_threshold_blocked(
        X: np.ndarray, urls: list, thr: float, max_rows: int | None = None, block: int = 2048
    ):
        Xn = l2_normalize_rows(X.astype(np.float32, copy=False))
        n = Xn.shape[0]
        pairs = []
        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            A = Xn[i0:i1]                  # (b, d)
            S = A @ Xn.T                   # (b, n)  # Dot-Product = Cosinus-Similarity
            for ii in range(i0, i1):
                row = S[ii - i0]
                js = np.where(row[ii+1:] >= thr)[0]
                if js.size:
                    base = ii + 1
                    scores = row[ii+1:][js]
                    js = js.astype(int)
                    for k in range(js.size):
                        j = base + js[k]
                        s = float(scores[k])
                        pairs.append({
                            "URL_A": urls[ii],
                            "URL_B": urls[j],
                            "Cosinus_Ähnlichkeit": s,
                            "Match-Typ": "Similarity (block-dot)"
                        })
                if max_rows and len(pairs) >= max_rows:
                    return pairs[:max_rows]
        return pairs

    # ----- FAISS Top-N Similarity-Export (schnell) -----
    def faiss_topn_cosine(X: np.ndarray, urls: list, k: int = 5, thr: float = 0.0, max_rows: int | None = None):
        Xn = l2_normalize_rows(X.astype('float32', copy=False))
        d = Xn.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner Product == Cosinus bei unit-norm
        index.add(Xn)
        D, I = index.search(Xn, min(k + 1, Xn.shape[0]))  # +1 wegen Selbsttreffer
        pairs = []
        n = Xn.shape[0]
        for i in range(n):
            rank = 0
            for j_idx, s in zip(I[i], D[i]):
                if j_idx == i:
                    continue  # Selbsttreffer
                if s < thr:
                    continue
                # nur obere Dreieckshälfte (i<j) exportieren -> keine Duplikate
                if i < j_idx:
                    pairs.append({
                        "URL_A": urls[i],
                        "URL_B": urls[j_idx],
                        "Cosinus_Ähnlichkeit": float(s),
                        "Match-Typ": f"Similarity (FAISS) Rank {rank+1}"
                    })
                    if max_rows and len(pairs) >= max_rows:
                        return pairs
                rank += 1
        # Nach Score absteigend sortieren
        pairs.sort(key=lambda r: r["Cosinus_Ähnlichkeit"], reverse=True)
        return pairs

    if export_csv:
        merged_cached = st.session_state.get("merged_cached")
        if merged_cached is not None:
            url_list = merged_cached[url_col].astype(str).tolist()
            X_raw = np.stack(merged_cached["embedding_vector"].tolist()).astype("float32", copy=False)
            thr = float(sim_threshold)

            if export_mode.startswith("Top-N") and FAISS_OK:
                with st.spinner("Berechne semantische Ähnlichkeiten (FAISS Top-N)…"):
                    pairs = faiss_topn_cosine(
                        X_raw, url_list, k=int(top_n or 5),
                        thr=thr,
                        max_rows=(None if unlimited_export else int(max_export_rows))
                    )
            else:
                with st.spinner("Berechne semantische Ähnlichkeiten (blockweise)…"):
                    # Warnung bei erwartbar großen Paarzahlen (grobe Heuristik)
                    n = len(url_list)
                    est_pairs = n * (n - 1) // 2
                    if unlimited_export and est_pairs > 2_000_000 and thr <= 0.2:
                        st.warning(f"Viele Paare erwartet (~{est_pairs:,}). "
                                   f"Niedrige Schwelle + kein Limit kann sehr große CSVs erzeugen.")
                    pairs = similar_pairs_threshold_blocked(
                        X_raw, url_list, thr=thr, max_rows=(None if unlimited_export else int(max_export_rows))
                    )

            if not pairs:
                st.warning("Keine Paare über der eingestellten Ähnlichkeitsschwelle gefunden.")
            else:
                sim_df = pd.DataFrame(pairs)
                sim_df = sim_df.sort_values("Cosinus_Ähnlichkeit", ascending=False, kind="stable")
                csv_bytes = sim_df.to_csv(index=False).encode("utf-8-sig")
                label = (
                    f"📥 Cosinus-Ähnlichkeiten als CSV (Top-{top_n}, ≥ {thr:.2f}, FAISS)"
                    if export_mode.startswith("Top-N") and FAISS_OK
                    else f"📥 Cosinus-Ähnlichkeiten als CSV (≥ {thr:.2f}, block-dot)"
                )
                st.download_button(
                    label=label,
                    data=csv_bytes,
                    file_name="cosine_similarity_pairs.csv",
                    mime="text/csv",
                )
        else:
            st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")

    if export_lowrel_csv:
        merged_cached = st.session_state.get("merged_cached")
        if merged_cached is not None:
            with st.spinner("Berechne Centroid-Ähnlichkeiten pro URL…"):
                X = np.stack(merged_cached["embedding_vector"].tolist()).astype("float32", copy=False)
                centroid_vec, centroid_mode_eff_export = compute_centroid(X, centroid_mode)
                Xn = l2_normalize_rows(X)
                cn = np.linalg.norm(centroid_vec)
                if cn == 0:
                    centroid_sim = np.zeros(Xn.shape[0], dtype=np.float32)
                else:
                    c_unit = centroid_vec / cn
                    centroid_sim = (Xn @ c_unit.astype(np.float32)).ravel()

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

# Globaler Fänger – zeigt echten Traceback statt generischem „Oh no“
except Exception as e:
    st.exception(e)
