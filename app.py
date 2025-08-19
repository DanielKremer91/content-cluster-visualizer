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

# --- scikit-learn Lazy-Import + Guard ---
SKLEARN_OK = True
_import_err = None
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
except Exception as e:
    SKLEARN_OK = False
    _import_err = e

# --- UMAP Lazy-Import (mit klarem Fehler in Sidebar) ---
UMAP_OK = True
_umap_err = None
try:
    # Manche Envs brauchen den Klassenzugriff separat
    import umap  # stellt das Paket sicher
    from umap import UMAP  # direkte Klasse
except Exception as e:
    UMAP_OK = False
    _umap_err = e

# --- FAISS Lazy-Import (für Export) ---
FAISS_OK = True
_faiss_err = None
try:
    import faiss
except Exception as e:
    FAISS_OK = False
    _faiss_err = e

import plotly.express as px
import plotly.graph_objects as go

# =============================
# Page setup & Branding
# =============================
st.set_page_config(page_title="ONE Semantic Content-Map", layout="wide")

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

# Sidebar: Versionen & Status
VER_PY = platform.python_version()
try:
    import sklearn as _skl
    VER_SKL = _skl.__version__
except Exception:
    VER_SKL = "n/a"
VER_NP = np.__version__
VER_PD = pd.__version__

st.sidebar.header("Systemstatus")
st.sidebar.write(f"🔧 Python {VER_PY}")
st.sidebar.write(f"🔢 NumPy {VER_NP} · pandas {VER_PD} · scikit-learn {VER_SKL}")

# UMAP/FAISS Status mit Fehlermeldung (wenn vorhanden)
if UMAP_OK:
    st.sidebar.success("UMAP: verfügbar")
else:
    st.sidebar.error(f"UMAP: nicht verfügbar\n\n{_umap_err}")

if FAISS_OK:
    st.sidebar.success("FAISS: verfügbar (Export nutzt FAISS)")
else:
    st.sidebar.warning(f"FAISS: nicht verfügbar – Fallback auf blockweise Sklearn-Methode\n\n{_faiss_err}")

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

Dieses Tool macht **thematische Strukturen einer Domain sichtbar**, hilft **Off-Topic-Content** zu erkennen und **Audit-Listen zu exportieren**.

### 🔄 Input
- **Pflicht:** *Embedding-Datei* (CSV/Excel) mit **URLs** und **Embedding-Spalte**  
  ↳ Optional: *Segment-Spalte* (Verzeichnisse/URL-Gruppen)

- **Optional:** *Performance-Datei* (GSC/SISTRIX/Ahrefs) – numerische Spalten können die **Bubblegröße** steuern.
""")

    st.markdown("""
### ⚙️ Wie funktioniert’s?
- **Projektions-Optionen:**  
  **t-SNE** mit **PCA-Vorschaltstufe** (50D *oder* 100D) für Stabilität/Geschwindigkeit.  
  **UMAP (Alternative):** Für große n meist **deutlich schneller**, bessere **globale Struktur**, stabil. (Python 3.11 wird unterstützt.)
- **Clustering:** K-Means, DBSCAN (Cosinus) oder vorhandene Segments-Spalte.
- **Darstellung (NEU):** **Serverseitiges Downsampling** nur für die Visualisierung (MiniBatchKMeans) auf **20k Repräsentanten**.  
  *Empfehlung:*  
  – „Alle Punkte“ bis ~**50k** URLs (UMAP packt oft mehr).  
  – **Downsampling** ab **50–100k** URLs oder mit t-SNE – schnelleres Rendering, weniger RAM.
- **Exports:**  
  – **Ähnliche Paare** (Cosinus ≥ Schwelle) **per FAISS range_search** (Turbo), Fallback Sklearn.  
  – **Low-Relevance** zum Centroid (Cosinus < Schwelle).
""")

# =============================
# Utilities
# =============================
def _cleanup_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def _read_csv_bytes(bytes_data, **kwargs):
    return pd.read_csv(BytesIO(bytes_data), dtype=str, low_memory=False, **kwargs)

@st.cache_data(show_spinner=False)
def robust_read_table_bytes(raw: bytes, name: str):
    name = (name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(BytesIO(raw))
            return _cleanup_headers(df)
        except Exception:
            pass
    try:
        df = _read_csv_bytes(raw, sep="\t", encoding="UTF-16")
        if df.shape[1] > 0:
            return _cleanup_headers(df)
    except Exception:
        pass
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
    span = max(1e-9, float(size_max) - float(size_min))  # robust
    diam = float(size_min) + s_norm * span
    return pd.Series(diam)

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
    return X.mean(axis=0), "Standard"

URL_CANDIDATES_BASE = [
    "URL", "Page", "Pages",
    "Adresse", "Address",
    "Seite", "Seiten",
    "URLs",
    "URL-Adresse", "URL Adresse",
]
URL_CANDIDATES_GSC_EXTRA = ["Landing Page", "Seiten-URL", "Seiten URL"]

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

    try:
        emb_raw = emb_file.getvalue()
        df = robust_read_table_bytes(emb_raw, emb_file.name)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Columns detected (Embedding-Datei): {list(df.columns)}")

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

    @st.cache_data(show_spinner=False)
    def parse_and_normalize_embeddings(df_in: pd.DataFrame, col: str):
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
    # Interaktive URL-Suche
    # =============================
    search_q = st.text_input(
        "🔍 URL-Suche (Teilstring)",
        value="",
        help="Markiert Treffer im Plot. Beeinflusst weder Berechnungen noch Exporte."
    )

    # Optional: Performance-Datei
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
    # Sidebar Controls
    # =============================
    st.sidebar.header("Einstellungen")

    # Projektionsmethode (UMAP nur, wenn verfügbar)
    proj_choices = ["t-SNE"]
    if UMAP_OK:
        proj_choices.append("UMAP")
    else:
        proj_choices.append("UMAP (nicht verfügbar)")
    proj_method = st.sidebar.selectbox(
        "Projektionsmethode",
        proj_choices,
        help=("Wähle die 2D-Projektion.\n\n"
              "• t-SNE: Detaillierte lokale Nachbarschaften; mit PCA (50/100D) stabiler & schneller.\n"
              "• UMAP: Für größere n meist deutlich schneller, stabilere globale Struktur.")
    )

    # t-SNE: PCA-Dimension
    with st.sidebar.expander("Erweitert: t-SNE (PCA-Vorschaltstufe)", expanded=False):
        pca_dims = st.radio(
            "PCA-Dimension vor t-SNE",
            [50, 100],
            index=0,
            help=("Reduziert Embeddings per PCA, bevor t-SNE läuft. 50D reicht meist; 100D erhält minimale Zusatzdetails.")
        )

    # Darstellung: Downsampling
    render_mode = st.sidebar.selectbox(
        "Darstellung: Punktmenge",
        ["Alle Punkte rendern", "Downsampling auf 20k Repräsentanten (MiniBatchKMeans)"],
        help=("Steuert die Punktmenge NUR für die Visualisierung. Exporte nutzen immer alle Daten.\n\n"
              "Empfehlung:\n"
              "• „Alle Punkte“ bis ca. 50k URLs (UMAP schafft häufig mehr).\n"
              "• „Downsampling 20k“ ab ~50–100k URLs oder wenn t-SNE genutzt wird – schnelleres Rendering.")
    )
    target_render_n = 20000

    cluster_options = ["K-Means", "DBSCAN (Cosinus)"]
    segment_col_global = segment_col_global
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
        help=("Legt die Anzahl der Cluster bei K-Means fest.")
    )

    metric_label = st.sidebar.selectbox(
        "Darstellungsmethode (Abstand)",
        ["Euklidisch", "Cosinus (schnell)"],
        help=("Cosinus (schnell): L2-Norm + euklidische Projektion (äquivalent & performant).")
    )
    use_cosine_equivalent = metric_label.startswith("Cosinus")

    size_options = ["Keine Skalierung"] + perf_metric_candidates if perf_metric_candidates else ["Keine Skalierung"]
    size_by = st.sidebar.selectbox(
        "Bubblegröße nach",
        size_options,
        index=0,
        help="Wähle eine numerische Spalte aus der Performance-Datei für die Bubblegröße."
    )
    size_method = st.sidebar.radio(
        "Skalierung",
        ["Logarithmisch (log1p)", "Linear (Min–Max)"],
        index=0
    )
    size_min = st.sidebar.slider("Min-Größe (px)", 1, 12, 2)
    size_max = st.sidebar.slider("Max-Größe (px)", 6, 40, 10)
    clip_low = st.sidebar.slider("Perzentil-Grenze unten (%)", 0, 20, 1)
    clip_high = st.sidebar.slider("Perzentil-Grenze oben (%)", 80, 100, 95)

    show_centroid = st.sidebar.checkbox("Centroid markieren", value=False)
    with st.sidebar.expander("Erweitert: Centroid", expanded=False):
        centroid_mode = st.radio("Centroid-Modus", ["Auto (empfohlen)", "Standard", "Unit-Norm"], index=0)
    centroid_size = st.sidebar.slider("Centroid-Sterngröße (px)", 10, 40, 22, 1, disabled=not show_centroid)

    if perf_df is not None and (size_by != "Keine Skalierung"):
        bubble_scale = st.sidebar.slider("Bubble-Scale (global)", 0.20, 2.00, 1.00, 0.05)
    else:
        bubble_scale = 1.0

    bg_color = st.sidebar.color_picker("Hintergrundfarbe für Bubble-Chart", value="#FFFFFF")

    st.sidebar.markdown("**Weitere Exportmöglichkeiten**")
    export_csv = st.sidebar.checkbox(
        "Semantisch ähnliche URLs exportieren", value=False,
        help="Export ähnlicher URL-Paare (Cosinus ≥ Schwelle) – nutzt FAISS range_search (Fallback Sklearn)."
    )
    sim_threshold = st.sidebar.slider("Ähnlichkeitsschwelle (Cosinus)", 0.00, 1.00, 0.00, 0.01, disabled=not export_csv)

    export_lowrel_csv = st.sidebar.checkbox(
        "Low-Relevance-URLs exportieren", value=False,
        help="Exportiert URLs, deren Cosinus-Ähnlichkeit zum Centroid unterhalb der Schwelle liegt."
    )
    lowrel_threshold = st.sidebar.slider("Schwelle zum Centroid (Cosinus)", 0.00, 1.00, 0.40, 0.01, disabled=not export_lowrel_csv)

    unlimited_export = st.sidebar.checkbox("Kein Limit für Export", value=False)
    if not unlimited_export:
        max_export_rows = st.sidebar.number_input("Max. Zeilen pro Export", 50_000, 5_000_000, 250_000, 50_000)
    else:
        max_export_rows = None

    recalc = st.sidebar.button("Let's Go / Refresh", type="primary")

    # =============================
    # Build data & cache
    # =============================
    def _build_hover_cols(merged, metric_col):
        h = {url_col: True, "Cluster": True}
        for extra in {metric_col}:
            if extra and extra in merged.columns:
                h[extra] = True
        return h

    def pick_representative_indices(X: np.ndarray, k: int, random_state: int = 42):
        n = X.shape[0]
        if n <= k:
            return np.arange(n, dtype=int)
        Xn = l2_normalize_rows(X)
        mbk = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096, n_init="auto")
        labels = mbk.fit_predict(Xn)
        centers = mbk.cluster_centers_
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Xn)
        _, idxs = nn.kneighbors(centers, return_distance=True)
        idxs = np.unique(idxs.flatten().astype(int))
        if idxs.size < k:
            remaining = np.setdiff1d(np.arange(n, dtype=int), idxs, assume_unique=False)
            rng = np.random.default_rng(random_state)
            fill = rng.choice(remaining, size=(k - idxs.size), replace=False)
            idxs = np.concatenate([idxs, fill])
        return idxs

    def build_data_and_cache():
        merged_all = df_valid.copy()
        X_all = embedding_matrix

        if isinstance(perf_df, pd.DataFrame) and perf_url_col:
            merged_all["__join"] = merged_all[url_col].apply(normalize_url)
            perf_local = perf_df.copy()
            perf_local["__join"] = perf_local[perf_url_col].apply(normalize_url)
            keep_cols = ["__join"]
            perf_keep = perf_local[keep_cols + list(set(perf_metric_candidates))].drop_duplicates("__join") \
                if perf_metric_candidates else perf_local[keep_cols].drop_duplicates("__join")
            merged_all = merged_all.merge(perf_keep, on="__join", how="left")
            merged_all.drop(columns=["__join"], inplace=True, errors="ignore")

        if render_mode.startswith("Downsampling"):
            k = min(target_render_n, X_all.shape[0])
            idx_render = pick_representative_indices(X_all, k)
        else:
            idx_render = np.arange(X_all.shape[0], dtype=int)

        merged = merged_all.iloc[idx_render].reset_index(drop=True)
        X = X_all[idx_render]

        use_centroid_flag = bool(show_centroid)
        centroid_mode_eff = None
        if use_centroid_flag:
            centroid_vec, centroid_mode_eff = compute_centroid(X_all, centroid_mode)

        use_cosine = use_cosine_equivalent

        # Projektion
        if proj_method.startswith("t-SNE"):
            X_for = l2_normalize_rows(X) if use_cosine else X
            d_pca = int(min(pca_dims, X_for.shape[1]))
            pca = PCA(n_components=d_pca, svd_solver="randomized", random_state=42)
            X_reduced = pca.fit_transform(X_for)

            if use_centroid_flag:
                c_vec = centroid_vec.copy().astype(np.float32)
                c_vec = l2_normalize_rows(c_vec[None, :]) if use_cosine else c_vec[None, :]
                c_red = pca.transform(c_vec)
                X_tsne_input = np.vstack([X_reduced, c_red])
            else:
                X_tsne_input = X_reduced

            n_tsne = X_tsne_input.shape[0]
            perplexity = max(5, min(50, n_tsne // 3, n_tsne - 1))

            tsne = TSNE(
                n_components=2,
                metric="euclidean",
                method="barnes_hut",
                init="pca",
                learning_rate="auto",
                n_iter=750,
                random_state=42,
                perplexity=perplexity
            )
            tsne_result = tsne.fit_transform(X_tsne_input)
            merged["tsne_x"] = tsne_result[: len(X_reduced), 0]
            merged["tsne_y"] = tsne_result[: len(X_reduced), 1]
            if use_centroid_flag:
                st.session_state["centroid_xy"] = (tsne_result[len(X_reduced), 0], tsne_result[len(X_reduced), 1])
            st.session_state["proj_title"] = f"🔍 2D-Projektion (t-SNE; PCA→{d_pca}D)"
            st.session_state["perplexity"] = perplexity
        else:
            if not UMAP_OK:
                st.error(f"UMAP ist nicht verfügbar: {_umap_err}")
                st.stop()
            metric = "cosine" if use_cosine else "euclidean"
            model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric=metric, random_state=42)
            X_for = l2_normalize_rows(X) if use_cosine else X
            if use_centroid_flag:
                c_vec = centroid_vec.copy().astype(np.float32)
                c_vec = l2_normalize_rows(c_vec[None, :]) if use_cosine else c_vec[None, :]
                X_umap_input = np.vstack([X_for, c_vec])
                umap_result = model.fit_transform(X_umap_input)
                merged["tsne_x"] = umap_result[: len(X_for), 0]
                merged["tsne_y"] = umap_result[: len(X_for), 1]
                st.session_state["centroid_xy"] = (umap_result[len(X_for), 0], umap_result[len(X_for), 1])
            else:
                umap_result = model.fit_transform(X_for)
                merged["tsne_x"] = umap_result[:, 0]
                merged["tsne_y"] = umap_result[:, 1]
            st.session_state["proj_title"] = "🔍 2D-Projektion (UMAP)"
            st.session_state["perplexity"] = None

        # Cluster
        method = cluster_method
        segment_col = segment_col_global
        if method == "K-Means":
            kmeans = KMeans(n_clusters=cluster_k, random_state=42)
            merged["Cluster"] = kmeans.fit_predict(X).astype(str)
        elif method == "DBSCAN (Cosinus)":
            Xn = l2_normalize_rows(X)
            dbscan = DBSCAN(eps=0.3, min_samples=5, metric="cosine")
            merged["Cluster"] = dbscan.fit_predict(Xn).astype(str)
        elif method == "Segments" and segment_col:
            merged["Cluster"] = merged_all.iloc[idx_render][segment_col].fillna("Unbekannt").astype(str).values
        else:
            merged["Cluster"] = "Kein Segment"

        # Bubblegrößen
        scaled = False
        metric_col = size_by if size_by != "Keine Skalierung" else None
        if metric_col and metric_col in merged.columns:
            mth = "log" if size_method.startswith("Log") else "linear"
            merged["__marker_size"] = scale_sizes(
                merged[metric_col], method=mth,
                size_min=size_min, size_max=size_max,
                clip_low=clip_low, clip_high=clip_high,
            )
            scaled = True
        else:
            merged["__marker_size"] = float(size_min)

        merged["__marker_px"] = (merged["__marker_size"] * float(bubble_scale)).clip(lower=1) if scaled \
            else max(1, int(size_min * float(bubble_scale)))

        # Cache
        st.session_state["merged_cached"] = merged
        st.session_state["merged_all"] = merged_all
        st.session_state["X_all"] = X_all
        st.session_state["scaled_cached"] = scaled
        st.session_state["hover_cols_cached"] = _build_hover_cols(merged, metric_col)
        st.session_state["plot_title_cached"] = "🔍 2D-Projektion"
        st.session_state["bg_color_cached"] = bg_color
        st.session_state["highlight_px_cached"] = max(int(size_min * float(bubble_scale)) + 6, 8)
        st.session_state["url_col_cached"] = url_col
        st.session_state["centroid_in_proj"] = use_centroid_flag

        if st.session_state.get("perplexity") is not None:
            st.caption(f"t-SNE Perplexity: {st.session_state['perplexity']} · Punkte im Plot: {len(merged)}")
        else:
            st.caption(f"Punkte im Plot: {len(merged)}")

    def render_plot_from_cache(q: str):
        merged = st.session_state.get("merged_cached")
        if merged is None:
            st.info("Bitte zuerst Einstellungen wählen und auf **Let's Go / Refresh** klicken.")
            return

        scaled = st.session_state.get("scaled_cached", False)
        hover_cols = st.session_state.get("hover_cols_cached", {st.session_state.get("url_col_cached", "URL"): True, "Cluster": True})
        title = st.session_state.get("proj_title", st.session_state.get("plot_title_cached", "🔍 2D-Projektion"))
        bg = st.session_state.get("bg_color_cached", "#FFFFFF")
        url_c = st.session_state.get("url_col_cached", "URL")
        centroid_xy = st.session_state.get("centroid_xy", None)

        q = (q or "").strip().lower()

        if q:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=merged["tsne_x"], y=merged["tsne_y"], mode="markers", name="Alle",
                marker=dict(size=merged["__marker_px"].tolist(), color="lightgray", opacity=0.35, line=dict(width=0.5, color="white")),
                hoverinfo="skip", showlegend=False
            ))
            mask = merged[url_c].astype(str).str.lower().str.contains(q, na=False)
            if mask.any():
                hi = merged[mask]
                hover_texts = []
                for _, row in hi.iterrows():
                    extras = []
                    if "Cluster" in row:
                        extras.append(f"Cluster: {row['Cluster']}")
                    hover_texts.append(f"{row[url_c]}<br>" + ("<br>".join(extras) if extras else ""))
                fig.add_trace(go.Scattergl(
                    x=hi["tsne_x"], y=hi["tsne_y"], mode="markers", name="Treffer",
                    marker=dict(size=hi["__marker_px"].tolist(), color="orange", line=dict(width=2, color="black")),
                    hovertext=hover_texts, hoverinfo="text",
                    hoverlabel=dict(bgcolor="orange", font_color="black", bordercolor="black"),
                    showlegend=False
                ))
                st.caption(f"✨ {int(mask.sum())} Treffer für „{q}“")
        else:
            merged["Cluster"] = merged["Cluster"].astype(str)
            cluster_labels = merged["Cluster"].unique().tolist()

            def _legend_sort_key(lbl):
                try:
                    return (0, float(lbl))
                except Exception:
                    return (1, str(lbl).lower())

            cluster_order = [lbl for lbl in sorted(cluster_labels, key=_legend_sort_key)]

            fig = px.scatter(
                merged, x="tsne_x", y="tsne_y",
                color="Cluster", category_orders={"Cluster": cluster_order},
                hover_data=hover_cols, template="plotly_white",
                title=title, render_mode="webgl",
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
                    x=[None], y=[None], mode="markers", name=name, legendgroup=name, showlegend=True,
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

        fig.update_layout(
            title=title, plot_bgcolor=bg, paper_bgcolor=bg,
            height=750, margin=dict(l=10, r=10, t=50, b=10),
            legend_title="Cluster", showlegend=True,
            dragmode="zoom", hovermode="closest",
            legend=dict(itemsizing="constant")
        )

        st.subheader("📈 Visualisierung")
        st.plotly_chart(fig, use_container_width=True)

        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="📥 Interaktive HTML-Datei herunterladen",
            data=html_bytes,
            file_name="projection_plot.html",
            mime="text/html",
        )

    # =============================
    # Run
    # =============================
    if recalc:
        with st.spinner("Berechne Projektion & erstelle Plot…"):
            build_data_and_cache()
            render_plot_from_cache(search_q)
    else:
        render_plot_from_cache(search_q)

    # =============================
    # Exporte
    # =============================
    def similar_pairs_threshold_blocked(X: np.ndarray, urls: list, thr: float, max_rows: int | None = None, block: int = 2048):
        Xn = l2_normalize_rows(X.astype(np.float32, copy=False))
        n = Xn.shape[0]
        pairs = []
        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            A = Xn[i0:i1]
            S = A @ Xn.T
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
                            "Match-Typ": "Similarity (block-dot Fallback)"
                        })
                if max_rows and len(pairs) >= max_rows:
                    return pairs[:max_rows]
        return pairs

    def faiss_range_search_pairs(X: np.ndarray, urls: list, thr: float, max_rows: int | None = None):
        if not FAISS_OK:
            st.warning(f"FAISS nicht verfügbar ({_faiss_err}). Fallback auf blockweises Dot-Product.")
            return similar_pairs_threshold_blocked(X, urls, thr=thr, max_rows=max_rows)

        X = X.astype("float32", copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype("float32")
        norms[norms == 0] = 1.0
        Xn = X / norms
        Xn[~np.isfinite(Xn)] = 0.0

        d = Xn.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(Xn)

        thr_adj = max(0.0, float(thr) - 1e-7)
        lims, D, I = index.range_search(Xn, thr_adj)

        pairs = []
        n = len(urls)
        for i in range(n):
            start, end = lims[i], lims[i + 1]
            if start == end:
                continue
            js = I[start:end]
            sims = D[start:end]
            for j, s in zip(js, sims):
                j = int(j)
                if j <= i:
                    continue
                pairs.append((i, j, float(s)))

        if not pairs:
            return []

        pairs.sort(key=lambda t: -t[2])
        if max_rows is not None and len(pairs) > max_rows:
            pairs = pairs[:max_rows]

        out = [{
            "URL_A": urls[i],
            "URL_B": urls[j],
            "Cosinus_Ähnlichkeit": s,
            "Match-Typ": "Similarity (FAISS range_search)"
        } for (i, j, s) in pairs]
        return out

    if export_csv:
        merged_all = st.session_state.get("merged_all")
        X_raw_all = st.session_state.get("X_all")
        if merged_all is not None and X_raw_all is not None:
            with st.spinner("Berechne semantische Ähnlichkeiten (FAISS range_search)…"):
                url_list = merged_all[url_col].astype(str).tolist()
                thr = float(sim_threshold)

                n = len(url_list)
                est_pairs = n * (n - 1) // 2
                if (unlimited_export and est_pairs > 2_000_000 and thr <= 0.2):
                    st.warning(f"Viele Paare erwartet (~{est_pairs:,}). "
                               f"Niedrige Schwelle + kein Limit kann sehr große CSVs erzeugen.")

                pairs = faiss_range_search_pairs(
                    X_raw_all, url_list, thr=thr,
                    max_rows=(None if unlimited_export else int(max_export_rows))
                )

                if not pairs:
                    st.warning("Keine Paare über der eingestellten Ähnlichkeitsschwelle gefunden.")
                else:
                    sim_df = pd.DataFrame(pairs).sort_values("Cosinus_Ähnlichkeit", ascending=False, kind="stable")
                    csv_bytes = sim_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"📥 Cosinus-Ähnlichkeiten als CSV (≥ {thr:.2f}, {'FAISS' if FAISS_OK else 'Fallback'})",
                        data=csv_bytes,
                        file_name=f"cosinus_aehnlichkeiten_ge_{thr:.2f}_{'faiss' if FAISS_OK else 'sklearn'}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")

    if export_lowrel_csv:
        merged_all = st.session_state.get("merged_all")
        if merged_all is not None:
            with st.spinner("Berechne Centroid-Ähnlichkeiten pro URL…"):
                X_all = st.session_state.get("X_all")
                centroid_vec, centroid_mode_eff_export = compute_centroid(X_all, centroid_mode)
                Xn = l2_normalize_rows(X_all)
                cn = np.linalg.norm(centroid_vec)
                if cn == 0:
                    centroid_sim = np.zeros(Xn.shape[0], dtype=np.float32)
                else:
                    c_unit = centroid_vec / cn
                    centroid_sim = (Xn @ c_unit.astype(np.float32)).ravel()

                low_thr = float(lowrel_threshold)
                export_df = pd.DataFrame({
                    "URL": merged_all[url_col].astype(str).values,
                    "Cosinus_Ähnlichkeit_zum_Centroid": centroid_sim
                })

                if "Cluster" in merged_all.columns:
                    export_df["Cluster"] = merged_all["Cluster"].astype(str).values

                if size_by != "Keine Skalierung" and size_by in merged_all.columns:
                    export_df[size_by] = merged_all[size_by].values

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

except Exception as e:
    st.exception(e)
