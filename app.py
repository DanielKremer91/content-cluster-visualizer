import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from io import BytesIO
from urllib.parse import urlparse

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import plotly.express as px
import plotly.graph_objects as go  # für graue Basisschicht & präzise Markersteuerung

# Optional: UMAP & FAISS (werden nur genutzt, wenn installiert)
try:
    import umap  # umap-learn
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import faiss  # faiss-cpu
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# =============================
# Page setup & Branding
# =============================
st.set_page_config(page_title="ONE Semantic Content-Map", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Semantic Content-Map")

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
- **2D-Projektion**: *t-SNE* oder *UMAP* projizieren hochdimensionale Embeddings auf 2D, um **Nachbarschaften** und **globale Struktur** sichtbar zu machen.  
  • **t-SNE**: stark für lokale Nachbarschaften.  
  • **UMAP**: schneller, skaliert besser, behält globale Struktur eher bei.
- **Clustering:** *K-Means* (feste k), *DBSCAN* (dichtebasiert, Cosinus-Distanz) oder vorhandene *Segments*-Spalte nutzen.
- **Abstände:** *Euklidisch* misst Luftlinie; *Cosinus* misst **Winkel/Ähnlichkeit** (bei L2-Norm = Skalarprodukt).  
- **Bubble-Größe:** nach beliebiger **numerischer KPI** aus der Performance-Datei darstellbar.
- **Suche:** interaktive **URL-Suche** – Treffer werden farbig markiert, restliche Bubbles ausgegraut.  
- **Centroid:** thematischen **Schwerpunkt** markieren (roter Stern) – wahlweise **robust** (Medoid/Geometric Median).

### 📤 Output (Ergebnisse)
- **Interaktives 2D-Chart** (HTML-Export möglich)  
- **CSV-Exports (optional):**  
  1) **Semantisch ähnliche Paare** (Cosinus-Similarity ≥ Schwellenwert) – *Exakt (sklearn, alle Paare)* oder *Schnell (FAISS top‑k)*  
  2) **Low-Relevance-URLs** (Cosinus-Similarity zum Zentrum < Schwelle), robust auf Wunsch
  3) **Cluster-Qualität**: Intra-/Inter-Cluster-Ähnlichkeiten (Mittel/Median) als Tabelle/CSV
""")

    # Optional: zweite Info-Box
    st.markdown("""
<div style="margin-top: 0.5rem; background:#fff8e6; border:1px solid #ffd28a; border-radius:8px; padding:10px 12px; color:#000;">
  <strong>💡 Komische Ergebnisse?</strong> Oft liegt es an der <strong>Embedding-Erzeugung</strong>. Genauigkeit ist entscheidend – Details siehe LinkedIn-Postings. 
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
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    def _read_csv(bytes_data, **kwargs):
        return pd.read_csv(BytesIO(bytes_data), dtype=str, low_memory=False, **kwargs)

    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(BytesIO(raw))
            return _cleanup_headers(df)
        except Exception:
            pass

    try:
        df = _read_csv(raw, sep="\t", encoding="UTF-16")
        if df.shape[1] > 0:
            return _cleanup_headers(df)
    except Exception:
        pass

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

# --- Centroid-Logik: Auto/Standard/Unit-Norm/Robust ---

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

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-12)
    Xn[~np.isfinite(Xn)] = 0.0
    return Xn

def geometric_median(X: np.ndarray, iters: int = 64, eps: float = 1e-6) -> np.ndarray:
    # Weiszfeld-Algorithmus (robustes Zentrum)
    y = X.mean(axis=0).copy()
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        d = np.where(d < eps, eps, d)
        w = 1.0 / d
        y_new = np.average(X, axis=0, weights=w)
        if np.linalg.norm(y - y_new) < eps:
            break
        y = y_new
    return y

def medoid_vector(X: np.ndarray) -> np.ndarray:
    # Index mit maximaler durchschnittlicher Cosine-Similarity
    Xn = l2_normalize(X)
    sims = Xn @ Xn.T  # (n,n)
    scores = sims.mean(axis=1)
    idx = int(np.argmax(scores))
    return X[idx]

def compute_centroid(X: np.ndarray, mode: str):
    # returns (centroid_vector_1d, effective_mode_str)
    if mode.startswith("Auto (robust)"):
        stats = norm_stats(X)
        # Grobe Heuristik: bei starker Normstreuung oder vielen Ausreißern -> Medoid
        if stats["level"] in ("warn", "high"):
            return medoid_vector(X), "Medoid (Auto)"
        else:
            return X.mean(axis=0), "Standard"
    if mode.startswith("Unit"):
        Xn = l2_normalize(X)
        c = Xn.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        return c, "Unit-Norm"
    if mode.startswith("Medoid"):
        return medoid_vector(X), "Medoid"
    if mode.startswith("Geometric Median"):
        return geometric_median(X), "Geometric Median"
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

# L2-Normalisierung (empfohlen)
use_l2 = st.sidebar.checkbox(
    "Vektoren L2-normalisieren (empfohlen)", value=True,
    help="Stellt sicher, dass Cosinus-Similarity rein richtungsbasiert ist (Skalarprodukt). Stabilisiert Clustering & Ähnlichkeiten."
)

# Projektion (t-SNE/UMAP)
proj_method = st.sidebar.selectbox(
    "2D-Projektion", ["t-SNE", "UMAP" if HAS_UMAP else "UMAP (nicht installiert)"] if True else ["t-SNE"], index=0,
    help=("Wähle die Methode zur 2D-Darstellung: t-SNE (sehr starke lokale Strukturen) oder UMAP (schneller, behält eher globale Struktur)."
          + ("\nHinweis: Für UMAP bitte 'umap-learn' installieren." if not HAS_UMAP else ""))
)

metric_label = st.sidebar.selectbox(
    "Darstellungsmethode (Abstand)",
    ["Euklidisch", "Cosinus"],
    help=("Bestimmt, wie Abstände/Ähnlichkeiten vor der 2D-Projektion gemessen werden.\n\n"
          "Euklidisch: Luftlinie im Embedding-Raum.\n"
          "Cosinus: Winkel/semantische Ähnlichkeit (bei L2-Norm = Skalarprodukt).")
)
tsne_metric = "euclidean" if metric_label == "Euklidisch" else "cosine"

# UMAP-Parameter (nur wenn verfügbar)
if HAS_UMAP and "UMAP" in proj_method:
    umap_n_neighbors = st.sidebar.slider("UMAP: n_neighbors", 5, 100, 15, 1,
                                         help="Größe des lokalen Nachbarschaftsgraphen. Höher = globalere Struktur, niedriger = lokale Details.")
    umap_min_dist = st.sidebar.slider("UMAP: min_dist", 0.0, 0.99, 0.10, 0.01,
                                      help="Wie dicht Punkte zusammenliegen dürfen. Kleiner = kompaktere Cluster.")

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

# Bubblegröße nach – dynamisch
size_options = ["Keine Skalierung"]
if perf_metric_candidates:
    size_options += perf_metric_candidates

size_by = st.sidebar.selectbox(
    "Bubblegröße nach",
    size_options,
    index=0,
    help=("Welche Spalte aus der Performance-/Metrik-Datei bestimmt die Blasengröße? 'Keine Skalierung' = konstant.")
)

size_method = st.sidebar.radio(
    "Skalierung",
    ["Logarithmisch (log1p)", "Linear (Min–Max)"],
    index=0,
    help=("Bestimmt, wie die Blasengrößen berechnet werden. Logarithmisch ist meist robuster.")
)

size_min = st.sidebar.slider("Min-Größe (px)", 1, 12, 2)
size_max = st.sidebar.slider("Max-Größe (px)", 6, 40, 10)
clip_low = st.sidebar.slider("Perzentil-Grenze unten (%)", 0, 20, 1)
clip_high = st.sidebar.slider("Perzentil-Grenze oben (%)", 80, 100, 95)

# Centroid-Optionen
show_centroid = st.sidebar.checkbox(
    "Zentrum (Centroid) markieren", value=False,
    help="Markiert den thematischen Schwerpunkt der analysierten URLs.")
with st.sidebar.expander("Erweitert: Zentrum", expanded=False):
    centroid_mode = st.radio(
        "Zentrums-Modus",
        ["Auto (robust)", "Standard", "Unit-Norm", "Medoid (robust)", "Geometric Median (robust)"],
        index=0,
        help=("Wie das Zentrum berechnet wird.\n"
              "• Auto (robust): wählt je nach Streuung automatisch ein robustes Zentrum (Medoid) oder Standard.\n"
              "• Standard: arithmetischer Mittelwert.\n"
              "• Unit-Norm: zuerst alle Vektoren L2-normalisieren, dann mitteln.\n"
              "• Medoid/Geometric Median: robust gegen Ausreißer (empfohlen bei 'ausfransenden' Themen).")
    )
centroid_size = st.sidebar.slider("Zentrum-Sterngröße (px)", 10, 40, 22, 1, disabled=not show_centroid)

# Bubble-Scale und Hintergrundfarbe
if perf_df is not None and (size_by != "Keine Skalierung"):
    bubble_scale = st.sidebar.slider("Bubble-Scale (global)", 0.20, 2.00, 1.00, 0.05)
else:
    bubble_scale = 1.0

bg_color = st.sidebar.color_picker("Hintergrundfarbe für Bubble-Chart", value="#FFFFFF")

# Weitere Exporte
st.sidebar.markdown("**Weitere Exportmöglichkeiten**")

# Export 1: Paar-Ähnlichkeiten (Cosinus)
export_csv = st.sidebar.checkbox(
    "Semantisch ähnliche URLs exportieren", value=False,
    help="Export semantisch ähnlicher URL-Paare mit einer Cosinus Similarity über dem gewählten Schwellenwert als CSV."
)

export_mode = st.sidebar.radio(
    "Export-Methode",
    ["Exakt (sklearn, alle Paare)", "Schnell (FAISS top-k)"],
    index=0,
    help=("Exakt: berechnet alle Paare (O(N²)).\nSchnell: nutzt FAISS für top-k Nachbarn je URL und filtert dann per Schwelle.")
    , disabled=not export_csv
)

sim_threshold = st.sidebar.slider(
    "Ähnlichkeitsschwelle (Cosinus)",
    min_value=0.00, max_value=1.00, value=0.00, step=0.01,
    help=("Nur Paare mit Cosinus-Ähnlichkeit ≥ Schwellenwert werden exportiert."),
    disabled=not export_csv
)

faiss_k = st.sidebar.slider("FAISS: top-k pro URL", 5, 200, 50, 5,
                            help="Wie viele Nachbarn pro URL FAISS liefern soll.", disabled=not (export_csv and export_mode.endswith("FAISS top-k")))

# Export 2: Low-Relevance
export_lowrel_csv = st.sidebar.checkbox(
    "Low-Relevance-URLs exportieren", value=False,
    help=("URLs mit Cosinus-Similarity zum Zentrum unterhalb der Schwelle. Robust mit Medoid/Geometric Median möglich.")
)
lowrel_threshold = st.sidebar.slider(
    "Ähnlichkeitsschwelle zum Zentrum (Cosinus)",
    min_value=0.00, max_value=1.00, value=0.40, step=0.01,
    disabled=not export_lowrel_csv
)

# Export 3: Cluster-Qualität
export_quality = st.sidebar.checkbox(
    "Cluster-Qualität (Intra/Inter) berechnen", value=False,
    help="Erzeugt Tabellen zu Intra-/Inter-Cluster-Ähnlichkeiten (Mittel/Median) und bietet CSV-Download an."
)

# Export-Limits
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

    # Basis-Matrix + optionale L2-Norm
    X = np.array(merged["embedding_vector"].tolist()).astype("float32")
    Xn = l2_normalize(X) if use_l2 else X.copy()

    # 2D-Projektion (t-SNE oder UMAP)
    use_umap = (HAS_UMAP and "UMAP" in proj_method)

    if use_umap:
        umap_metric = "euclidean" if metric_label == "Euklidisch" else "cosine"
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=int(umap_n_neighbors),
                            min_dist=float(umap_min_dist),
                            metric=umap_metric,
                            random_state=42)
        proj_input = Xn if umap_metric == "cosine" else X
        Y = reducer.fit_transform(proj_input)
        centroid_mode_eff = None  # wird unten gesetzt, falls Zentrum aktiviert ist
    else:
        perplexity = int(min(30, max(5, len(merged) // 3)))
        tsne_input = Xn if tsne_metric == "cosine" else X
        tsne = TSNE(n_components=2, metric=tsne_metric, random_state=42, perplexity=perplexity)
        Y = tsne.fit_transform(tsne_input)
        centroid_mode_eff = None

    merged["tsne_x"], merged["tsne_y"] = Y[:, 0], Y[:, 1]

    # Cluster im Originalraum (immer auf Xn, wenn L2 aktiv, sonst X)
    cluster_input = Xn if use_l2 else X

    method = cluster_method
    segment_col = segment_col_global

    if method == "K-Means":
        kmeans = KMeans(n_clusters=cluster_k, random_state=42)
        merged["Cluster"] = kmeans.fit_predict(cluster_input).astype(str)
    elif method == "DBSCAN (Cosinus)":
        cos_dist = cosine_distances(cluster_input)
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

    # Zentrum optional (Marker-Koordinaten aus 2D-Raum)
    st.session_state["centroid_in_tsne"] = bool(show_centroid)
    st.session_state["centroid_xy"] = None
    st.session_state["centroid_mode_eff"] = None

    if show_centroid:
        c_vec, centroid_mode_eff = compute_centroid(cluster_input, centroid_mode)
        st.session_state["centroid_mode_eff"] = centroid_mode_eff
        # Näherung: projiziere c_vec in 2D, indem du den gleichen Reducer erneut aufsetzt
        if use_umap:
            reducer2 = umap.UMAP(n_components=2,
                                 n_neighbors=int(umap_n_neighbors),
                                 min_dist=float(umap_min_dist),
                                 metric=("euclidean" if metric_label == "Euklidisch" else "cosine"),
                                 random_state=42)
            Z = reducer2.fit_transform(np.vstack([cluster_input, c_vec[None, :]]))
        else:
            tsne2 = TSNE(n_components=2, metric=tsne_metric, random_state=42, perplexity=min(30, max(5, len(merged) // 3)))
            Z = tsne2.fit_transform(np.vstack([cluster_input if tsne_metric == "cosine" else X, c_vec[None, :]]))
        st.session_state["centroid_xy"] = (float(Z[-1, 0]), float(Z[-1, 1]))

    # Cache
    st.session_state["merged_cached"] = merged
    st.session_state["scaled_cached"] = scaled
    st.session_state["hover_cols_cached"] = _build_hover_cols(merged, metric_col)
    st.session_state["plot_title_cached"] = "🔍 2D-Projektion der Seiten-Embeddings (mit Skalierung)" if scaled else "🔍 2D-Projektion der Seiten-Embeddings"
    st.session_state["bg_color_cached"] = bg_color
    st.session_state["highlight_px_cached"] = max(int(size_min * float(bubble_scale)) + 6, 8)
    st.session_state["url_col_cached"] = url_col


def render_plot_from_cache(q: str):
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

    if q:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
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
            fig.add_trace(go.Scatter(
                x=hi["tsne_x"], y=hi["tsne_y"], mode="markers", name="Treffer",
                marker=dict(size=hi["__marker_px"].tolist(), color="orange", line=dict(width=2, color="black")),
                hovertext=hover_texts, hoverinfo="text", hoverlabel=dict(bgcolor="orange", font_color="black", bordercolor="black"),
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
            merged, x="tsne_x", y="tsne_y", color="Cluster", category_orders={"Cluster": cluster_order},
            hover_data=hover_cols, template="plotly_white", title=title,
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
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, legendgroup=name, showlegend=True,
                                     marker=dict(size=12, color=color_by_name.get(name, None), line=dict(width=0.5, color="white")),
                                     hoverinfo="skip"))

    if centroid_xy is not None:
        cx, cy = centroid_xy
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers", name="Zentrum",
                                 marker=dict(symbol="star", size=int(centroid_size), color="red", line=dict(width=1, color="black")),
                                 hoverlabel=dict(bgcolor="red", font_color="white", bordercolor="black")))

    fig.update_layout(title=title, plot_bgcolor=bg, paper_bgcolor=bg, height=750,
                      margin=dict(l=10, r=10, t=50, b=10), legend_title="Cluster", showlegend=True,
                      dragmode="zoom", hovermode="closest", legend=dict(itemsizing="constant"))

    st.subheader("📈 Visualisierung")
    if centroid_mode_eff:
        st.caption(f"Zentrums-Modus aktiv: {centroid_mode_eff}")
    st.plotly_chart(fig, use_container_width=True)

    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button(label="📥 Interaktive HTML-Datei herunterladen", data=html_bytes, file_name="embedding_plot.html", mime="text/html")

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
if export_csv:
    merged_cached = st.session_state.get("merged_cached")
    if merged_cached is not None:
        with st.spinner("Berechne semantische Ähnlichkeiten…"):
            url_list = merged_cached[url_col].astype(str).tolist()
            X_raw = np.array(merged_cached["embedding_vector"].tolist()).astype("float32")
            Xn = l2_normalize(X_raw) if use_l2 else X_raw
            thr = float(sim_threshold)
            pairs = []

            if export_mode.startswith("Exakt"):
                sim_matrix = cosine_similarity(Xn)
                n = len(url_list)
                est_pairs = n * (n - 1) // 2
                if unlimited_export and est_pairs > 2_000_000 and thr <= 0.2:
                    st.warning(f"Viele Paare erwartet (~{est_pairs:,}). Niedrige Schwelle + kein Limit kann sehr große CSVs erzeugen.")
                for i in range(n):
                    row = sim_matrix[i, i+1:]
                    j_idx = np.where(row >= thr)[0]
                    if len(j_idx):
                        for off in j_idx:
                            j = i + 1 + int(off)
                            s = float(sim_matrix[i, j])
                            pairs.append({"URL_A": url_list[i], "URL_B": url_list[j], "Cosinus_Ähnlichkeit": s, "Match-Typ": "Similarity (sklearn)"})

            else:  # FAISS top-k
                if not HAS_FAISS:
                    st.error("FAISS ist nicht installiert. Bitte 'faiss-cpu' in requirements aufnehmen oder Exakt-Modus wählen.")
                else:
                    # Cosine via Inner Product auf L2-normalisierten Vektoren
                    d = Xn.shape[1]
                    # Annäherung: HNSW-Index (schnell & gut)
                    index = faiss.IndexHNSWFlat(d, 32)
                    index.hnsw.efConstruction = 200
                    faiss.normalize_L2(Xn)
                    index.add(Xn)
                    index.hnsw.efSearch = max(64, faiss_k)
                    sims, ids = index.search(Xn, faiss_k)

                    # Paare sammeln (nur i<j) und Schwelle beachten
                    seen = set()
                    n = len(url_list)
                    for i in range(n):
                        for rank in range(faiss_k):
                            j = int(ids[i, rank])
                            if j < 0 or j == i:
                                continue
                            a, b = (i, j) if i < j else (j, i)
                            if (a, b) in seen:
                                continue
                            s = float(sims[i, rank])
                            if s >= thr:
                                seen.add((a, b))
                                pairs.append({
                                    "URL_A": url_list[a],
                                    "URL_B": url_list[b],
                                    "Cosinus_Ähnlichkeit": s,
                                    "Match-Typ": f"Similarity (FAISS top-{faiss_k})"
                                })

            if not pairs:
                st.warning("Keine Paare über der eingestellten Ähnlichkeitsschwelle gefunden.")
            else:
                if (max_export_rows is not None) and (len(pairs) > max_export_rows):
                    st.warning(f"Export auf {int(max_export_rows):,} Zeilen begrenzt (von {len(pairs):,}).")
                    pairs = pairs[: int(max_export_rows)]

                sim_df = pd.DataFrame(pairs).sort_values("Cosinus_Ähnlichkeit", ascending=False, kind="stable")
                csv_bytes = sim_df.to_csv(index=False).encode("utf-8-sig")
                label = (f"📥 Cosinus-Ähnlichkeiten als CSV (≥ {thr:.2f}, sklearn)" if export_mode.startswith("Exakt")
                         else f"📥 Cosinus-Ähnlichkeiten als CSV (≥ {thr:.2f}, FAISS top-{faiss_k})")
                fname = (f"cosinus_aehnlichkeiten_ge_{thr:.2f}_sklearn.csv" if export_mode.startswith("Exakt")
                         else f"cosinus_aehnlichkeiten_ge_{thr:.2f}_faiss_top{faiss_k}.csv")
                st.download_button(label=label, data=csv_bytes, file_name=fname, mime="text/csv")
    else:
        st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")

if export_lowrel_csv:
    merged_cached = st.session_state.get("merged_cached")
    if merged_cached is not None:
        with st.spinner("Berechne Zentrums-Ähnlichkeiten pro URL…"):
            X_raw = np.array(merged_cached["embedding_vector"].tolist())
            Xn = l2_normalize(X_raw) if use_l2 else X_raw
            c_vec, centroid_mode_eff_export = compute_centroid(Xn if use_l2 else X_raw, centroid_mode)
            # Similarity immer auf L2-normierten Vektoren bewerten
            c_unit = l2_normalize(c_vec[None, :])[0]
            centroid_sim = (l2_normalize(X_raw) @ c_unit) if use_l2 else cosine_similarity(X_raw, c_unit[None, :]).ravel()
            centroid_sim = np.array(centroid_sim).ravel()

            low_thr = float(lowrel_threshold)
            export_df = pd.DataFrame({
                "URL": merged_cached[url_col].astype(str).values,
                "Cosinus_Ähnlichkeit_zum_Zentrum": centroid_sim
            })

            if "Cluster" in merged_cached.columns:
                export_df["Cluster"] = merged_cached["Cluster"].astype(str).values

            if size_by != "Keine Skalierung" and size_by in merged_cached.columns:
                export_df[size_by] = merged_cached[size_by].values

            export_df = export_df[export_df["Cosinus_Ähnlichkeit_zum_Zentrum"] < low_thr].copy()
            export_df = export_df.sort_values("Cosinus_Ähnlichkeit_zum_Zentrum", ascending=True)

            if export_df.empty:
                st.warning("Keine Seiten unterhalb der eingestellten Zentrums-Schwelle gefunden.")
            else:
                if (max_export_rows is not None) and (len(export_df) > max_export_rows):
                    st.warning(f"Export auf {int(max_export_rows):,} Zeilen begrenzt (von {len(export_df):,}).")
                    export_df = export_df.head(int(max_export_rows))

                csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 Low-Relevance-URLs als CSV (Zentrum < {low_thr:.2f})",
                    data=csv_bytes,
                    file_name=f"low_relevance_urls_center_lt_{low_thr:.2f}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Für den Export bitte zuerst **Let's Go / Refresh** ausführen.")

# Cluster-Qualität: Intra- & Inter-Ähnlichkeiten
if export_quality:
    merged_cached = st.session_state.get("merged_cached")
    if merged_cached is not None:
        with st.spinner("Berechne Cluster-Qualität…"):
            X_raw = np.array(merged_cached["embedding_vector"].tolist()).astype("float32")
            Xn = l2_normalize(X_raw) if use_l2 else X_raw
            clusters = merged_cached["Cluster"].astype(str).values
            labels = pd.Index(sorted(pd.unique(clusters), key=lambda x: ("~" if x == "-1" else "") + x))

            # Intra
            intra_rows = []
            for c in labels:
                idx = np.where(clusters == c)[0]
                if len(idx) < 2:
                    intra_rows.append({"Cluster": c, "N": int(len(idx)), "Intra_Mean": np.nan, "Intra_Median": np.nan})
                    continue
                S = (Xn[idx] @ Xn[idx].T)
                tril = S[np.tril_indices(len(idx), k=-1)]
                intra_rows.append({"Cluster": c, "N": int(len(idx)), "Intra_Mean": float(np.mean(tril)), "Intra_Median": float(np.median(tril))})
            intra_df = pd.DataFrame(intra_rows)

            # Inter (Matrix)
            inter_mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
            for i, ci in enumerate(labels):
                idx_i = np.where(clusters == ci)[0]
                Xi = Xn[idx_i]
                for j, cj in enumerate(labels):
                    if j < i:
                        continue
                    idx_j = np.where(clusters == cj)[0]
                    Xj = Xn[idx_j]
                    if len(idx_i) == 0 or len(idx_j) == 0:
                        val_mean = val_med = np.nan
                    else:
                        S = Xi @ Xj.T
                        val_mean = float(np.mean(S))
                        val_med = float(np.median(S))
                    inter_mat.loc[ci, cj] = val_mean
                    inter_mat.loc[cj, ci] = val_mean

            st.subheader("📊 Cluster-Qualität: Intra-Ähnlichkeit")
            st.dataframe(intra_df, use_container_width=True)
            st.download_button("📥 Intra-Ähnlichkeiten (CSV)", data=intra_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="cluster_intra_similarity.csv", mime="text/csv")

            st.subheader("📊 Cluster-Qualität: Inter-Ähnlichkeits-Matrix (Mean)")
            st.dataframe(inter_mat, use_container_width=True)
            st.download_button("📥 Inter-Ähnlichkeiten (CSV)", data=inter_mat.to_csv(index=True).encode("utf-8-sig"),
                               file_name="cluster_inter_similarity_mean.csv", mime="text/csv")

            st.caption("Interpretation: Hohe Intra-Werte = kohärente Cluster. Niedrige Inter-Werte = gut trennbare Themen. Wenn zwei Cluster hohe Inter-Werte haben, könnten sie zusammengehören.")
    else:
        st.info("Für die Qualitätsberechnung bitte zuerst **Let's Go / Refresh** ausführen.")
