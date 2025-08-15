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

# =============================
# Page setup & Branding
# =============================
st.set_page_config(page_title="ONE Content-Cluster-Visualizer", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Content-Cluster-Visualizer – Domains visuell analysieren")

st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 600px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</div>
<hr>
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
            # --- HARTE SICHERUNG ---
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
            # Wenn hier: ok oder bewusst 1 Spalte
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
            # Falls Werte '0.1, 0.2, 0.3' ohne Klammern
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
            # JSON-Liste oder viele Zahlen/Kommas -> wie Embedding-Vektor
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

# -------------------------------------------------
# URL-Kandidaten (für beide Dateien wiederverwendet)
# -------------------------------------------------
URL_CANDIDATES_BASE = [
    "URL", "Page", "Pages",
    "Adresse", "Address",
    "Seite", "Seiten",
    "URLs",
    "URL-Adresse", "URL Adresse",
]

# Für GSC typische Zusatz-Bezeichnungen
URL_CANDIDATES_GSC_EXTRA = [
    "Landing Page", "Seiten-URL", "Seiten URL"
]

# =============================
# Uploads
# =============================
st.subheader("1) Embedding-Datei hochladen")
emb_file = st.file_uploader("CSV/Excel mit URLs und Embeddings", type=["csv", "xlsx", "xls"], key="emb")

st.subheader("2) Optional: Search-Console-Datei hochladen (Klicks/Impressionen)")
gsc_file = st.file_uploader("GSC CSV/Excel (optional)", type=["csv", "xlsx", "xls"], key="gsc")

if emb_file is None:
    st.info("Bitte zuerst die Embedding-Datei hochladen.")
    st.stop()

# Read embeddings table
try:
    df = robust_read_table(emb_file)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sichtbare Diagnose: welche Spalten hat die Datei?
st.caption(f"Columns detected (Embedding-Datei): {list(df.columns)}")

# --- Spaltenfindung (robust) ---

# URL-Spalte im Embedding-Dokument
url_col = find_column(URL_CANDIDATES_BASE, df.columns)

# Fuzzy-Fallback: alles, was sinngemäß nach URL/Page aussieht
if url_col is None:
    for c in df.columns:
        n = str(c).lower().replace("-", " ")
        if any(tok in n for tok in ["url", "page", "adresse", "address", "seite", "seiten"]):
            url_col = c
            break

# Embedding-Spalte
# 1) Harte Kandidatenliste
embedding_col = find_column([
    "ChatGPT Embedding Erzeugung",
    "ChatGPT Embedding Erzeugung 1",
    "Embedding",
    "Embeddings",
    "Embedding Vector",
    "OpenAI Embedding",
    "Extract embeddings from page content",  # häufige Export-Bezeichnung
], df.columns)

# 2) NEU: Fuzzy-Match per Spaltenname
if embedding_col is None:
    for c in df.columns:
        colname = str(c).lower()
        if ("embedding" in colname) or ("embed" in colname) or ("vector" in colname):
            embedding_col = c
            break

# 3) Fallback: heuristische Inhalts-Erkennung
if embedding_col is None:
    embedding_col = autodetect_embedding_column(df)

# Falls URL oder Embedding weiterhin fehlen: aussagekräftige Fehlermeldung
if url_col is None or embedding_col is None:
    st.error(
        "❌ URL- oder Embedding-Spalte nicht gefunden.\n\n"
        f"Gefundene Spalten: {list(df.columns)}\n\n"
        "Erwarte URL-Spalte (z. B. URL/Adresse/Page) und eine Embedding-Spalte (z. B. 'Embedding' oder eine Liste von Zahlen)."
    )
    st.stop()

# Parse/normalize embeddings
with st.spinner("Verarbeite Embeddings…"):
    df["embedding_vector"] = df[embedding_col].apply(parse_embedding)
    df_valid = df[df["embedding_vector"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    df_valid["embedding_vector"], dim = normalize_embedding_lengths(df_valid["embedding_vector"])

if len(df_valid) < 5:
    st.error("❌ Zu wenige gültige Embeddings. Mindestens 5 erforderlich.")
    st.stop()

embedding_matrix = np.array(df_valid["embedding_vector"].tolist())
st.caption(f"✅ Gültige Embeddings: {len(df_valid)} · Vektor-Dim: {embedding_matrix.shape[1]}")

# Optional: read GSC file
perf_df = None
perf_url_col = clicks_col = impressions_col = None
if gsc_file is not None:
    try:
        perf_df = robust_read_table(gsc_file)
        st.caption(f"Columns detected (GSC-Datei): {list(perf_df.columns)}")

        # URL-Spalte im GSC-Dokument
        perf_url_col = find_column(URL_CANDIDATES_BASE + URL_CANDIDATES_GSC_EXTRA, perf_df.columns)

        # Fuzzy-Fallback auch für GSC
        if perf_url_col is None:
            for c in perf_df.columns:
                n = str(c).lower().replace("-", " ")
                if any(tok in n for tok in ["url", "page", "adresse", "address", "seite", "seiten", "landing"]):
                    perf_url_col = c
                    break

        # Klicks/Impressions
        for c in perf_df.columns:
            name = str(c).strip().lower()
            if clicks_col is None and ("klick" in name or "click" in name):
                clicks_col = c
            if impressions_col is None and ("impress" in name):
                impressions_col = c

        if perf_url_col is None:
            st.warning("⚠️ Konnte URL-Spalte in der GSC-Datei nicht erkennen – Bubbles werden nicht nach GSC-Metriken skaliert.")
            perf_df = None
    except Exception as e:
        st.warning(f"GSC-Datei konnte nicht verarbeitet werden: {e}")
        perf_df = None

# =============================
# Sidebar Controls (Look & Feel wie ONE Redirector)
# =============================
st.sidebar.header("Einstellungen")
cluster_method = st.sidebar.selectbox(
    "Clustermethode",
    ["K-Means", "Segments", "DBSCAN (Cosinus)"],
    help="K-Means: feste Clusterzahl • Segments: nimmt Segment-Spalte z.B. aus Screaming Frog Crawl • DBSCAN: dichtebasiert (Cosinus)",
)

cluster_k = st.sidebar.slider(
    "Cluster (nur K-Means)",
    min_value=2,
    max_value=20,
    value=8,
    step=1,
    help="Legt fest, in wie viele Gruppen (Cluster) die Punkte bei der K-Means-Methode unterteilt werden. "
         "Eine höhere Zahl erzeugt kleinere, spezialisiertere Gruppen; eine niedrigere Zahl erzeugt größere, "
         "allgemeinere Cluster. Nur relevant, wenn ‚K-Means‘ gewählt ist."
)

# Darstellungsmethode (Abstand) – ehemals t-SNE-Metrik
metric_label = st.sidebar.selectbox(
    "Darstellungsmethode (Abstand)",
    ["Euklidisch", "Cosinus"],
    help=("Bestimmt, wie der Abstand bzw. die Ähnlichkeit zwischen Seiten berechnet wird, bevor die t-SNE-Visualisierung erstellt wird.\n\n"
          "Euklidisch: misst die „Luftlinie“ im Embedding-Raum, stabil und Standard.\n\n"
          "Cosinus: misst den Winkel zwischen Vektoren, gut geeignet für semantische Ähnlichkeiten.\n"
          "Die Wahl beeinflusst nur die Darstellung, nicht die eigentlichen Daten.")
)
tsne_metric = "euclidean" if metric_label == "Euklidisch" else "cosine"

# Size scaling options
size_options = ["Keine Skalierung"]
if clicks_col:
    size_options.append("Klicks")
if impressions_col:
    size_options.append("Impressionen")

size_by = st.sidebar.selectbox(
    "Bubblegröße nach",
    size_options,
    help="Welche Metrik bestimmt die Blasengröße (Klicks/Impressionen)? 'Keine Skalierung' = konstant.",
)
size_method = st.sidebar.radio("Skalierung", ["Logarithmisch (log1p)", "Linear (Min–Max)"], index=0)
size_min = st.sidebar.slider("Min-Größe (px)", 1, 12, 2)
size_max = st.sidebar.slider("Max-Größe (px)", 6, 40, 10)
clip_low = st.sidebar.slider("Clip low %", 0, 20, 1)
clip_high = st.sidebar.slider("Clip high %", 80, 100, 95)

bubble_scale = st.sidebar.slider("Bubble-Scale (global)", 0.20, 1.0, 0.55, 0.05)
show_centroid = st.sidebar.checkbox("Centroid markieren", value=False)
export_csv = st.sidebar.checkbox("Cosinus-CSV exportieren", value=False)

bg_color = st.sidebar.color_picker("Hintergrundfarbe", value="#FFFFFF")
search_q = st.sidebar.text_input("🔍 URL-Suche (Teilstring)")

recalc = st.sidebar.button("Let's Go", type="primary")

# =============================
# Processing & Visualization
# =============================

def build_plot():
    # Merge GSC metrics
    merged = df_valid.copy()
    if isinstance(perf_df, pd.DataFrame) and perf_url_col:
        merged["__join"] = merged[url_col].apply(normalize_url)
        perf_local = perf_df.copy()
        perf_local["__join"] = perf_local[perf_url_col].apply(normalize_url)
        keep_cols = ["__join"] + [c for c in [clicks_col, impressions_col] if c and c in perf_local.columns]
        perf_keep = perf_local[keep_cols].drop_duplicates("__join")
        merged = merged.merge(perf_keep, on="__join", how="left")
        merged.drop(columns=["__join"], inplace=True, errors="ignore")

    # t-SNE
    perplexity = int(min(30, max(5, len(merged) // 3)))
    X = embedding_matrix
    use_centroid = bool(show_centroid)
    if use_centroid:
        centroid_vec = np.mean(embedding_matrix, axis=0, keepdims=True)
        X = np.vstack([embedding_matrix, centroid_vec])

    try:
        tsne = TSNE(n_components=2, metric=tsne_metric, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(X)
    except Exception as e:
        st.error(f"❌ Fehler bei t-SNE: {e}")
        return None, None

    merged["tsne_x"] = tsne_result[: len(embedding_matrix), 0]
    merged["tsne_y"] = tsne_result[: len(embedding_matrix), 1]

    # Cluster
    method = cluster_method

    # --- Segment/Cluster-Spalte robust erkennen ---
    # 1) Exakte Kandidaten (case-insensitive via find_column)
    segment_col = find_column(["Segmente", "Segment", "Segments", "Cluster"], df.columns)

    # 2) Fuzzy: Namen, die "segment" oder "cluster" enthalten (Bindestriche/Underscores egal)
    if segment_col is None:
        for c in df.columns:
            n = str(c).lower().replace("-", " ").replace("_", " ")
            tokens = n.split()
            if any(tok in tokens for tok in ["segment", "segments", "cluster"]):
                segment_col = c
                break

    if method == "K-Means":
        kmeans = KMeans(n_clusters=cluster_k, random_state=42)
        merged["Cluster"] = kmeans.fit_predict(embedding_matrix).astype(str)

    elif method == "DBSCAN (Cosinus)":
        cos_dist = cosine_distances(embedding_matrix)
        dbscan = DBSCAN(eps=0.3, min_samples=5, metric="precomputed")
        merged["Cluster"] = dbscan.fit_predict(cos_dist).astype(str)

    elif method == "Segments":
        if segment_col:
            merged["Cluster"] = merged[segment_col].fillna("Unbekannt").astype(str)
        else:
            st.warning("⚠️ Keine Segment-/Cluster-Spalte gefunden – falle zurück auf 'Kein Segment'.")
            merged["Cluster"] = "Kein Segment"

    else:
        merged["Cluster"] = "Kein Segment"

    # Suche
    merged["Highlight"] = False
    q = (search_q or "").strip().lower()
    if q:
        merged["Highlight"] = merged[url_col].astype(str).str.lower().str.contains(q, na=False)
        st.caption(f"✨ {int(merged['Highlight'].sum())} Treffer für „{q}“")

    # Bubble sizes (pixel diameter)
    scaled = False
    if size_by != "Keine Skalierung":
        metric_col = clicks_col if size_by == "Klicks" else impressions_col
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
            st.warning("⚠️ Gewählte Metrik nicht gefunden – konstante Bubble-Größe.")
    if not scaled:
        merged["__marker_size"] = float(size_min)

    shrink = float(bubble_scale)
    if scaled:
        merged["__marker_px"] = (merged["__marker_size"] * shrink).clip(lower=1)
    else:
        merged["__marker_px"] = max(1, int(size_min * shrink))

    # Plot
    title = "🔍 t-SNE der Seiten-Embeddings (mit GSC-Skalierung)" if scaled else "🔍 t-SNE der Seiten-Embeddings"
    hover_cols = {url_col: True, "Cluster": True}
    if clicks_col and clicks_col in merged.columns:
        hover_cols[clicks_col] = True
    if impressions_col and impressions_col in merged.columns:
        hover_cols[impressions_col] = True

    fig = px.scatter(
        merged,
        x="tsne_x",
        y="tsne_y",
        color=merged["Cluster"].astype(str),
        hover_data=hover_cols,
        template="plotly_white",
        title=title,
    )

    # Assign exact per-point pixel sizes per trace
    for tr in fig.data:
        mask = (merged["Cluster"].astype(str) == tr.name)
        sizes = merged.loc[mask, "__marker_px"].tolist()
        tr.marker.update(size=sizes, sizemode="diameter", opacity=0.55, line=dict(width=0.5, color="white"))

    # Centroid
    if use_centroid:
        cx, cy = tsne_result[len(embedding_matrix), 0], tsne_result[len(embedding_matrix), 1]
        centroid_trace = px.scatter(x=[cx], y=[cy]).update_traces(
            marker=dict(symbol="star", size=14, color="red"),  # line optional
            name="Centroid",
        )
        fig.add_trace(centroid_trace.data[0])

    # Highlight layer
    if merged["Highlight"].any():
        hi = merged[merged["Highlight"]]
        highlight_trace = px.scatter(hi, x="tsne_x", y="tsne_y", hover_data={url_col: True}).update_traces(
            marker=dict(size=max(int(size_min * shrink) + 6, 8), color="yellow", line=dict(width=2, color="black")),
            showlegend=False,
        )
        fig.add_trace(highlight_trace.data[0])

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=750,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title="Cluster",
        showlegend=True,
        dragmode="zoom",
        hovermode="closest",
    )

    return fig, merged


if recalc:
    with st.spinner("Berechne t-SNE & erstelle Plot…"):
        fig, merged = build_plot()
        if fig is not None:
            st.subheader("📈 Visualisierung")
            st.plotly_chart(fig, use_container_width=True)

            # Downloads
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                label="📥 Interaktive HTML-Datei herunterladen",
                data=html_bytes,
                file_name="tsne_plot.html",
                mime="text/html",
            )

            if export_csv:
                with st.spinner("Berechne Cosinus-Ähnlichkeiten…"):
                    url_list = merged[url_col].astype(str).tolist()
                    sim_matrix = cosine_similarity(embedding_matrix)
                    pairs = []
                    for i in range(len(url_list)):
                        for j in range(i + 1, len(url_list)):
                            pairs.append(
                                {
                                    "URL_A": url_list[i],
                                    "URL_B": url_list[j],
                                    "Cosinus_Ähnlichkeit": float(sim_matrix[i, j]),
                                }
                            )
                    sim_df = pd.DataFrame(pairs)
                    csv_bytes = sim_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 Cosinus-Ähnlichkeiten als CSV herunterladen",
                        data=csv_bytes,
                        file_name="cosinus_aehnlichkeiten.csv",
                        mime="text/csv",
                    )
else:
    st.info("Wähle Einstellungen in der Sidebar und klicke auf **Let's Go**.")
