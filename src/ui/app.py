import streamlit as st
import pandas as pd
import torch
from pathlib import Path
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image
import io

from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import create_engine, inspect
from sentence import Sentence
import time
import sys
from pathlib import Path

# On récupère la racine du projet (2 niveaux au-dessus de src/ui/app.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Maintenant tu peux importer normalement
from src.agent.main import setup_agent, run
from src.reports import generate_briefing


# ============================================================
# 1) INITIALISATION DE L'AGENT (GROQ + PROMPT ENGINEERING + DONNÉES)
# ============================================================
@st.cache_resource
def get_agent_resources():
    # Initialise le LLM Groq avec accès aux données réelles et le system prompt
    llm, engines, tools, system_prompt = setup_agent()
    return llm, engines, system_prompt

llm, engines, system_prompt = get_agent_resources()


# ============================================================
# 0) CONFIG STREAMLIT
# ============================================================
st.set_page_config(page_title="Mobility Copilot - Hackathon 2026", layout="wide")
st.title("Mobility Copilot : Analyse Intelligente")


# ============================================================
# 1) INITIALISATION LLM (NL -> SQL)
# ============================================================
@st.cache_resource
def load_settings():
    device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)
    Settings.llm = Ollama(model="qwen2.5-coder:7b", base_url="http://127.0.0.1:11434", request_timeout=600.0)
    return Settings.llm


llm = load_settings()


# ============================================================
# 2) CONNEXION SQLITE + SCHEMA
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
db_path = ROOT / "data" / "raw" / "mobility.db"

engine = create_engine(f"sqlite:///{db_path.as_posix()}")
sql_database = SQLDatabase(engine)


def get_real_schema() -> str:
    inst = inspect(engine)
    out = []
    for table in ["requetes_311", "collisions", "weather_montreal"]:
        cols = [c["name"] for c in inst.get_columns(table)]
        out.append(f"- Table '{table}': Colonnes {cols}")
    return "\n".join(out)


# ============================================================
# 3) MOTEUR NL -> SQL
# ============================================================
descriptions_tables = {
    "requetes_311": "Signalements 311. Date: 'DDS_DATE_CREATION'. Motif/catégorie: 'NATURE' ou 'ACTI_NOM'.",
    "collisions": "Accidents. Date: 'DT_ACCDN'. Gravité: 'GRAVITE'. Localisation: 'LOC_LAT', 'LOC_LONG'.",
    "weather_montreal": "Météo. Date: 'time'. Colonnes: precipitation_sum, snowfall_sum, temperature_2m_max, temperature_2m_min.",
}

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["requetes_311", "collisions", "weather_montreal"],
    context_query_kwargs=descriptions_tables,
)


# ============================================================
# 4) SIDEBAR
# ============================================================
st.sidebar.header("Schéma de la Base")
st.sidebar.caption(f"DB: {db_path}")
with st.sidebar.expander("Voir les colonnes réelles"):
    st.text(get_real_schema())


# ============================================================
# 5) DATA ACCESS — COLLISIONS
# ============================================================
@st.cache_data(show_spinner=False)
def load_collisions_points(limit: int = 200_000) -> pd.DataFrame:
    q = f"""
    SELECT
        DT_ACCDN, HEURE_ACCDN, GRAVITE, LOC_LAT, LOC_LONG,
        NB_VICTIMES_TOTAL, NB_MORTS, NB_BLESSES_GRAVES, NB_BLESSES_LEGERS,
        CD_ETAT_SURFC, CD_ECLRM, CD_ENVRN_ACCDN
    FROM collisions
    WHERE LOC_LAT IS NOT NULL AND LOC_LONG IS NOT NULL
      AND TRIM(LOC_LAT) != '' AND TRIM(LOC_LONG) != ''
    LIMIT {int(limit)}
    """
    df = pd.read_sql_query(q, engine)

    df["LOC_LAT"] = pd.to_numeric(df["LOC_LAT"], errors="coerce")
    df["LOC_LONG"] = pd.to_numeric(df["LOC_LONG"], errors="coerce")
    df = df.dropna(subset=["LOC_LAT", "LOC_LONG"])

    dt_norm = df["DT_ACCDN"].astype(str).str.strip().str.replace("/", "-", regex=False)
    df["DT_ACCDN_DT"] = pd.to_datetime(dt_norm, errors="coerce")
    df = df.dropna(subset=["DT_ACCDN_DT"])
    df["DT_ACCDN_STR"] = df["DT_ACCDN_DT"].dt.strftime("%Y-%m-%d")

    df["GRAVITE"] = df["GRAVITE"].astype(str).str.strip()
    df.loc[df["GRAVITE"].isin(["", "None", "nan"]), "GRAVITE"] = "Inconnue"

    for c in ["NB_VICTIMES_TOTAL", "NB_MORTS", "NB_BLESSES_GRAVES", "NB_BLESSES_LEGERS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ============================================================
# 6) DATA ACCESS — 311 (MOTIFS)
# ============================================================
@st.cache_data(show_spinner=False)
def load_311_arrondissements() -> list[str]:
    q = """
    SELECT DISTINCT ARRONDISSEMENT
    FROM requetes_311
    WHERE ARRONDISSEMENT IS NOT NULL AND TRIM(ARRONDISSEMENT) != ''
    ORDER BY ARRONDISSEMENT
    """
    df = pd.read_sql_query(q, engine)
    return ["Tous"] + df["ARRONDISSEMENT"].astype(str).tolist()


@st.cache_data(show_spinner=False)
def load_311_agg(date_start: str, date_end: str, motif_col: str, arrondissement: str, limit: int = 300) -> pd.DataFrame:
    q = f"""
    SELECT
        COALESCE(NULLIF(TRIM({motif_col}), ''), 'Inconnue') AS motif,
        COUNT(*) AS cnt
    FROM requetes_311
    WHERE DATE(DDS_DATE_CREATION) BETWEEN DATE(:ds) AND DATE(:de)
      AND (:arr = 'Tous' OR ARRONDISSEMENT = :arr)
    GROUP BY COALESCE(NULLIF(TRIM({motif_col}), ''), 'Inconnue')
    ORDER BY cnt DESC
    LIMIT {int(limit)}
    """
    return pd.read_sql_query(q, engine, params={"ds": date_start, "de": date_end, "arr": arrondissement})


# ============================================================
# 7) DATA ACCESS — METEO ↔ INCIDENTS (JOURNALIER)
# ============================================================
@st.cache_data(show_spinner=False)
def load_daily_metrics(date_start: str, date_end: str) -> pd.DataFrame:
    q = """
    WITH
    c AS (
        SELECT DATE(REPLACE(DT_ACCDN, '/', '-')) AS d, COUNT(*) AS collisions_count
        FROM collisions
        WHERE DATE(REPLACE(DT_ACCDN, '/', '-')) BETWEEN DATE(:ds) AND DATE(:de)
        GROUP BY DATE(REPLACE(DT_ACCDN, '/', '-'))
    ),
    r AS (
        SELECT DATE(DDS_DATE_CREATION) AS d, COUNT(*) AS req311_count
        FROM requetes_311
        WHERE DATE(DDS_DATE_CREATION) BETWEEN DATE(:ds) AND DATE(:de)
        GROUP BY DATE(DDS_DATE_CREATION)
    ),
    w AS (
        SELECT DATE(time) AS d,
               AVG(precipitation_sum) AS precipitation_sum,
               AVG(snowfall_sum) AS snowfall_sum,
               AVG(temperature_2m_max) AS temperature_2m_max,
               AVG(temperature_2m_min) AS temperature_2m_min
        FROM weather_montreal
        WHERE DATE(time) BETWEEN DATE(:ds) AND DATE(:de)
        GROUP BY DATE(time)
    )
    SELECT
        w.d AS date,
        COALESCE(c.collisions_count, 0) AS collisions_count,
        COALESCE(r.req311_count, 0) AS req311_count,
        COALESCE(w.precipitation_sum, 0) AS precipitation_sum,
        COALESCE(w.snowfall_sum, 0) AS snowfall_sum,
        w.temperature_2m_max,
        w.temperature_2m_min
    FROM w
    LEFT JOIN c ON c.d = w.d
    LEFT JOIN r ON r.d = w.d
    ORDER BY w.d
    """
    df = pd.read_sql_query(q, engine, params={"ds": date_start, "de": date_end})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["collisions_count", "req311_count", "precipitation_sum", "snowfall_sum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def pearson(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return None
    return float(a[m].corr(b[m], method="pearson"))


# ============================================================
# 8) UI — TABS
# ============================================================
tabs = st.tabs(["Collisions (Carte)", "311 (Motifs)", "Météo ↔ Incidents", "Question (NLSQL)", "Question (NLSQL) Audio", "Briefing"])



# ============================================================
# 9) TAB 1 — CARTE / HEATMAP COLLISIONS
# ============================================================
with tabs[0]:
    st.subheader("Collisions — Carte / heatmap")

    df_col = load_collisions_points(limit=200_000)

    min_date = df_col["DT_ACCDN_DT"].min().date()
    max_date = df_col["DT_ACCDN_DT"].max().date()

    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    with c1:
        date_range = st.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="coll_date_range")
    with c2:
        gravites = sorted(df_col["GRAVITE"].dropna().unique().tolist())
        grav_sel = st.multiselect("Gravité", options=gravites, default=gravites, key="coll_grav")
    with c3:
        max_points = st.slider("Nb de points", 5_000, 200_000, 5_000, step=5_000, key="coll_max_points")

    if max_points > 50_000:
        with st.popover("⚠️ Risque de pertes de performance"):
            st.warning("Afficher plus de 50 000 points peut ralentir le navigateur.")
            st.checkbox("Je confirme", value=False, key="confirm_big_points")
        if not st.session_state["confirm_big_points"]:
            st.stop()

    d1, d2 = date_range
    df_map = df_col.loc[
        (df_col["DT_ACCDN_DT"].dt.date >= d1)
        & (df_col["DT_ACCDN_DT"].dt.date <= d2)
        & (df_col["GRAVITE"].isin(grav_sel))
    ].copy()

    if len(df_map) > max_points:
        df_map = df_map.sample(n=max_points, random_state=42)

    center_lat = float(df_map["LOC_LAT"].mean())
    center_lon = float(df_map["LOC_LONG"].mean())

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_map,
        get_position="[LOC_LONG, LOC_LAT]",
        radiusPixels=60,
    )

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[LOC_LONG, LOC_LAT]",
        get_radius=25,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<b>Date:</b> {DT_ACCDN_STR}<br/>"
            "<b>Heure:</b> {HEURE_ACCDN}<br/>"
            "<b>Gravité:</b> {GRAVITE}<br/>"
            "<b>Victimes:</b> {NB_VICTIMES_TOTAL}<br/>"
            "<b>Morts:</b> {NB_MORTS}<br/>"
            "<b>Blessés graves:</b> {NB_BLESSES_GRAVES}<br/>"
            "<b>Blessés légers:</b> {NB_BLESSES_LEGERS}<br/>"
            "<b>État surface:</b> {CD_ETAT_SURFC}<br/>"
            "<b>Éclairage:</b> {CD_ECLRM}<br/>"
            "<b>Environnement:</b> {CD_ENVRN_ACCDN}"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10, pitch=0),
        layers=[heat_layer, points_layer],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


# ============================================================
# 10) TAB 2 — 311 (NUAGE DE MOTS + TOP 10)
# ============================================================
with tabs[1]:
    st.subheader("311 — Nuage de mots des motifs")

    ds_default, de_default = st.session_state["coll_date_range"]
    arrs = load_311_arrondissements()

    c1, c2, c3 = st.columns([1.3, 1.0, 1.0])
    with c1:
        d1, d2 = st.date_input("Période (311)", value=(ds_default, de_default), key="req311_date_range")
    with c2:
        arrondissement = st.selectbox("Arrondissement", options=arrs, index=0, key="req311_arr")
    with c3:
        motif_col = st.selectbox("Colonne motif", options=["NATURE", "ACTI_NOM"], index=0, key="req311_motif_col")

    agg = load_311_agg(str(d1), str(d2), motif_col=motif_col, arrondissement=arrondissement, limit=300)

    # === GRAPHIQUE BAR PLOTLY ===
    top10 = agg.head(10).copy().sort_values("cnt", ascending=True)
    fig_bar = go.Figure(data=[
        go.Bar(
            x=top10["cnt"],
            y=top10["motif"],
            orientation='h',
            marker=dict(
                color=top10["cnt"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Nombre")
            ),
            text=top10["cnt"],
            textposition='auto',
        )
    ])
    fig_bar.update_layout(
        title=f"🔝 Top 10 motifs 311 ({motif_col})",
        xaxis_title="Nombre de signalements",
        yaxis_title=motif_col,
        height=500,
        hovermode='closest',
        template='plotly_dark'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # === WORD CLOUD EN IMAGE ===
    freq = {row["motif"]: int(row["cnt"]) for _, row in agg.iterrows()}
    wc = WordCloud(width=1200, height=500, background_color="white", colormap="viridis").generate_from_frequencies(freq)
    
    # Convertir en image PIL et afficher
    img_buffer = io.BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    img_buffer.seek(0)
    st.image(img_buffer, caption=f"☁️ Nuage de mots — motifs 311 ({motif_col})", use_container_width=True)


# ============================================================
# 11) TAB 3 — METEO ↔ INCIDENTS (SERIES + CORRELATIONS)
# ============================================================
with tabs[2]:
    st.subheader("Corrélations météo ↔ incidents (journalier)")

    ds_default, de_default = st.session_state["coll_date_range"]
    d1, d2 = st.date_input("Période (météo↔incidents)", value=(ds_default, de_default), key="met_date_range")

    daily = load_daily_metrics(str(d1), str(d2))

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Jours couverts", f"{len(daily):,}")
    with k2:
        st.metric("Collisions (total)", f"{int(daily['collisions_count'].sum()):,}")
    with k3:
        st.metric("311 (total)", f"{int(daily['req311_count'].sum()):,}")

    st.caption("Séries journalières (incidents vs météo)")
    plot_df = daily.set_index("date")[["collisions_count", "req311_count", "precipitation_sum", "snowfall_sum"]]
    st.line_chart(plot_df)

    corr_rows = [
        ("collisions_count", "precipitation_sum", pearson(daily["collisions_count"], daily["precipitation_sum"])),
        ("collisions_count", "snowfall_sum", pearson(daily["collisions_count"], daily["snowfall_sum"])),
        ("req311_count", "precipitation_sum", pearson(daily["req311_count"], daily["precipitation_sum"])),
        ("req311_count", "snowfall_sum", pearson(daily["req311_count"], daily["snowfall_sum"])),
    ]
    corr_df = pd.DataFrame(corr_rows, columns=["incident", "meteo", "pearson_r"])
    st.subheader("Corrélation (Pearson)")
    st.dataframe(corr_df, use_container_width=True)

    st.subheader("Comparaison jours météo (moyennes)")
    snow_days = daily["snowfall_sum"].fillna(0) > 0
    rain_days = daily["precipitation_sum"].fillna(0) > 0

    def mean_if(mask: pd.Series, col: str) -> float:
        s = pd.to_numeric(daily.loc[mask, col], errors="coerce")
        return float(s.mean()) if s.notna().any() else 0.0

    comp = pd.DataFrame(
        [
            ["Neige", "Collisions", mean_if(snow_days, "collisions_count"), mean_if(~snow_days, "collisions_count")],
            ["Neige", "311", mean_if(snow_days, "req311_count"), mean_if(~snow_days, "req311_count")],
            ["Pluie", "Collisions", mean_if(rain_days, "collisions_count"), mean_if(~rain_days, "collisions_count")],
            ["Pluie", "311", mean_if(rain_days, "req311_count"), mean_if(~rain_days, "req311_count")],
        ],
        columns=["Condition", "Incident", "Moyenne jours AVEC", "Moyenne jours SANS"],
    )
    st.dataframe(comp, use_container_width=True)

# ============================================================
# 12) TAB 4 — CHAT LANGGRAPH (Remplace NLSQL)
# ============================================================
with tabs[3]:
    st.subheader("Assistant Intelligent (LangGraph + Groq)")

    question = st.text_input("Posez votre question à l'analyste :", key="nl_question")

    if st.button("Analyser", key="nl_btn"):
        if question:
            with st.spinner("L'agent analyse les fichiers CSV et réfléchit..."):
                try:
                    # Exécution asynchrone de ton graphe
                    import asyncio
                    response_text = asyncio.run(run(llm, question, engines, system_prompt))

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.chat_message("assistant").write(response_text)
                        st.success("Analyse terminée")

                    with col2:
                        st.info("💡 **Note :** Cet agent utilise Llama-3.1 via Groq pour manipuler directement les DataFrames (Pandas) des collisions, du 311 et de la météo.")
                
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'agent : {e}")
                    
# ============================================================
# 13) TAB 5 — CHAT NLSQL (VOCAL)
# ============================================================

with tabs[4]:
    st.header("Copilote vocal (voiture)")
    
    voice_manager = Sentence(language='fr-FR')
    
    # --- 1. ZONE D'UPLOAD CENTRÉE ---
    empty_l, center_col, empty_r = st.columns([1, 2, 1])
    
    with center_col:
        st.write("### Posez votre question oralement")
        uploaded_audio = st.file_uploader(
            "Déposez votre fichier .wav ou .flac pour lancer l'analyse", 
            type=["wav", "flac"], 
            key="audio_auto_upload"
        )

    # --- 2. TRAITEMENT AUTOMATIQUE ---
    if uploaded_audio:
        st.markdown("---")
        
        final_response = None
        transcript = ""
        audio_file_path = "auto_answer.mp3"
        
        # --- BLOC DE CHARGEMENT VISUEL ---
        with st.status("Traitement vocal...", expanded=True) as status:
            st.write("Transcription de l'audio...")
            temp_path = "temp_vocal_auto.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            transcript = voice_manager.toText(temp_path)
            
            if transcript and "[Error]" not in transcript:
                st.write("Analyse via LangGraph...")
                try:
                    import asyncio
                    final_response = asyncio.run(run(llm, transcript, engines, system_prompt))
                    
                    if final_response:
                        st.write("Génération de la réponse vocale...")
                        voice_manager.toSpeech(final_response, audio_file_path)
                        status.update(label="✅ Analyse terminée !", state="complete")
                except Exception as e:
                    st.error(f"Erreur agent: {e}")
                    import traceback
                    st.write(traceback.format_exc())
        
        # --- AFFICHAGE DE LA CONVERSATION (APRÈS LE STATUS) ---
        if transcript and "[Error]" not in transcript:
            st.markdown("---")
            st.subheader("📝 Conversation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.chat_message("user").write(f"**Votre question :**\n\n{transcript}")
            with col2:
                if final_response:
                    st.chat_message("assistant").write(f"**Réponse de l'assistant :**\n\n{final_response}")
                else:
                    st.info("En attente de la réponse...")
            
            # --- LECTURE AUDIO ---
            if final_response and Path(audio_file_path).exists():
                st.markdown("---")
                st.subheader("🔊 Écouter la réponse")
                with open(audio_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mpeg", autoplay=False)

# ============================================================
# 14) TAB 6 — BRIEFINGS AUTOMATIQUES
# ============================================================

with tabs[5]:
    st.subheader("📋 Briefings Automatiques")
    
    # Charger les données une seule fois
    @st.cache_data(show_spinner=False)
    def load_full_datasets():
        """Charge les datasets complets pour les briefings"""
        df_coll = pd.read_sql_query("SELECT * FROM collisions", engine)
        df_311 = pd.read_sql_query("SELECT * FROM requetes_311", engine)
        return df_coll, df_311
    
    df_coll_full, df_311_full = load_full_datasets()
    
    # Sélection du type de briefing
    c1, c2 = st.columns([1.5, 1])
    with c1:
        briefing_type = st.radio("Type de briefing", options=['daily', 'weekly', 'monthly'], 
                                horizontal=True, key='briefing_type')
    with c2:
        audience = st.radio("Audience", options=['Grand Public', 'Municipalité'], 
                           horizontal=True, key='briefing_audience',
                           label_visibility="collapsed")
    
    audience_map = {'Grand Public': 'public', 'Municipalité': 'municipality'}
    
    # Bouton pour générer
    if st.button("📊 Générer le Briefing", key="gen_briefing", use_container_width=True):
        with st.spinner("Génération du briefing en cours..."):
            try:
                briefing_content = generate_briefing(
                    df_coll_full,
                    df_311_full,
                    briefing_type=briefing_type,
                    target_audience=audience_map[audience]
                )
                
                # Affichage du briefing
                st.markdown(briefing_content)
                
                # Options d'export
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Télécharger en Markdown",
                        data=briefing_content,
                        file_name=f"briefing_{briefing_type}_{audience_map[audience]}.md",
                        mime="text/markdown",
                        key="dl_md"
                    )
                
                with col2:
                    st.success("✅ Briefing généré avec succès!")
                    
            except Exception as e:
                st.error(f"Erreur lors de la génération : {str(e)}")
                import traceback
                st.write(traceback.format_exc())

st.caption("Hackathon IA 2026 - Prototype fonctionnel")