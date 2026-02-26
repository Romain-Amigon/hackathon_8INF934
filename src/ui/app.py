import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import time
import sys
import os
import asyncio
from pathlib import Path

# ============================================================
# 0) GESTION DES CHEMINS (ROOT)
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Importations depuis ton architecture agentique
from src.agent.main import setup_agent, run, setup_retriever
from src.agent.graph import create_graph
from src.reports import generate_briefing

try:
    from sentence import Sentence
except ImportError:
    st.warning("Module 'sentence' non trouvé. La transcription vocale pourrait ne pas fonctionner.")
    Sentence = None

# ============================================================
# 1) CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(page_title="Mobility Copilot - Hackathon 2026", layout="wide")
st.title("Mobility Copilot : Analyse Intelligente")

# ============================================================
# 2) CHARGEMENT DE L'AGENT ET DES DONNÉES EN MÉMOIRE (PANDAS)
# ============================================================
@st.cache_resource
def get_agent_resources():
    # On charge l'agent, les moteurs (et donc les DataFrames) et le RAG
    agent_instance, engines = setup_agent()
    retriever = setup_retriever()
    app_graph = create_graph(agent_instance, engines, retriever)
    return app_graph, engines

app_graph, engines = get_agent_resources()

# On extrait les DataFrames purs pour l'affichage Streamlit
df_311 = engines["311"]._df
df_coll = engines["coll"]._df
df_meteo = engines["meteo"]._df

# ============================================================
# 3) ACCÈS AUX DONNÉES (RÉÉCRIT POUR PANDAS DIRECT)
# ============================================================
@st.cache_data(show_spinner=False)
def load_collisions_points(limit: int = 200_000) -> pd.DataFrame:
    df = df_coll.copy()
    
    # Nettoyage des coordonnées
    df["LOC_LAT"] = pd.to_numeric(df.get("LOC_LAT"), errors="coerce")
    df["LOC_LONG"] = pd.to_numeric(df.get("LOC_LONG"), errors="coerce")
    df = df.dropna(subset=["LOC_LAT", "LOC_LONG"])
    
    # Formatage de la date
    df["DT_ACCDN_DT"] = pd.to_datetime(df["DATE"], format='mixed', errors="coerce")
    df = df.dropna(subset=["DT_ACCDN_DT"])
    df["DT_ACCDN_STR"] = df["DT_ACCDN_DT"].dt.strftime("%Y-%m-%d")
    
    # Gravité
    df["GRAVITE"] = df["GRAVITE"].astype(str).str.strip()
    df.loc[df["GRAVITE"].isin(["", "None", "nan"]), "GRAVITE"] = "Inconnue"
    
    return df.head(limit)

@st.cache_data(show_spinner=False)
def load_311_arrondissements() -> list[str]:
    if "ARRONDISSEMENT" not in df_311.columns:
        return ["Tous"]
    arrs = df_311["ARRONDISSEMENT"].dropna().unique().tolist()
    return ["Tous"] + sorted([str(a).strip() for a in arrs if str(a).strip() != ''])

@st.cache_data(show_spinner=False)
def load_311_agg(date_start, date_end, motif_col: str, arrondissement: str, limit: int = 300) -> pd.DataFrame:
    df = df_311.copy()
    df['DATE_DT'] = pd.to_datetime(df.get("DATE", df.get("DATE")), format='mixed', errors='coerce')
    df = df.dropna(subset=['DATE_DT'])
    
    # Filtre de dates
    d1 = pd.to_datetime(date_start).date()
    d2 = pd.to_datetime(date_end).date()
    mask = (df['DATE_DT'].dt.date >= d1) & (df['DATE_DT'].dt.date <= d2)
    
    if arrondissement != 'Tous' and "ARRONDISSEMENT" in df.columns:
        mask &= (df['ARRONDISSEMENT'] == arrondissement)
        
    df_filtered = df[mask]
    
    # Agrégation
    agg = df_filtered[motif_col].fillna("Inconnue").value_counts().reset_index()
    agg.columns = ['motif', 'cnt']
    return agg.head(limit)

@st.cache_data(show_spinner=False)
def load_daily_metrics(date_start, date_end) -> pd.DataFrame:
    d1 = pd.to_datetime(date_start).date()
    d2 = pd.to_datetime(date_end).date()
    
    # Météo
    df_m = df_meteo.copy()
    df_m['date'] = pd.to_datetime(df_m['DATE'], format='mixed', errors='coerce').dt.date
    df_m = df_m[(df_m['date'] >= d1) & (df_m['date'] <= d2)]
    
    # Collisions
    df_c = df_coll.copy()
    df_c['date'] = pd.to_datetime(df_c['DATE'], format='mixed', errors='coerce').dt.date
    df_c = df_c[(df_c['date'] >= d1) & (df_c['date'] <= d2)]
    c_counts = df_c.groupby('date').size().reset_index(name='collisions_count')
    
    # 311
    df_3 = df_311.copy()
    df_3['date'] = pd.to_datetime(df_3.get("DATE", df_3.get("DATE")), format='mixed', errors='coerce').dt.date
    df_3 = df_3[(df_3['date'] >= d1) & (df_3['date'] <= d2)]
    r_counts = df_3.groupby('date').size().reset_index(name='req311_count')
    
    # Fusion
    daily = df_m.merge(c_counts, on='date', how='left').merge(r_counts, on='date', how='left')
    daily['collisions_count'] = daily['collisions_count'].fillna(0)
    daily['req311_count'] = daily['req311_count'].fillna(0)
    
    return daily

def pearson(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 3: return None
    return float(a[m].corr(b[m], method="pearson"))


# ============================================================
# 4) INTERFACE — TABS
# ============================================================
tabs = st.tabs(["Collisions (Carte)", "311 (Motifs)", "Météo ↔ Incidents", "Assistant Analyste", "Copilote Vocal", "Briefing Automatique"])

# --- TAB 1: CARTE ---
with tabs[0]:
    st.subheader("Collisions — Carte & Heatmap")
    df_col_map = load_collisions_points()
    
    if not df_col_map.empty:
        min_date, max_date = df_col_map["DT_ACCDN_DT"].min().date(), df_col_map["DT_ACCDN_DT"].max().date()
        
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            date_range = st.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="coll_date_range")
        with c2:
            gravites = sorted(df_col_map["GRAVITE"].unique().tolist())
            grav_sel = st.multiselect("Gravité", options=gravites, default=gravites)
        with c3:
            max_points = st.slider("Nb de points max", 5000, 50000, 10000, step=5000)
            
        if len(date_range) == 2:
            d1, d2 = date_range
            df_map = df_col_map[
                (df_col_map["DT_ACCDN_DT"].dt.date >= d1) & 
                (df_col_map["DT_ACCDN_DT"].dt.date <= d2) & 
                (df_col_map["GRAVITE"].isin(grav_sel))
            ].copy()
            
            if len(df_map) > max_points:
                df_map = df_map.sample(n=max_points, random_state=42)
                
            if not df_map.empty:
                center_lat, center_lon = df_map["LOC_LAT"].mean(), df_map["LOC_LONG"].mean()
                
                heat_layer = pdk.Layer("HeatmapLayer", data=df_map, get_position="[LOC_LONG, LOC_LAT]", radiusPixels=60)
                points_layer = pdk.Layer("ScatterplotLayer", data=df_map, get_position="[LOC_LONG, LOC_LAT]", get_radius=25, pickable=True, auto_highlight=True)
                
                tooltip = {"html": "<b>Date:</b> {DT_ACCDN_STR}<br/><b>Gravité:</b> {GRAVITE}", "style": {"backgroundColor": "white", "color": "black"}}
                deck = pdk.Deck(map_style="mapbox://styles/mapbox/light-v10", initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10), layers=[heat_layer, points_layer], tooltip=tooltip)
                st.pydeck_chart(deck, use_container_width=True)
            else:
                st.warning("Aucune donnée pour cette sélection.")
    else:
        st.error("Impossible de charger les données de collisions géolocalisées.")

# --- TAB 2: 311 MOTIFS ---
with tabs[1]:
    st.subheader("311 — Motifs & Signalements")
    
    d_start, d_end = st.session_state.get("coll_date_range", (pd.to_datetime('2021-01-01').date(), pd.to_datetime('2021-12-31').date()))
    if len(st.session_state.get("coll_date_range", [])) != 2:
        d_start, d_end = pd.to_datetime('2021-01-01').date(), pd.to_datetime('2021-12-31').date()

    arrs = load_311_arrondissements()
    c1, c2, c3 = st.columns([1.3, 1.0, 1.0])
    with c1:
        d_range_311 = st.date_input("Période (311)", value=(d_start, d_end), key="req311_date_range")
    with c2:
        arrondissement = st.selectbox("Arrondissement", options=arrs, index=0)
    with c3:
        motif_col = st.selectbox("Colonne motif", options=["ACTI_NOM", "NATURE"], index=0)
        
    if len(d_range_311) == 2:
        agg = load_311_agg(str(d_range_311[0]), str(d_range_311[1]), motif_col=motif_col, arrondissement=arrondissement)
        
        if not agg.empty:
            top10 = agg.head(10).sort_values("cnt", ascending=True)
            fig_bar = px.bar(top10, x="cnt", y="motif", orientation='h', title=f"Top 10 motifs ({motif_col})", color="cnt")
            fig_bar.update_layout(template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            freq = dict(zip(agg['motif'], agg['cnt']))
            wc = WordCloud(width=1200, height=500, background_color="white", colormap="viridis").generate_from_frequencies(freq)
            img_buffer = io.BytesIO()
            wc.to_image().save(img_buffer, format="PNG")
            st.image(img_buffer, caption="Nuage de mots des signalements", use_container_width=True)
        else:
            st.warning("Aucune donnée 311 pour cette période.")

# --- TAB 3: METEO ---
with tabs[2]:
    st.subheader("Corrélations Météo ↔ Incidents")
    
    d_start, d_end = st.session_state.get("coll_date_range", (pd.to_datetime('2021-01-01').date(), pd.to_datetime('2021-12-31').date()))
    if len(st.session_state.get("coll_date_range", [])) != 2:
        d_start, d_end = pd.to_datetime('2021-01-01').date(), pd.to_datetime('2021-12-31').date()
        
    d_range_met = st.date_input("Période", value=(d_start, d_end), key="met_date_range")
    
    if len(d_range_met) == 2:
        daily = load_daily_metrics(str(d_range_met[0]), str(d_range_met[1]))
        
        if not daily.empty:
            k1, k2, k3 = st.columns(3)
            k1.metric("Jours couverts", f"{len(daily):,}")
            k2.metric("Collisions (total)", f"{int(daily['collisions_count'].sum()):,}")
            k3.metric("311 (total)", f"{int(daily['req311_count'].sum()):,}")
            
            st.line_chart(daily.set_index("date")[["collisions_count", "req311_count", "precipitation_sum", "snowfall_sum"]])
            
            st.subheader("Comparaison : Neige vs Temps Sec")
            snow_days = daily["snowfall_sum"].fillna(0) > 0
            
            def safe_mean(mask, col):
                return round(daily.loc[mask, col].mean(), 1) if mask.sum() > 0 else 0
                
            comp = pd.DataFrame([
                ["Collisions", safe_mean(snow_days, "collisions_count"), safe_mean(~snow_days, "collisions_count")],
                ["311", safe_mean(snow_days, "req311_count"), safe_mean(~snow_days, "req311_count")],
            ], columns=["Incident", "Moyenne (Jours de Neige)", "Moyenne (Sans Neige)"])
            st.dataframe(comp, use_container_width=True)

# --- TAB 4: ASSISTANT LANGGRAPH ---
with tabs[3]:
    st.subheader("Assistant Analyste (LangGraph)")
    question = st.text_input("Posez votre question complexe sur les données :")
    
    if st.button("Lancer l'Analyse", type="primary"):
        if question:
            with st.spinner("L'agent génère et exécute le code Pandas..."):
                try:
                    # On appelle le graphe LangGraph via sa fonction run native
                    response_text, reflexions = asyncio.run(run(app_graph, question))
                    
                    st.success("Analyse terminée")
                    st.markdown(f"**Réponse :**\n{response_text}")
                    
                    with st.expander("Voir les réflexions et le code exécuté"):
                        for ref in reflexions:
                            st.text(ref)
                except Exception as e:
                    st.error(f"Erreur d'exécution du graphe : {e}")

# --- TAB 5: VOCAL ---
with tabs[4]:
    st.header("Copilote Vocal")
    if Sentence is None:
        st.error("Le module audio n'est pas installé.")
    else:
        voice_manager = Sentence(language='fr-FR')
        uploaded_audio = st.file_uploader("Déposez votre question audio (.wav, .flac)", type=["wav", "flac"])
        
        if uploaded_audio:
            with st.status("Traitement vocal...", expanded=True) as status:
                temp_path = "temp_vocal_auto.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                    
                transcript = voice_manager.toText(temp_path)
                st.write(f"**Transcription:** {transcript}")
                
                if transcript and "[Error]" not in transcript:
                    try:
                        final_response, _ = asyncio.run(run(app_graph, transcript))
                        st.markdown(f"**Réponse :** {final_response}")
                        status.update(label="✅ Terminé !", state="complete")
                    except Exception as e:
                        st.error(f"Erreur Agent: {e}")

# --- TAB 6: BRIEFINGS ---
with tabs[5]:
    st.subheader("Briefings Automatiques")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        briefing_type = st.radio("Type", options=['daily', 'weekly', 'monthly'], horizontal=True)
    with c2:
        audience = st.radio("Audience", options=['Grand Public', 'Municipalité'], horizontal=True)
        
    audience_map = {'Grand Public': 'public', 'Municipalité': 'municipality'}
    
    if st.button("📊 Générer le Briefing"):
        with st.spinner("Rédaction du briefing en cours..."):
            try:
                # On passe directement les DataFrames chargés en mémoire
                briefing_content = generate_briefing(
                    df_coll,
                    df_311,
                    briefing_type=briefing_type,
                    target_audience=audience_map[audience]
                )
                st.markdown(briefing_content)
            except Exception as e:
                st.error(f"Erreur lors de la génération : {str(e)}")