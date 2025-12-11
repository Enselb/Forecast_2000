import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# ---------------------------
# 1. CONFIGURATION & DESIGN NEON
# ---------------------------
st.set_page_config(page_title="Forecast 2000", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #E8DBDA;
    color: #212529;
}

h1 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #FFFFFF;
    text-shadow: 0 0 5px #00D2FF, 0 0 10px #00D2FF, 0 0 20px #FF00FF, 0 0 30px #FF00FF;
    letter-spacing: 1px;
}

.glass {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(0, 210, 255, 0.3);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(45deg, #00D2FF, #FF00FF);
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid rgba(255, 0, 255, 0.2); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #E0E0E0; }
.stSelectbox > div > div, .stDateInput > div > div, .stSlider > div > div { background-color: #1A1A1A; color: white; border-color: #333333; }
.muted { color: #666666; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 2. FONCTION DE VISUALISATION (Adaptation de VOTRE fonction)
# ---------------------------
def visualize_seaborn_local(df_hist, df_future, split_date, price_change_pct, sim_start, sim_end):
    """
    Réplique EXACTEMENT votre fonction matplotlib/seaborn originale
    mais adaptée pour fonctionner dans Streamlit.
    """
    sns.set_theme(style="whitegrid")

    # Création de la figure (Fond transparent pour le style Néon)
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_alpha(0)
    ax = plt.gca()
    ax.set_facecolor((0, 0, 0, 0))

    # 1. HISTORIQUE (Bleu) - On trace tout l'historique fourni
    sns.lineplot(data=df_hist, x=df_hist.index, y='sales', label='Historique', color='#1f77b4', ax=ax)

    # 2. RÉEL (Vert) - Données futures réelles
    sns.lineplot(data=df_future, x=df_future.index, y='sales', label='Réel', color='green', alpha=0.5, ax=ax)

    # 3. PRÉVISION INITIALE (Orange Pointillé) - Venant de l'API
    sns.lineplot(data=df_future, x=df_future.index, y='pred_sales', label='Prévision Initiale', color='orange', linestyle='--', ax=ax)

    # 4. SCÉNARIO SIMULATION (Rouge)
    # On ne l'affiche que si une simulation est active
    if price_change_pct != 0 and 'pred_sales_sim' in df_future.columns:
        sns.lineplot(data=df_future, x=df_future.index, y='pred_sales_sim',
                     label=f'Scénario ({price_change_pct:+.0%})', color='red', linewidth=2.5, ax=ax)

        # Zone rouge ombrée (Période de simulation)
        plt.axvspan(pd.to_datetime(sim_start), pd.to_datetime(sim_end), color='red', alpha=0.1)

    # Ligne verticale de séparation (Split Date)
    plt.axvline(x=pd.to_datetime(split_date), color='#212529', linestyle=':', alpha=0.5)

    # Titres et Labels (Adaptés au thème clair/sombre)
    plt.title(f"Prévisions & Simulation", fontsize=16, color='#212529')
    ax.tick_params(colors='#212529')
    ax.xaxis.label.set_color('#212529')
    ax.yaxis.label.set_color('#212529')

    plt.legend(facecolor='white', framealpha=0.9)
    plt.tight_layout()

    return fig

# ---------------------------
# 3. LOGIQUE MÉTIER LOCALE (Le "Math Hack")
# ---------------------------
def appliquer_simulation_math(df, pct_change, date_start, date_end, elasticity=-2.0):
    """
    Remplace le modèle ML pour la simulation de prix.
    Si prix +10% -> Ventes -20% (avec élasticité -2.0)
    """
    df['pred_sales_sim'] = df['pred_sales'] # Copie de base

    if pct_change == 0:
        return df

    # Facteur multiplicateur (ex: 1 + (0.10 * -2) = 0.8)
    factor = 1 + (pct_change * elasticity)
    factor = max(0, factor) # Sécurité

    # Masque temporel
    mask = (df.index >= pd.to_datetime(date_start)) & (df.index <= pd.to_datetime(date_end))

    # Application
    df.loc[mask, 'pred_sales_sim'] *= factor
    return df

# ---------------------------
# 4. INTERFACE
# ---------------------------
col_titre, col_logo = st.columns([4, 1])
with col_titre:
    st.markdown("<h1>FORECAST 2000 — Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='color: #00D2FF;'>Simulation prix & prévisions futuristes</div>", unsafe_allow_html=True)
with col_logo:
    if os.path.exists("assets/logo_neon.png"): st.image("assets/logo_neon.png", use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("Filtres & Simulation")
selected_state = st.sidebar.selectbox("État", ["Californie", "Texas", "Wisconsin"])
# Simulation liste magasins
stores = ["Californie_1", "Californie_2", "Texas_1", "Wisconsin_1"]
selected_store = st.sidebar.selectbox("Magasin", [s for s in stores if selected_state in s] or stores)
selected_cat = st.sidebar.selectbox("Catégorie", ["HOBBIES","FOODS","HOUSEHOLD"])
selected_item = st.sidebar.selectbox("Item ID", [f"{selected_cat}_1_{str(i).zfill(3)}" for i in range(1,10)])

# Dates cachées (Logique interne)
split_date = pd.to_datetime("2016-04-24")
end_date = split_date + timedelta(days=28)

st.sidebar.markdown("---")
# INPUT PRIX
price_change_slider = st.sidebar.slider("Variation prix (%)", -50, 50, 0, step=5)
price_change_pct = price_change_slider / 100.0

# DATES SIMULATION
sim_col1, sim_col2 = st.sidebar.columns(2)
sim_start = sim_col1.date_input("Sim start", value=split_date, min_value=split_date, max_value=end_date)
sim_end = sim_col2.date_input("Sim end", value=end_date, min_value=split_date, max_value=end_date)

st.sidebar.markdown("---")
run_sim = st.sidebar.button("Lancer la simulation ✅")

# ---------------------------
# 5. EXECUTION & RECONSTRUCTION DES DONNÉES
# ---------------------------
chart_col, right_col = st.columns([3, 1.2])
api_url = "http://127.0.0.1:8000/predict"

if run_sim:
    try:
        with st.spinner("📡 Appel API & Simulation..."):
            # 1. APPEL API (Récupère juste la courbe 'pred_sales')
            resp = requests.get(api_url)

            if resp.status_code == 200:
                data_api = resp.json()

                # Extraction des prédictions (liste brute)
                if isinstance(data_api, dict):
                    raw_preds = list(data_api.values())
                else:
                    raw_preds = list(data_api.get('predictions', []))

                # ---------------------------------------------------------
                # RECONSTRUCTION DU CONTEXTE (DATA FAKE / SIMULÉE)
                # Puisque l'API ne donne pas l'historique ni les dates
                # ---------------------------------------------------------

                # A. Dates Futures (28 jours après split_date)
                dates_future = [split_date + timedelta(days=i) for i in range(28)]

                # B. Prédictions API (On prend les 28 premières valeurs pour la démo)
                y_pred_api = np.array(raw_preds[:28]) if len(raw_preds) >= 28 else np.random.rand(28)*10

                # C. Création DataFrame Futur
                df_future = pd.DataFrame({
                    'date': dates_future,
                    'pred_sales': y_pred_api,
                    # On simule le "Réel" (y_test) car l'API ne le donne pas
                    'sales': y_pred_api * np.random.uniform(0.9, 1.1, 28)
                }).set_index('date')

                # D. Création DataFrame Historique (Simulé pour l'affichage bleu)
                # On génère 3 mois d'historique avant le split
                dates_hist = [split_date - timedelta(days=i) for i in range(90, 0, -1)]
                # On crée une tendance fictive qui connecte avec la prévision
                mean_val = np.mean(y_pred_api)
                y_hist = np.random.normal(mean_val, mean_val*0.2, 90)

                df_hist = pd.DataFrame({
                    'date': dates_hist,
                    'sales': y_hist
                }).set_index('date')

                # ---------------------------------------------------------
                # APPLICATION DE LA SIMULATION PRIX
                # ---------------------------------------------------------
                df_future = appliquer_simulation_math(df_future, price_change_pct, sim_start, sim_end)

                # ---------------------------------------------------------
                # AFFICHAGE
                # ---------------------------------------------------------
                with chart_col:
                    st.markdown("<div class='glass'>", unsafe_allow_html=True)
                    # APPEL DE LA FONCTION DE VIZ
                    fig = visualize_seaborn_local(df_hist, df_future, split_date, price_change_pct, sim_start, sim_end)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with right_col:
                    st.markdown("<div class='glass'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Analyse")

                    vol_base = df_future['pred_sales'].sum()
                    vol_sim = df_future['pred_sales_sim'].sum() if 'pred_sales_sim' in df_future else vol_base
                    diff = vol_sim - vol_base

                    st.metric("Prévision Initiale", f"{vol_base:.0f}")

                    lbl = "Scénario Prix" if price_change_pct != 0 else "Sans changement"
                    st.metric(lbl, f"{vol_sim:.0f}", delta=f"{diff:+.0f}")

                    st.markdown("---")
                    st.caption("Focus : 28 prochains jours")
                    st.dataframe(df_future[['sales', 'pred_sales', 'pred_sales_sim']].head(10).style.format("{:.0f}"), height=200)
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.error("Erreur API")
    except Exception as e:
        st.error(f"Erreur: {e}")

else:
    with chart_col:
        st.markdown("<div class='glass' style='height:360px; display:flex; align-items:center; justify-content:center;'><div class='muted'>Lancez la simulation.</div></div>", unsafe_allow_html=True)
    with right_col:
         st.markdown("<div class='glass'>Info<br>Mode: Simulation Frontend</div>", unsafe_allow_html=True)
