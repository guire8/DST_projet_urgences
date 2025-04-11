import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
from utils.preprocessing import (
    preprocess_common,
    preprocess_for_attente,
    preprocess_for_hospitalisation,
    preprocess_for_moyenne
)

st.set_page_config(page_title="Projet Urgences", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("data/liste_sejours_new.xlsx")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df_raw = load_data()
df_cleaned = preprocess_common(df_raw.copy())

# -------------------- ONGLET 1 : DATAVIZ --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualisation", "🤖 Temps d'attente", "🏥 Hospitalisation", "📈 Estimation moyenne"])

with tab1:
    st.title("📊 Analyse exploratoire des données urgences")
    st.subheader("1. Distribution des variables clés")

    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(df_cleaned, x="Age_Moyen_Sejour_Annees", nbins=50, title="Distribution de l'âge")
        st.plotly_chart(fig_age)
    with col2:
        fig_ioa = px.histogram(df_cleaned, x="Tri_IOA", title="Répartition des niveaux IOA")
        st.plotly_chart(fig_ioa)

    fig_heatmap = df_cleaned.groupby(["Jour", "Heure_Entree"]).size().reset_index(name="count")
    fig_heatmap = px.density_heatmap(
        fig_heatmap,
        x="Heure_Entree", y="Jour",
        z="count", color_continuous_scale="Viridis",
        title="Carte thermique des arrivées par heure et jour"
    )
    st.plotly_chart(fig_heatmap)

    st.subheader("2. Analyse des valeurs manquantes")
    missing = df_raw.isnull().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["Colonne", "Taux de valeurs manquantes"]
    fig_missing = px.bar(missing, x="Colonne", y="Taux de valeurs manquantes", title="Taux de valeurs manquantes")
    st.plotly_chart(fig_missing)

# -------------------- FORMULAIRE UTILISATEUR --------------------

def formulaire(df_base, form_key_prefix=""):
    now = datetime.now()
    date_entree = st.date_input("Date d'entrée", now.date(), key=f"{form_key_prefix}_date_entree")
    heure_entree = st.time_input("Heure d'entrée", now.time(), key=f"{form_key_prefix}_heure_entree")
    datetime_entree = datetime.combine(date_entree, heure_entree)

    age = st.number_input("Âge du patient", min_value=0, max_value=110, value=35, key=f"{form_key_prefix}_age")

    motif = st.selectbox("Motif de recours", sorted(df_base["Motif_de_recours"].dropna().unique()), key=f"{form_key_prefix}_motif")
    date_ioa = st.date_input("Date PEC IOA", now.date(), key=f"{form_key_prefix}_date_ioa")
    heure_ioa = st.time_input("Heure PEC IOA", now.time(), key=f"{form_key_prefix}_heure_ioa")
    datetime_ioa = datetime.combine(date_ioa, heure_ioa)

    discipline = st.selectbox("Discipline d'examen", sorted(df_base["Discipline_Examen"].dropna().unique()), key=f"{form_key_prefix}_discipline")
    type_pec = st.selectbox("Type de PEC", sorted(df_base["Type_de_PEC"].dropna().unique()), key=f"{form_key_prefix}_type_pec")
    tri_ioa = st.selectbox("Tri IOA", sorted(df_base["Tri_IOA"].dropna().unique()), key=f"{form_key_prefix}_tri_ioa")

    nb_present = st.number_input("Nombre de patients présents", min_value=0, value=np.random.randint(20, 80), key=f"{form_key_prefix}_nb_present")
    salle_ioa = st.number_input("Patients en salle d'attente IOA", min_value=0, value=np.random.randint(1, 10), key=f"{form_key_prefix}_salle_ioa")
    salle_med = st.number_input("Patients en salle d'attente MED", min_value=0, value=np.random.randint(2, 15), key=f"{form_key_prefix}_salle_med")

    return {
        "Date_Heure_Entree_Sejour": datetime_entree,
        "AGE": age,
        "Motif_de_recours": motif,
        "Date_Heure_PEC_IOA": datetime_ioa,
        "Discipline_Examen": discipline,
        "Type_de_PEC": type_pec,
        "Tri_IOA": tri_ioa,
        "nombre_patients_present": nb_present,
        "Salle_attente_IOA": salle_ioa,
        "Salle_attente_MED": salle_med,
    }

# -------------------- ONGLET 2 : TEMPS D'ATTENTE --------------------
with tab2:
    st.title("⏱️ Prédiction du temps d'attente")
    input_data = formulaire(df_cleaned, form_key_prefix="attente")

    if st.button("Prédire (Temps d'attente)"):
        model = load_model("models/model_temps_attente.pkl")
        df_input = preprocess_for_attente(df_raw.copy(), input_data)
        pred = model.predict(df_input)[0]
        st.success(f"Temps d'attente estimé : {round(pred, 1)} minutes")

# -------------------- ONGLET 3 : HOSPITALISATION --------------------
with tab3:
    st.title("🏥 Prédiction du risque d'hospitalisation")
    input_data = formulaire(df_cleaned, form_key_prefix="hospit")

    if st.button("Prédire (Hospitalisation)"):
        model = load_model("models/model_hospit.pkl")
        df_input = preprocess_for_hospitalisation(df_raw.copy(), input_data)
        proba = model.predict_proba(df_input)[0][1]
        st.info(f"Probabilité d'hospitalisation : {round(proba * 100, 1)} %")

# -------------------- ONGLET 4 : ESTIMATION MOYENNE --------------------
with tab4:
    st.title("📈 Estimation simple par moyenne")
    moyenne = preprocess_for_moyenne(df_raw)
    st.metric("Durée moyenne de passage aux urgences", f"{round(moyenne, 1)} minutes")