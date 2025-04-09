import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from utils.preprocessing import transformer_donnees_utilisateur

st.set_page_config(page_title="Projet Urgences", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("data/liste_sejours_new.xlsx")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df_base = load_data()

model_attente = load_model("models/model_temps_attente.pkl")
model_hospit = load_model("models/model_hospit.pkl")

st.title("🚑 Plateforme Urgences - Prédictions ML")

tabs = st.tabs(["Temps d'attente", "Estimation par moyenne", "Risque d'hospitalisation"])

# Onglet 1 : Temps d'attente
from PIL import Image
# Charger une image avec PIL
image = Image.open("chemin/vers/votre/image.jpg")

# Afficher l'image dans Streamlit
st.image(image, caption="Image chargée avec PIL", use_column_width=True)

with tabs[0]:
    st.header("⏱️ Prédiction du temps d'attente")

    with st.form("form_attente"):
        now = datetime.now()
        datetime_entree = datetime.combine(st.date_input("Date entrée", now.date()),
                                           st.time_input("Heure entrée", now.time()))
        age = st.number_input("Âge du patient", 0, 110, 35)
        datetime_ioa = datetime.combine(st.date_input("Date PEC IOA", now.date()),
                                        st.time_input("Heure PEC IOA", now.time()))
        motif = st.selectbox("Motif de recours", df_base["Motif_de_recours"].dropna().unique())
        type_pec = st.selectbox("Type de PEC", df_base["Type_de_PEC"].dropna().unique())
        tri_ioa = st.selectbox("Tri IOA", sorted(df_base["Tri_IOA"].dropna().unique()))

        if st.form_submit_button("Prédire"):
            df_user = pd.DataFrame([{
                "Date_Heure_Entree_Sejour": datetime_entree,
                "Date_Heure_PEC_IOA": datetime_ioa,
                "AGE": age,
                "Motif_de_recours": motif,
                "Type_de_PEC": type_pec,
                "Tri_IOA": tri_ioa,
            }])
            df_user_transforme = transformer_donnees_utilisateur(df_user.copy(), df_base)
            pred = model_attente.predict(df_user_transforme)[0]
            st.success(f"Temps d'attente estimé : {round(pred, 1)} minutes")

# Onglet 2 : Moyenne
with tabs[1]:
    st.header("📊 Estimation simple par moyenne")
    moyenne = df_base["Temps_Attente"].mean() if "Temps_Attente" in df_base.columns else 97.2
    st.metric("Temps d'attente moyen (base)", f"{round(moyenne, 1)} minutes")

# Onglet 3 : Hospitalisation
with tabs[2]:
    st.header("🏥 Prédiction du risque d'hospitalisation")

    with st.form("form_hospit"):
        datetime_entree = datetime.combine(st.date_input("Date entrée (Hospit)", now.date()),
                                           st.time_input("Heure entrée (Hospit)", now.time()))
        age = st.number_input("Âge (Hospit)", 0, 110, 35)
        datetime_ioa = datetime.combine(st.date_input("Date PEC IOA (Hospit)", now.date()),
                                        st.time_input("Heure PEC IOA (Hospit)", now.time()))
        motif = st.selectbox("Motif de recours (Hospit)", df_base["Motif_de_recours"].dropna().unique())
        type_pec = st.selectbox("Type de PEC (Hospit)", df_base["Type_de_PEC"].dropna().unique())
        tri_ioa = st.selectbox("Tri IOA (Hospit)", sorted(df_base["Tri_IOA"].dropna().unique()))

        if st.form_submit_button("Prédire hospitalisation"):
            df_user = pd.DataFrame([{
                "Date_Heure_Entree_Sejour": datetime_entree,
                "Date_Heure_PEC_IOA": datetime_ioa,
                "AGE": age,
                "Motif_de_recours": motif,
                "Type_de_PEC": type_pec,
                "Tri_IOA": tri_ioa,
            }])
            df_user_transforme = transformer_donnees_utilisateur(df_user.copy(), df_base)
            proba = model_hospit.predict_proba(df_user_transforme)[0][1]
            st.info(f"Probabilité d'hospitalisation : {round(proba * 100, 1)} %")
