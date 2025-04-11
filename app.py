import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from random import randint
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

# Initialiser les valeurs aléatoires une seule fois
if "nb_present" not in st.session_state:
    st.session_state["nb_present"] = np.random.randint(20, 80)
if "salle_ioa" not in st.session_state:
    st.session_state["salle_ioa"] = np.random.randint(1, 10)
if "salle_med" not in st.session_state:
    st.session_state["salle_med"] = np.random.randint(2, 15)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualisation", "🤖 Temps de passage", "🏥 Hospitalisation", "📈 Estimation moyenne"])

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

def formulaire(df_base, form_key_prefix=""):
    date_entree = st.date_input("Date d'entrée", key=f"{form_key_prefix}_date_entree")
    heure_entree = st.time_input("Heure d'entrée", key=f"{form_key_prefix}_heure_entree")
    datetime_entree = datetime.combine(date_entree, heure_entree)

    age = st.number_input("Âge du patient", min_value=0, max_value=110, value=35, key=f"{form_key_prefix}_age")

    # Récupération des motifs depuis le fichier initial
    motifs = df_raw["Motif de recours"].dropna().unique()
    motifs = sorted([m.strip() for m in motifs if m.strip() != "#VALEURMULTI"])
    motif = st.selectbox("Motif de recours", motifs, key=f"{form_key_prefix}_motif")

    # PEC IOA avec valeur actuelle par défaut
    now = datetime.now()
    date_ioa = st.date_input("Date PEC IOA", now.date(), key=f"{form_key_prefix}_date_ioa")
    heure_ioa = st.time_input("Heure PEC IOA", now.time(), key=f"{form_key_prefix}_heure_ioa")
    datetime_ioa = datetime.combine(date_ioa, heure_ioa)

    discipline = st.selectbox("Discipline d'examen", sorted(df_base["Discipline_Examen"].dropna().unique()), key=f"{form_key_prefix}_discipline")
    type_pec = st.selectbox("Type de PEC", sorted(df_base["Type_de_PEC"].dropna().unique()), key=f"{form_key_prefix}_type_pec")
    tri_ioa = st.selectbox("Tri IOA", sorted(df_base["Tri_IOA"].dropna().unique()), key=f"{form_key_prefix}_tri_ioa")

    nb_present = st.number_input("Nombre de patients présents", min_value=0, value=st.session_state["nb_present"], key=f"{form_key_prefix}_nb_present")
    salle_ioa = st.number_input("Patients en salle d'attente IOA", min_value=0, value=st.session_state["salle_ioa"], key=f"{form_key_prefix}_salle_ioa")
    salle_med = st.number_input("Patients en salle d'attente MED", min_value=0, value=st.session_state["salle_med"], key=f"{form_key_prefix}_salle_med")

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

with tab2:
    st.title("⏱️ Prédiction du temps de passage total")

    with st.form("form_temps_attente"):
        col1, col2 = st.columns(2)
        now = datetime.now()

        with col1:
            date_entree = st.date_input("Date d'entrée", value=now.date(), key="attente_date")
            heure_defaut = (now - timedelta(minutes=20)).time()
            heure_entree = st.time_input("Heure d'entrée", value=heure_defaut, key="attente_heure")
            age = st.number_input("Âge du patient", min_value=0, max_value=110, value=35, key="attente_age")
            nb_present = st.number_input("Patients présents aux urgences", min_value=0, max_value=200, value=randint(20, 80), key="attente_present")
            nb_ioa = st.number_input("En salle d'attente IOA", min_value=0, max_value=50, value=randint(1, 10), key="attente_ioa")
            nb_med = st.number_input("En salle d'attente Médecin", min_value=0, max_value=50, value=randint(2, 15), key="attente_med")

        with col2:
            date_ioa = st.date_input("Date PEC IOA", now.date(), key="attente_date_ioa")
            heure_ioa = st.time_input("Heure PEC IOA", now.time(), key="attente_heure_ioa")
            motif = st.selectbox("Motif de recours", df_raw["Motif de recours"].dropna().unique(), key="attente_motif")
            type_pec = st.selectbox("Type de PEC", df_cleaned["Type_de_PEC"].dropna().unique(), key="attente_pec")
            tri_ioa = st.selectbox("Tri IOA", sorted(df_cleaned["Tri_IOA"].dropna().unique()), key="attente_tri")
            discipline = st.selectbox("Discipline Examen", df_cleaned["Discipline_Examen"].dropna().unique(), key="attente_discipline")

        datetime_entree = datetime.combine(date_entree, heure_entree)
        datetime_ioa = datetime.combine(date_ioa, heure_ioa)

        input_data = {
            "Date_Heure_Entree_Sejour": datetime_entree,
            "Date_Heure_PEC_IOA": datetime_ioa,
            "AGE": age,
            "Motif_de_recours": motif,
            "Type_de_PEC": type_pec,
            "Tri_IOA": tri_ioa,
            "Discipline_Examen": discipline,
        }

        submitted = st.form_submit_button("Prédire (Temps avant sortie)")

    if submitted:
        model = load_model("models/model_temps_attente.pkl")
        df_input = preprocess_for_attente(df_raw.copy(), input_data)
        pred = model.predict(df_input)[0]
        st.success(f"Temps total estimé avant la sortie: {round(pred, 1)} heures")



with tab3:
    st.title("🏥 Prédiction du risque d'hospitalisation")

    with st.form("form_hospitalisation"):
        col1, col2 = st.columns(2)
        now = datetime.now()
        heure_defaut = (now - timedelta(minutes=20)).time()

        with col1:
            date_entree = st.date_input("Date d'entrée", value=now.date(), key="hospit_date")
            heure_entree = st.time_input("Heure d'entrée", value=heure_defaut, key="hospit_heure")
            age = st.number_input("Âge du patient", min_value=0, max_value=110, value=35, key="hospit_age")
            nb_present = st.number_input("Patients présents aux urgences", min_value=0, max_value=200, value=randint(20, 80), key="hospit_present")
            nb_ioa = st.number_input("En salle d'attente IOA", min_value=0, max_value=50, value=randint(1, 10), key="hospit_ioa")
            nb_med = st.number_input("En salle d'attente Médecin", min_value=0, max_value=50, value=randint(2, 15), key="hospit_med")

        with col2:
            date_ioa = st.date_input("Date PEC IOA", now.date(), key="hospit_date_ioa")
            heure_ioa = st.time_input("Heure PEC IOA", now.time(), key="hospit_heure_ioa")
            motif = st.selectbox("Motif de recours", df_raw["Motif de recours"].dropna().unique(), key="hospit_motif")
            type_pec = st.selectbox("Type de PEC", df_cleaned["Type_de_PEC"].dropna().unique(), key="hospit_pec")
            tri_ioa = st.selectbox("Tri IOA", sorted(df_cleaned["Tri_IOA"].dropna().unique()), key="hospit_tri")
            discipline = st.selectbox("Discipline Examen", df_cleaned["Discipline_Examen"].dropna().unique(), key="hospit_discipline")

        datetime_entree = datetime.combine(date_entree, heure_entree)
        datetime_ioa = datetime.combine(date_ioa, heure_ioa)

        input_data = {
            "Date_Heure_Entree_Sejour": datetime_entree,
            "Date_Heure_PEC_IOA": datetime_ioa,
            "AGE": age,
            "Motif_de_recours": motif,
            "Type_de_PEC": type_pec,
            "Tri_IOA": tri_ioa,
            "Discipline_Examen": discipline,
            "nombre_patients_present": nb_present,
            "Salle_attente_IOA": nb_ioa,
            "Salle_attente_MED": nb_med,
        }

        submitted = st.form_submit_button("Prédire (Hospitalisation)")

    if submitted:
        model_pack = load_model("models/model_hospit.pkl")
        model = model_pack["model"]
        preprocessor = model_pack["preprocessor"]
        threshold = model_pack.get("manual_threshold", 0.32)

        df_input = preprocess_for_hospitalisation(df_raw.copy(), input_data)
        X = preprocessor.transform(df_input)
        proba = model.predict_proba(X)[0][1]

        # Jauge
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(proba * 100, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité d'hospitalisation", 'font': {'size': 20}},
            delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Message dynamique
        if proba < 0.3:
            color = "green"
            label = "Retour à domicile probable"
        elif proba < 0.7:
            color = "orange"
            label = "Hospitalisation possible"
        else:
            color = "red"
            label = "Hospitalisation probable"

        st.markdown(
            f"<div style='padding: 1rem; background-color: {color}; color: white; border-radius: 0.5rem; font-weight: bold;'>"
            f"🏥 Prédiction : {label}<br>Probabilité : {round(proba*100, 1)} % (seuil {threshold})"
            f"</div>",
            unsafe_allow_html=True
        )


with tab4:
    st.title("📈 Estimation simple par moyenne")

    df_moy = preprocess_common(df_raw.copy())

    import holidays
    fr_holidays = holidays.FR()
    df_moy["Date"] = df_moy["Date_Heure_Entree_Sejour"].dt.day
    df_moy["Mois"] = df_moy["Date_Heure_Entree_Sejour"].dt.month
    df_moy["jour_ferie"] = df_moy["Date_Heure_Entree_Sejour"].dt.date.isin(fr_holidays).astype(int)
    df_moy["Jour_Entree"] = df_moy["Date_Heure_Entree_Sejour"].dt.day_name(locale="fr_FR").str.capitalize()

    nb_jours = df_moy["Date_Heure_Entree_Sejour"].dt.date.nunique()

    now = datetime.now()
    jour_mapping = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }
    jour_defaut = jour_mapping[now.strftime("%A")]
    date_defaut = now.day
    mois_defaut = now.month
    heure_defaut = now.hour

    st.subheader("🧪 Estimation personnalisée")

    afficher_outliers = st.toggle("Afficher les outliers", value=False, key="toggle_outliers_tab4")

    col1, col2, col3, col4 = st.columns(4)
    jours_dispos = sorted(df_moy["Jour_Entree"].dropna().unique())
    dates_dispos = sorted(df_moy["Date"].dropna().unique())
    mois_dispos = sorted(df_moy["Mois"].dropna().unique())
    heures_dispos = sorted(df_moy["Heure_Entree"].dropna().unique())

    jour = col1.selectbox("Jour", jours_dispos, index=jours_dispos.index(jour_defaut))
    date = col2.selectbox("Date (n° jour)", dates_dispos, index=dates_dispos.index(date_defaut))
    mois = col3.selectbox("Mois", mois_dispos, index=mois_dispos.index(mois_defaut))
    heure = col4.selectbox("Heure", heures_dispos, index=heures_dispos.index(heure_defaut))

    def remove_outliers_iqr(df, columns, iqr_multiplier=3):
        df_filtered = df.copy()
        for col in columns:
            q1 = df_filtered[col].quantile(0.25)
            q3 = df_filtered[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + iqr_multiplier * iqr
            df_filtered = df_filtered[
                (df_filtered[col] >= 0) & (df_filtered[col] <= upper_bound)
            ]
        return df_filtered

    def calcul_moyennes_indicateurs(df, jour, date, mois, outliers=False):
        df_filtre = df[
            (df["Jour_Entree"] == jour) |
            (df["Date"] == date) |
            (df["Mois"] == mois) |
            (df["jour_ferie"] == 1)
        ]
        if not outliers:
            df_filtre = remove_outliers_iqr(df_filtre, [
                "Delai_entree_IOA_heure",
                "Delai_entree_MED_heure",
                "Duree_totale_heure"
            ])
        return {
            "IOA_moy": df_filtre["Delai_entree_IOA_heure"].mean(),
            "IOA_std": df_filtre["Delai_entree_IOA_heure"].std(),
            "MED_moy": df_filtre["Delai_entree_MED_heure"].mean(),
            "MED_std": df_filtre["Delai_entree_MED_heure"].std(),
            "TOT_moy": df_filtre["Duree_totale_heure"].mean(),
            "TOT_std": df_filtre["Duree_totale_heure"].std(),
            "filtered_df": df_filtre
        }

    res = calcul_moyennes_indicateurs(df_moy, jour, date, mois, afficher_outliers)

    col1, col2, col3 = st.columns(3)
    col1.metric("🩺 IOA", f"{int(res['IOA_moy']*60)} ± {int(res['IOA_std']*60)} min")
    col2.metric("👨‍⚕️ Médecin", f"{int(res['MED_moy']*60)} ± {int(res['MED_std']*60)} min")
    col3.metric("🚪 Sortie", f"{int(res['TOT_moy']*60)} ± {int(res['TOT_std']*60)} min")

    # Données affluence non filtrées
    df_affluence = df_moy.groupby("Heure_Entree").agg(
        Moy_arrivees=("Duree_totale_heure", "size")
    ).reset_index()
    df_affluence["Moy_arrivees"] = df_affluence["Moy_arrivees"] / nb_jours

    # Données durées filtrées
    df_graph = res["filtered_df"].dropna(subset=["Duree_totale_heure"])
    df_duree = df_graph.groupby("Heure_Entree").agg(
        Duree_moy_min=("Duree_totale_heure", lambda x: x.mean() * 60)
    ).reset_index()

    # Fusion
    df_plot = pd.merge(df_duree, df_affluence, on="Heure_Entree", how="left")
    df_plot["Heure_affichage"] = df_plot["Heure_Entree"].astype(int).astype(str) + "h-" + (df_plot["Heure_Entree"] + 1).astype(int).astype(str) + "h"

    import plotly.graph_objects as go
    couleurs_barres = ["#003f7f" if h == heure else "lightblue" for h in df_plot["Heure_Entree"]]

    fig = go.Figure()

    # Barres d’affluence
    fig.add_trace(go.Bar(
        x=df_plot["Heure_affichage"],
        y=df_plot["Moy_arrivees"],
        name="Arrivées moyennes",
        marker=dict(color=couleurs_barres),
        yaxis="y2",
        opacity=0.8,
        legendrank=2
    ))

    fig.add_trace(go.Scatter(
        x=df_plot["Heure_affichage"],
        y=df_plot["Duree_moy_min"],
        mode="lines+markers",
        name="Durée moyenne (min)",
        line=dict(color="crimson", width=3),
        legendrank=1
    ))

    fig.update_layout(
        title="📊 Durée moyenne de passage et affluence pour les paramètres sélectionnés",
        xaxis=dict(title="Tranche horaire", tickmode="array", tickvals=df_plot["Heure_affichage"], ticktext=df_plot["Heure_affichage"], tickangle=0),
        yaxis=dict(title="Durée moyenne (min)"),
        yaxis2=dict(title="Entrées moyennes", overlaying="y", side="right", range=[0, 8], layer="below traces"),
        legend=dict(x=0.01, y=0.99),
        bargap=0.2,
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)