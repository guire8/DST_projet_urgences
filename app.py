import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from random import randint
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
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
df_nona = df_cleaned[
    ~df_cleaned.map(lambda x: isinstance(x, str) and x.strip().lower() == "nan")
]

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

    # --- Création des figures ---
    colors = px.colors.sequential.Magma

    # Âge
    fig_age = px.histogram(
        df_cleaned,
        x="Age_Moyen_Sejour_Annees",
        nbins=15,
        title="Distribution de l'âge",
        color_discrete_sequence=["#6c2b6d"]
    )
    fig_age.update_layout(yaxis_title="Nombre de séjours", bargap=0.05)

    # Tri_IOA
    ordre_tri = ['Tri 1', 'Tri 2', 'Tri 3A', 'Tri 3B', 'Tri 4', 'Tri 5']
    df_tri = df_cleaned["Tri_IOA"].value_counts().reindex(ordre_tri).reset_index()
    df_tri.columns = ["Tri_IOA", "count"]
    fig_tri = px.bar(
        df_tri,
        x="Tri_IOA",
        y="count",
        title="Distribution du Tri_IOA",
        color_discrete_sequence=colors
    )
    fig_tri.update_layout(yaxis_title="Nombre de séjours")

    # Motif de recours
    df_motif = df_cleaned[
    df_cleaned["Motif_de_recours"].apply(lambda x: isinstance(x, str) and x.strip().lower() != "nan")]["Motif_de_recours"].value_counts().reset_index()
    df_motif.columns = ["Motif_de_recours", "count"]
    fig_motif = px.bar(
        df_motif,
        x="Motif_de_recours",
        y="count",
        title="Répartition des motifs de recours",
        color_discrete_sequence=colors
    )
    fig_motif.update_layout(
        yaxis_title="Nombre de séjours",
        xaxis_tickangle=45
    )

    # Affichage des trois figures
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        st.plotly_chart(fig_tri, use_container_width=True)

    st.plotly_chart(fig_motif, use_container_width=True)

    # 🔥 Carte thermique des arrivées - version améliorée
    st.subheader("2. Carte thermique des arrivées")

    # Préparer les données groupées
    heatmap_data = df_cleaned.groupby(["Jour", "Heure_Entree"]).size().reset_index(name="count")

    # Mapper les jours pour qu’ils soient affichés correctement dans l'ordre
    jours_mapping = {
        0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi",
        4: "Vendredi", 5: "Samedi", 6: "Dimanche"
    }
    heatmap_data["JourNom"] = heatmap_data["Jour"].map(jours_mapping)

    # Création de la heatmap Plotly
    fig_heatmap = px.density_heatmap(
        heatmap_data,
        x="Heure_Entree",
        y="JourNom",
        z="count",
        color_continuous_scale="RdBu_r",  # proche de coolwarm
        title="Carte thermique des arrivées par jour et heure",
        nbinsx=24,
        category_orders={"JourNom": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]}
    )

    fig_heatmap.update_layout(
        xaxis=dict(
            title="Heure de la journée",
            tickmode="linear",
            tick0=0,
            dtick=1
        ),
        xaxis_title="Heure de la journée",
        yaxis_title="Jour de la semaine",
        height=600,
        width=1200,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_heatmap, use_container_width=False)

    st.subheader("3. Analyse des valeurs manquantes")
    missing = df_raw.isnull().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["Colonne", "Taux de valeurs manquantes"]
    fig_missing = px.bar(missing, x="Colonne", y="Taux de valeurs manquantes", title="Taux de valeurs manquantes")
    st.plotly_chart(fig_missing)

    import plotly.graph_objects as go

     # Calcul des pourcentages de séjours avec valeurs manquantes
    pourcent_sortie_sejour = (
        df_cleaned[df_cleaned["Date_Heure_Sortie_Urgences"].isna()]
        .groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year)
        .size()
        / df_cleaned.groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year).size()
        * 100
    )

    pourcent_PEC_MED = (
        df_cleaned[df_cleaned["Date_Heure_PEC_MED"].isna()]
        .groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year)
        .size()
        / df_cleaned.groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year).size()
        * 100
    )

    pourcent_PEC_IOA = (
        df_cleaned[df_cleaned["Date_Heure_PEC_IOA"].isna()]
        .groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year)
        .size()
        / df_cleaned.groupby(df_cleaned["Date_Heure_Entree_Sejour"].dt.year).size()
        * 100
    )

    # Création du graphique Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pourcent_sortie_sejour.index,
            y=pourcent_sortie_sejour.values,
            mode="lines+markers",
            name="Sortie Séjour Manquant",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pourcent_PEC_MED.index,
            y=pourcent_PEC_MED.values,
            mode="lines+markers",
            name="PEC MED Manquant",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pourcent_PEC_IOA.index,
            y=pourcent_PEC_IOA.values,
            mode="lines+markers",
            name="PEC IOA Manquant",
        )
    )

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title_text="Pourcentage de Séjours avec Valeurs Manquantes par Année",
        xaxis_title="Année",
        yaxis_title="Pourcentage (%)",
        height=600,
        width=800,
    )

    # Affichage dans Streamlit
    st.title("Analyse des Séjours avec Valeurs Manquantes")
    st.plotly_chart(fig, use_container_width=True)

    # Analyse de l'impact de la dispersion entre 2021_2024 et 2023_2024
    #Creation de dataset pour 2023-2024 et 2021-2024 :
    df_2324 = df_cleaned[df_cleaned['Date_Heure_Entree_Sejour'].dt.year.isin([2023,2024])].copy()
    df_2124 = df_cleaned[df_cleaned['Date_Heure_Entree_Sejour'].dt.year.isin([2021,2022,2023,2024])].copy()
    
    from plotly.subplots import make_subplots
   

    fig = make_subplots(rows=1, cols=8, shared_xaxes=True)

    # Trace des boxplots pour les 4 durees
    df_2124['Duree_totale_heure'] = pd.to_timedelta(df_2124['Duree_totale_heure'], unit='h')
    df_2324['Duree_totale_heure'] = pd.to_timedelta(df_2324['Duree_totale_heure'], unit='h')

    df_2124['Delai_entree_IOA_heure'] = pd.to_timedelta(df_2124['Delai_entree_IOA_heure'], unit='h')
    df_2324['Delai_entree_IOA_heure'] = pd.to_timedelta(df_2324['Delai_entree_IOA_heure'], unit='h')

    df_2124['Delai_IOA_MED_heure'] = pd.to_timedelta(df_2124['Delai_IOA_MED_heure'], unit='h')
    df_2324['Delai_IOA_MED_heure'] = pd.to_timedelta(df_2324['Delai_IOA_MED_heure'], unit='h')

    df_2124['Delai_MED_sortie_heure'] = pd.to_timedelta(df_2124['Delai_MED_sortie_heure'], unit='h')
    df_2324['Delai_MED_sortie_heure'] = pd.to_timedelta(df_2324['Delai_MED_sortie_heure'], unit='h')
    fig.add_trace(go.Box(y=(df_2124['Duree_totale_heure'].dt.total_seconds() / 3600).round(3), name='Duree_totale 2124'), row=1, col=1)
    fig.add_trace(go.Box(y=(df_2324['Duree_totale_heure'].dt.total_seconds()/3600).round(3), name='Duree_Totale 2324'), row=1, col=2)

    fig.add_trace(go.Box(y=(df_2124['Delai_entree_IOA_heure'].dt.total_seconds()/3600).round(3), name='Delai_entree_IOA 2124'), row=1, col=3)
    fig.add_trace(go.Box(y=(df_2324['Delai_entree_IOA_heure'].dt.total_seconds()/3600).round(3), name='Delai_entree_IOA 2324'), row=1, col=4)

    fig.add_trace(go.Box(y=(df_2124['Delai_IOA_MED_heure'].dt.total_seconds()/3600).round(3), name='Delai_IOA_MED 2124'), row=1, col=5)
    fig.add_trace(go.Box(y=(df_2324['Delai_IOA_MED_heure'].dt.total_seconds()/3600).round(3), name='Delai_IOA_MED 2324'), row=1, col=6)

    fig.add_trace(go.Box(y=(df_2124['Delai_MED_sortie_heure'].dt.total_seconds()/3600).round(3), name='Delai_MED_sortie 2124'), row=1, col=7)
    fig.add_trace(go.Box(y=(df_2324['Delai_MED_sortie_heure'].dt.total_seconds()/3600).round(3), name='Delai_MED_sortie 2324'), row=1, col=8)

    fig.update_layout(height=600, width=1600, title_text="Boxplots des Durees aux Urgences (en heures) / Impact des valeurs aberrantes et outliers", title_x= 0.5)


    fig.add_annotation(
        x= 0.05,  # Coordonnee x negative pour aller à gauche
        y=1.08,  # Coordonnee y au centre vertical
        text="Duree entre Entree et Sortie Urgence",  # Texte
        showarrow=False,
        font=dict(size=14, color="white"),
        xref="paper",
        yref="paper",
        textangle=0  # Texte incline verticalement
    )

    fig.add_annotation(
        x= 0.38,  # Coordonnee x negative pour aller à gauche
        y=1.08,  # Coordonnee y au centre vertical
        text="Duree entre Entree et PEC IOA",  # Texte
        showarrow=False,
        font=dict(size=14, color="white"),
        xref="paper",
        yref="paper",
        textangle=0  # Texte incline verticalement
    )

    fig.add_annotation(
        x= 0.63,  # Coordonnee x negative pour aller à gauche
        y=1.08,  # Coordonnee y au centre vertical
        text="Duree entre PEC IOA et PEC MED",  # Texte
        showarrow=False,
        font=dict(size=14, color="white"),
        xref="paper",
        yref="paper",
        textangle=0  # Texte incline verticalement
    )

    fig.add_annotation(
        x= 0.98,  # Coordonnee x negative pour aller à gauche
        y=1.08,  # Coordonnee y au centre vertical
        text="Duree entre PEC MED et Sortie",  # Texte
        showarrow=False,
        font=dict(size=14, color="white"),
        xref="paper",
        yref="paper",
        textangle=0  # Texte incline verticalement
    )


    # Mise à jour des echelles des axes Y
    fig.update_layout(
        height=600,
        width=1600,
        title_text="Boxplots des Durees aux Urgences (en heures) / Impact des valeurs aberrantes et outliers",
        title_x=0.5,
        yaxis=dict(range=[0, 30]),  # Pour le premier subplot
        yaxis2=dict(range=[0, 30]),  # Pour le deuxième subplot
        yaxis3=dict(range=[0, 30]),  # Pour le troisième subplot
        yaxis4=dict(range=[0, 30]),   # Pour le quatrième subplot
        yaxis5=dict(range=[0, 30]),   # Pour le cinquième subplot
        yaxis6=dict(range=[0, 30]),   # Pour le sixième subplot
        yaxis7=dict(range=[0, 30]),   # Pour le septième subplot
        yaxis8=dict(range=[0, 30])
    )


    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

     st.markdown("""
            On constate qu'il n'y a pas de modification majeures sur les différents temps d'attente entre 2020-2024 et 2023-2024.
            Par conséquent, noous décidons de ne travailler que sur le dataset 2023-2024 qui à moins de données manquantes.
            """)


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
    now = datetime.now(ZoneInfo("Europe/Paris"))
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
        now = datetime.now(ZoneInfo("Europe/Paris"))

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
            pec_options = [x for x in df_cleaned["Type_de_PEC"].unique() if isinstance(x, str) and x.strip().lower() not in ["nan", ""]]
            type_pec = st.selectbox("Type de PEC", sorted(pec_options), key="attente_pec")
            tri_options = [x for x in df_cleaned["Tri_IOA"].unique() if isinstance(x, str) and x.strip().lower() not in ["nan", ""]]
            tri_ioa = st.selectbox("Tri IOA", sorted(tri_options), key="attente_tri")
            discipline_options = [x for x in df_cleaned["Discipline_Examen"].unique() if isinstance(x, str) and x.strip() not in ["", "-", "nan"]]
            discipline = st.selectbox("Discipline Examen", sorted(discipline_options), key="attente_discipline")

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

    # Comparaison des modèles de régression
    data = {
        "Modèle": [
            "Gradient Boosting Regressor",
            "Gradient Boosting Regressor_huber",
            "XGBoost Regressor",
            "CatBoost Regressor",
            "Random_Forest",
            "Linear Regression"
        ],
        "RMSE": [5.21, 5.23, 5.10, 5.41, 5.01, 5.27],
        "MAE": [3.73, 3.61, 3.65, 3.96, 3.58, 3.83],
        "MAPE (%)": [65.7, 59.2, 63.8, 81.0, 67.8, 71.1],
        "R²": [0.4, 0.4, 0.4, 0.4, 0.5, 0.4],
        "Cross-Val Score": [0.405, 0.395, 0.421, 0.359, 0.431, 0.388]
    }
    
    # Création du DataFrame
    df = pd.DataFrame(data)
    
    # Affichage dans Streamlit
    st.title("Comparaison des modèles de régression")
    st.dataframe(df, use_container_width=True)



df_comparatif_models = pd.DataFrame({
    "Modèle": [
        "Logistic Regression", "Logistic Regression",
        "Random Forest", "Random Forest",
        "Gradient Boosting", "Gradient Boosting",
        "XGBoost", "XGBoost"
    ],
    "Stratégie": [
        "Sans SMOTE", "Avec SMOTE",
        "Sans SMOTE", "Avec SMOTE",
        "Sans SMOTE", "Avec SMOTE",
        "Sans SMOTE", "Avec SMOTE"
    ],
    "Accuracy": [0.7424, 0.7430, 0.8301, 0.8157, 0.8133, 0.7891, 0.8146, 0.7848],
    "Precision (Classe 1)": [0.46, 0.46, 0.70, 0.60, 0.64, 0.53, 0.64, 0.52],
    "Recall (Classe 1)": [0.79, 0.79, 0.44, 0.58, 0.41, 0.65, 0.41, 0.66],
    "F1-score (Classe 1)": [0.58, 0.58, 0.54, 0.59, 0.50, 0.58, 0.50, 0.58]
})

df_results_recall = pd.DataFrame({
    "Modèle": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
    "Accuracy": [0.7360, 0.7760, 0.7736, 0.7532],
    "Precision (Classe 1)": [0.45, 0.50, 0.50, 0.47],
    "Recall (Classe 1)": [0.80, 0.72, 0.71, 0.73],
    "F1-score (Classe 1)": [0.58, 0.59, 0.59, 0.57]
})

df_results_precision = pd.DataFrame({
    "Modèle": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
    "Accuracy": [0.7432, 0.7728, 0.8194, 0.8179],
    "Precision (Classe 1)": [0.46, 0.50, 0.62, 0.62],
    "Recall (Classe 1)": [0.78, 0.73, 0.52, 0.49],
    "F1-score (Classe 1)": [0.58, 0.59, 0.57, 0.55]
})

df_stacking_results = pd.DataFrame({
    "Métrique": ["Accuracy", "Recall", "Precision", "F1-score"],
    "Score": [0.81, 0.56, 0.46, 0.51]
})

df_final_stacking_results = pd.DataFrame({
    "Modèle": ["Logistic Regression (Opt.)", "Random Forest (Opt.)", "Gradient Boosting (Opt.)", "XGBoost (Opt.)", "Stacking Final"],
    "Accuracy": [0.75, 0.82, 0.83, 0.84, 0.87],
    "Recall": [0.71, 0.78, 0.79, 0.80, 0.83],
    "Precision": [0.69, 0.77, 0.78, 0.79, 0.81],
    "F1-score": [0.70, 0.77, 0.78, 0.79, 0.82],
    "AUC-ROC": [0.76, 0.85, 0.86, 0.87, 0.89]
})

# --- Fonction d’affichage de tableau Plotly ---
def show_table_plotly(df, title):
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='lightblue',
            align='center',
            height=35
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='white',
            align='center',
            height=30
        )
    )])

    row_height = 30
    base_height = 80
    max_height = 500
    total_height = min(base_height + row_height * len(df), max_height)

    fig.update_layout(
        title=title,
        margin=dict(t=40, b=10),
        height=total_height
    )
    st.plotly_chart(fig, use_container_width=True)


code_preparation = '''
df=pd.read_pickle('/content/drive/MyDrive/CDA_Urgences/df_urgences_2324.pkl')

df["Delai_entree_IOA_min"] = pd.to_timedelta(((df["Date_Heure_PEC_IOA"] - df["Date_Heure_Entree_Sejour"]))).dt.total_seconds() / 60
df["Delai_IOA_MED_min"] = pd.to_timedelta(((df["Date_Heure_PEC_MED"] - df["Date_Heure_PEC_IOA"]))).dt.total_seconds() / 60
df['Delai_entree_MED_min'] = pd.to_timedelta(((df["Date_Heure_PEC_MED"] - df["Date_Heure_Entree_Sejour"]))).dt.total_seconds() / 60
df["Delai_MED_sortie_min"] = pd.to_timedelta(((df["Date_Heure_Sortie_Urgences"] - df["Date_Heure_PEC_MED"]))).dt.total_seconds() / 60
df["Duree_totale_min"] = pd.to_timedelta(((df["Date_Heure_Sortie_Urgences"] - df["Date_Heure_Entree_Sejour"]))).dt.total_seconds() / 60

# Sélection des features et de la target
feats_cols = ["Type_de_PEC", "Motif_de_recours", "Tri_IOA", "Delai_entree_IOA_min",
              "Age_Moyen_Sejour_Annees", "Jour_Entree", "Heure_Entree", "nombre_patients_present",
              "Jour", "Mois", "Annee", "Semaine_Annee", "jour_ferie"]
target_col = "Hospitalisation"

X = df[feats_cols]
y = df[target_col]

# Séparation des variables numériques et catégoriques
numerical_features = X.select_dtypes(include=["int64", "float64", "int32", "UInt32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Transformation des données AVANT SMOTE
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_transformed = preprocessor.fit_transform(X)
'''
code_eval_initial = '''
# Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Appliquer SMOTE uniquement sur les données transformées
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Définition des modèles avec class_weight et SMOTE séparés
strategies = {
    "Sans SMOTE (class_weight)": (X_train, y_train),
    "Avec SMOTE": (X_train_smote, y_train_smote)
}

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", use_label_encoder=False)
}

# Tester chaque stratégie (Sans SMOTE vs Avec SMOTE)
results = {}

for strat_name, (X_train_mod, y_train_mod) in strategies.items():
    print(f"\n🚀 **Stratégie : {strat_name}** 🚀\n")

    for name, model in models.items():
        model.fit(X_train_mod, y_train_mod)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n🔹 {name} ({strat_name}) 🔹")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non hospitalisé", "Hospitalisé"], yticklabels=["Non hospitalisé", "Hospitalisé"])
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité")
        plt.title(f"Matrice de confusion - {name} ({strat_name})")
        plt.show()

        # Stocker les résultats
        results[f"{name} ({strat_name})"] = accuracy
'''

code_random_search_recall = '''
# Définition des modèles et paramètres réduits pour `RandomizedSearchCV`
models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": np.logspace(-3, 3, 5),  # Réduction du nombre de valeurs testées
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],  # Réduction du nombre d'estimateurs
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced"]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, len(y_train_smote[y_train_smote == 0]) / len(y_train_smote[y_train_smote == 1])]
        }
    }
}

# Optimisation avec RandomizedSearchCV (10 itérations max)
best_models = {}

for name, mp in models_params.items():
    print(f"\n🚀 Optimisation de {name}...\n")

    search = RandomizedSearchCV(
        mp["model"], mp["params"],
        scoring="recall", cv=3, n_jobs=-1, random_state=42, n_iter=10
    )

    search.fit(X_train_smote, y_train_smote)

    best_models[name] = search.best_estimator_
    print(f"🔹 Meilleur modèle {name}: {search.best_params_}")

# Évaluation des meilleurs modèles trouvés
results = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🔹 {name} (Optimisé) 🔹")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non hospitalisé", "Hospitalisé"], yticklabels=["Non hospitalisé", "Hospitalisé"])
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.title(f"Matrice de confusion - {name} (Optimisé)")
    plt.show()

    # Stocker les résultats
    results[name] = accuracy
'''

code_random_search_precision = '''
# Définition des modèles et paramètres réduits pour `RandomizedSearchCV`
models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": np.logspace(-3, 3, 5),  # Réduction du nombre de valeurs testées
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],  # Réduction du nombre d'estimateurs
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced"]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, len(y_train_smote[y_train_smote == 0]) / len(y_train_smote[y_train_smote == 1])]
        }
    }
}

# Optimisation avec RandomizedSearchCV (10 itérations max)
best_models = {}

for name, mp in models_params.items():
    print(f"\n🚀 Optimisation de {name}...\n")

    search = RandomizedSearchCV(
        mp["model"], mp["params"],
        scoring="precision", cv=3, n_jobs=-1, random_state=42, n_iter=10
    )

    search.fit(X_train_smote, y_train_smote)

    best_models[name] = search.best_estimator_
    print(f"🔹 Meilleur modèle {name}: {search.best_params_}")

# Évaluation des meilleurs modèles trouvés
results = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🔹 {name} (Optimisé) 🔹")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non hospitalisé", "Hospitalisé"], yticklabels=["Non hospitalisé", "Hospitalisé"])
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.title(f"Matrice de confusion - {name} (Optimisé)")
    plt.show()

    # Stocker les résultats
    results[name] = accuracy
'''
code_random_mlp = '''
# Paramètres de recherche
param_distributions = {
    "hidden_layer_sizes": [(50,), (100,), (50, 25), (128, 64)],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "learning_rate_init": [1e-3, 1e-2, 1e-1],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"]
}

# Création du modèle
mlp = MLPClassifier(max_iter=500, random_state=42)

# Définition du scorer basé sur le recall (classe 1)
recall_scorer = make_scorer(recall_score)

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Randomized Search
search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=recall_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Entraînement sur les données SMOTE
search.fit(X_train_smote, y_train_smote)

# Meilleur modèle
best_mlp = search.best_estimator_

print("Meilleurs hyperparamètres (RandomizedSearch - Recall):")
print(search.best_params_)
'''
code_final_optimization = '''
# Modèles fixes
gb_best_recall = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.01,
    random_state=42
)

lr_best_precision = LogisticRegression(
    solver='liblinear',
    C=0.001,
    max_iter=500,
    random_state=42
)

best_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    alpha=0.009485527090157502,
    learning_rate_init=0.001,
    activation='tanh',
    solver='adam',
    max_iter=500,
    random_state=42
)

def objective(trial):
    # Sélection du 3e modèle à tester
    third_model_name = trial.suggest_categorical("third_model", ["rf_recall", "xgb_recall", "xgb_precision", "lr_recall", "gb_precision"])

    if third_model_name == "rf_recall":
        third_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42)
    elif third_model_name == "xgb_recall":
        third_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, scale_pos_weight=1.0, use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif third_model_name == "xgb_precision":
        third_model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.2, scale_pos_weight=1.0, eval_metric="logloss", random_state=42)
    elif third_model_name == "lr_recall":
        third_model = LogisticRegression(solver='liblinear', C=0.001, max_iter=500, random_state=42)
    elif third_model_name == "gb_precision":
        third_model = GradientBoostingClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)

    estimators = [
        ("gb", gb_best_recall),
        ("lr", lr_best_precision),
        ("third", third_model)
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=best_mlp,
        passthrough=False,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(stacking, X_train_smote, y_train_smote, cv=skf)

    # Sauvegarde du nom de la combinaison et des métriques
    trial.set_user_attr("third_model", third_model_name)
    trial.set_user_attr("f1", f1_score(y_train_smote, y_pred))
    trial.set_user_attr("recall", recall_score(y_train_smote, y_pred))
    trial.set_user_attr("precision", precision_score(y_train_smote, y_pred))
    trial.set_user_attr("accuracy", accuracy_score(y_train_smote, y_pred))

    return f1_score(y_train_smote, y_pred)

study = optuna.create_study(direction="maximize", study_name="stacking_best3rd")
study.optimize(objective, n_trials=20)

# Résumé des résultats de tous les essais
print("\\nRésultats de toutes les combinaisons testées :\\n")

results = []
for t in study.trials:
    if t.values:
        results.append({
            "Modèle 3e colonne": t.user_attrs.get("third_model"),
            "F1-score": round(t.user_attrs.get("f1", 0), 3),
            "Recall": round(t.user_attrs.get("recall", 0), 3),
            "Précision": round(t.user_attrs.get("precision", 0), 3),
            "Accuracy": round(t.user_attrs.get("accuracy", 0), 3)
        })

df_results = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
print(df_results)

print("Meilleure combinaison :")
print(study.best_trial.params)
'''

# --- Affichage de l’onglet Présentation ---
with tab3:
    st.title("🏥 Prédiction du risque d'hospitalisation")
    onglet_presentation, onglet_resultats, onglet_test = st.tabs(["🧠 Présentation du modèle", "📊 Résultats du modèle", "✪ Test en conditions réelles"])

    with onglet_presentation:
        st.subheader("🧪 Démarche de modélisation")

        st.markdown("### 🗃️ Chargement et préparation du jeu de données")
        with st.expander("Voir le code"):
            st.code(code_preparation, language="python")

        st.markdown("### 🔍 Comparaison initiale des modèles avec/sans SMOTE")
        with st.expander("Voir le code"):
            st.code(code_eval_initial, language="python")
        show_table_plotly(df_comparatif_models, "Comparaison des modèles avec et sans SMOTE")

        st.markdown("""
        On constate que l'utilisation de SMOTE permet une nette amélioration du recall et du F1 score sur l'ensemble des modèles avec un léger recul sur la précision.  
        Seul le modèle de régression logistique n'est pas impacté.  
        On décide donc de conserver l'utilisation de SMOTE pour la suite.
        \n De plus, on remarque pour l'ensemble des modèles un déséquilibre entre précision et recall.  
        On décide donc de s'orienter vers un modèle de stacking classifier avec des modèles optimisés sur le recall et la precision afin de combiner les forces des différents modèles et obtenir les meilleurs résultats possibles.
        """)

        st.markdown("### 🔧 Optimisation par RandomizedSearchCV (Recall)")
        with st.expander("Voir le code"):
            st.code(code_random_search_recall, language="python")
        df_results_recall_sorted = df_results_recall.sort_values(by="Recall (Classe 1)", ascending=False)
        show_table_plotly(df_results_recall_sorted, "Résultats après RandomizedSearch (Recall)")

        st.markdown("""
        La régression logistique, après optimisation, est le modèle qui montre les meilleurs performances sur le recall.
        \n Meilleur modèle Logistic Regression: {'solver': 'liblinear', 'C': np.float64(0.001)}
        """)

        st.markdown("### 🔧 Optimisation par RandomizedSearchCV (Précision)")
        with st.expander("Voir le code"):
            st.code(code_random_search_precision, language="python")
        df_results_precision_sorted = df_results_precision.sort_values(by="Precision (Classe 1)", ascending=False)
        show_table_plotly(df_results_precision_sorted, "Résultats après RandomizedSearch (Précision)")

        st.markdown("""
        On constate que XGBoost et Gradient Boosting obtiennent des résultats équivalents en précision après optimisation.  
        Toutefois, Gradient Boosting est légèrement meilleur que XGBoost sur les autres métriques nous allons donc le retenir comme meilleur modèle pour la precision.  
        \n Meilleur modèle Gradient Boosting: {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1}
        """)

        st.markdown("### 🧬 Optimisation de MLP (modèle final du stacking)")
        with st.expander("🔍 Pourquoi un MLP comme modèle final ?"):
            st.markdown("""
            Afin de comparer objectivement les performances des différentes combinaisons de modèles dans le cadre du stacking classifier, nous avons initialement testé plusieurs modèles en tant que méta-apprenants.  
            Parmi ceux-ci, le MLP (réseau de neurones multicouches) s’est rapidement démarqué en offrant les meilleurs résultats globaux avant toute phase d’optimisation.  

            Compte tenu de ses performances initiales, de sa capacité à modéliser des relations non linéaires complexes entre les prédictions des modèles de base, ainsi que des références disponibles dans la littérature scientifique — notamment en contexte médical et en urgences hospitalières — nous avons fait le choix de **nous concentrer exclusivement sur le MLP** pour l’étape d’optimisation.  

            Cette approche nous a permis de garantir une évaluation cohérente des différentes combinaisons de modèles de base tout en limitant la complexité computationnelle liée à l’optimisation de plusieurs méta-modèles.
            """)

        with st.expander("📚 Références scientifiques"):
            st.markdown("""
            1. Neshat M, et al. *Effective Predictive Modeling for Emergency Department Visits...* arXiv:2411.11275, 2024.  
            2. *Evaluation of stacked ensemble model performance to predict clinical outcomes.* Int J Med Inform, 2023.
            """)

        with st.expander("Voir le code"):
            st.code(code_random_mlp, language="python")
        show_table_plotly(df_stacking_results, "Scores du Stacking Classifier")
        
        st.markdown("""
        Meilleur modèle MLP: {'activation': 'tanh', 'alpha': 0.009485527090157502, 'hidden_layer_sizes': (128, 64)}
        """)

        st.markdown("### 🚀 Optimisation finale du stacking")
        st.markdown("""
        On optimise désormais le modèle de stacking dans son ensemble.  
        Pour cela, on garde nos meilleurs modèles sur chacune des métriques (recall et precision) ainsi que notre mlp optimisé en final_estimator.
        Le but étant de tester les différentes combinaisons possibles pour le 3ème modèle de notre stacking afin d'obtenir les meilleurs résultats.
        """)
        with st.expander("Voir le code"):
            st.code(code_final_optimization, language="python")
        show_table_plotly(df_final_stacking_results, "Comparaison finale des modèles optimisés")

        st.markdown("### 📋 Analyse des résultats")
        st.markdown("""
- Nette amélioration des résultats sur le recall de la classe 1  
- Combinaison de plusieurs techniques d'optimisation (random_search, optuna, adaptation du seuil) permettant un équilibre final correct sur la classe 1 entre précision et recall et donc un F1 score amélioré  
- Scores encore insuffisant pour une prédiction fiable  
- Limites de temps et de ressources ont empêché d'explorer plus d'options au niveau du stacking (essayer d'intégrer plus de modèles, en tester d'autres ou essayer de manière plus poussée d'autres final_estimator)
        \n Ce modèle ne faisait pas partie de nos objectifs initiaux. Nous avons décidé de l'explorer en fin de projet face aux résultats non satisfaisants sur les temps d'attente.  
        Nous avons toutefois réussi à obtenir des premiers résultats encourageants. Avec plus de temps pour mieux optimiser la construction du modèle, nous aurions sans doute pu améliorer nos résultats.  
        Dans tous les cas, ce modèle comme les autres reste limité par la qualité de nos données.
        """)

    with onglet_resultats:
        st.subheader("🔄 Résultats dynamiques selon le seuil")

        model_pack = load_model("models/model_hospit_v2.pkl")
        model = model_pack["model"]
        preprocessor = model_pack["preprocessor"]
        X_test = model_pack["X_test_transformed"]
        y_test = model_pack["y_test"]

        st.markdown("**Choix du seuil de classification**")
        threshold = st.slider("Sélectionnez le seuil de classification :", 0.0, 1.0, 0.32, 0.01)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1-score", f"{f1_score(y_test, y_pred):.2f}")
        col2.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
        col4.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

        # Affichage côte à côte : matrice + jauge F1 centrée
        col_left, col_right = st.columns(2)

        with col_left:
            center = st.columns([1, 2, 1])
            with center[1]:
                cm = confusion_matrix(y_test, y_pred)
                labels = ["Non hospitalisé", "Hospitalisé"]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm.tolist(),
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=False,
                    text=cm.astype(str),
                    texttemplate="%{text}",
                    hovertemplate="Prédit: %{x}<br>Réel: %{y}<br>Nombre: %{z}<extra></extra>"
                ))
                fig_cm.update_layout(
                    title="Matrice de confusion",
                    xaxis_title="Prédiction",
                    yaxis_title="Réalité",
                    width=500,
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_cm, use_container_width=False)

        with col_right:
            center = st.columns([1, 2, 1])
            with center[1]:
                score_f1 = round(f1_score(y_test, y_pred) * 100, 1)
                fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score_f1,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "F1-score global (%)", 'font': {'size': 20}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "lightgray"},
            {'range': [50, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': score_f1
        },
        'shape': "angular"
    }
))
                fig_gauge.update_layout(margin=dict(l=10, r=10, t=50, b=0), height=350)
                st.plotly_chart(fig_gauge, use_container_width=False)

    # --- Onglet 3 : Test d'un patient ---
    with onglet_test:
        st.subheader("🚗 Simulation d'un patient")

        with st.form("form_hospitalisation"):
            col1, col2 = st.columns(2)
            now = datetime.now(ZoneInfo("Europe/Paris"))
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
                pec_options = [x for x in df_cleaned["Type_de_PEC"].unique() if isinstance(x, str) and x.strip().lower() not in ["nan", ""]]
                type_pec = st.selectbox("Type de PEC", sorted(pec_options), key="hospit_pec")
                tri_options = [x for x in df_cleaned["Tri_IOA"].unique() if isinstance(x, str) and x.strip().lower() not in ["nan", ""]]
                tri_ioa = st.selectbox("Tri IOA", sorted(tri_options), key="hospit_tri")
                discipline_options = [x for x in df_cleaned["Discipline_Examen"].unique() if isinstance(x, str) and x.strip() not in ["", "-", "nan"]]
                discipline = st.selectbox("Discipline Examen", sorted(discipline_options), key="hospit_discipline")

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
            model_pack = load_model("models/model_hospit_v2.pkl")
            model = model_pack["model"]
            preprocessor = model_pack["preprocessor"]

            df_input = preprocess_for_hospitalisation(df_raw.copy(), input_data)
            X = preprocessor.transform(df_input)
            proba = model.predict_proba(X)[0][1]

            bas = max(0, threshold - 0.15)
            haut = min(1, threshold + 0.15)

            # --- Affichage de la jauge ---
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
                        {'range': [0, bas * 100], 'color': "lightgreen"},
                        {'range': [bas * 100, haut * 100], 'color': "orange"},
                        {'range': [haut * 100, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # --- Interprétation dynamique selon la probabilité et les bornes ---
            if proba < bas:
                color = "green"
                label = "Faible risque d'hospitalisation"
            elif proba < haut:
                color = "orange"
                label = "Risque modéré d'hospitalisation – à surveiller"
            else:
                color = "red"
                label = "Risque élevé d’hospitalisation (à confirmer)"

            st.markdown(
                f"<div style='padding: 1rem; background-color: {color}; color: white; border-radius: 0.5rem; font-weight: bold;'>"
                f"🏥 Prédiction : {label}<br>Probabilité : {round(proba*100, 1)} % (seuil {threshold})"
                f"</div>",
                unsafe_allow_html=True
            )


with tab4:
    st.title("📈 Estimation simple par moyenne")

    df_moy = preprocess_common(df_raw.copy())
    from zoneinfo import ZoneInfo

    # Conversion forcée UTC → Europe/Paris
    df_moy["Date_Heure_Entree_Sejour"] = pd.to_datetime(df_moy["Date_Heure_Entree_Sejour"], utc=True)
    df_moy["Date_Heure_Entree_Sejour"] = df_moy["Date_Heure_Entree_Sejour"].dt.tz_convert("Europe/Paris")

    # Recalcul de l'heure locale pour les graphes
    df_moy["Heure_Entree"] = df_moy["Date_Heure_Entree_Sejour"].dt.hour

    jour_mapping = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }
    
    import holidays
    fr_holidays = holidays.FR()
    df_moy["Date"] = df_moy["Date_Heure_Entree_Sejour"].dt.day
    df_moy["Mois"] = df_moy["Date_Heure_Entree_Sejour"].dt.month
    df_moy["jour_ferie"] = df_moy["Date_Heure_Entree_Sejour"].dt.date.isin(fr_holidays).astype(int)
    df_moy["Jour_Entree"] = df_moy["Date_Heure_Entree_Sejour"].dt.day_name().map(jour_mapping)

    nb_jours = df_moy["Date_Heure_Entree_Sejour"].dt.date.nunique()

    now = datetime.now(ZoneInfo("Europe/Paris"))

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
