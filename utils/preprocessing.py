import pandas as pd
import numpy as np
from datetime import timedelta
import holidays

def preprocess_common(df):
    # Renommage
    df.columns = df.columns.str.replace('[/ ]', '_', regex=True)
    df.columns = df.columns.str.replace('é', 'e')
    df.columns = df.columns.str.replace('[()]', '', regex=True)
    
    # Conversion des dates
    date_cols = ['Date_Heure_PEC_IOA', 'Date_Heure_PEC_MED']
    for col in ['Date_Heure_PEC_IOA', 'Date_Heure_PEC_MED']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calcul des délais
    df['Delai_entree_IOA_heure'] = (df['Date_Heure_PEC_IOA'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600
    df['Delai_IOA_MED_heure'] = (df['Date_Heure_PEC_MED'] - df['Date_Heure_PEC_IOA']).dt.total_seconds() / 3600
    df['Delai_entree_MED_heure'] = (df['Date_Heure_PEC_MED'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600
    df['Delai_MED_sortie_heure'] = (df['Date_Heure_Sortie_Urgences'] - df['Date_Heure_PEC_MED']).dt.total_seconds() / 3600
    df['Duree_totale_heure'] = (df['Date_Heure_Sortie_Urgences'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600

    # Colonnes temporelles
    df['Jour_Entree'] = df['Date_Heure_Entree_Sejour'].dt.day_name()
    df['Heure_Entree'] = df['Date_Heure_Entree_Sejour'].dt.hour
    df['Jour'] = df['Date_Heure_Entree_Sejour'].dt.dayofweek
    df['Mois'] = df['Date_Heure_Entree_Sejour'].dt.month
    df['Annee'] = df['Date_Heure_Entree_Sejour'].dt.year
    df['Semaine_Annee'] = df['Date_Heure_Entree_Sejour'].dt.isocalendar().week

    # Jour férié (France)
    fr_holidays = holidays.FR()
    df['jour_ferie'] = df['Date_Heure_Entree_Sejour'].dt.date.apply(lambda d: d in fr_holidays).astype(int)

    return df

def preprocess_for_attente(df, input_row):
    df = preprocess_common(df.copy())

    # Filtrage 2023/2024
    df = df[df['Annee'].isin([2023, 2024])]

    # Ajout ligne utilisateur
    df_input = preprocess_common(pd.DataFrame([input_row]))
    df = pd.concat([df, df_input], ignore_index=True)

    # Dummy values pour les colonnes nécessaires si besoin
    df['Salle_attente_IOA'] = df.get('Salle_attente_IOA', 0)
    df['Salle_attente_MED'] = df.get('Salle_attente_MED', 0)
    df['nombre_patients_present'] = df.get('nombre_patients_present', 50)

    return df.tail(1)

def preprocess_for_hospitalisation(df, input_row):
    df = preprocess_common(df.copy())
    df['Hospitalisation'] = df['Orientation'].apply(lambda x: 1 if isinstance(x, str) and 'HOSP' in x.upper() else 0)
    df = df[df['Annee'].isin([2023, 2024])]

    df_input = preprocess_common(pd.DataFrame([input_row]))
    df_input['Hospitalisation'] = 0
    return df_input

def preprocess_for_moyenne(df):
    df = preprocess_common(df.copy())
    df = df[df['Annee'].isin([2023, 2024])]
    duree_moyenne = df['Duree_totale_heure'].mean()
    if pd.api.types.is_timedelta64_dtype(duree_moyenne):
        return duree_moyenne.total_seconds() / 60
    else:
        return duree_moyenne