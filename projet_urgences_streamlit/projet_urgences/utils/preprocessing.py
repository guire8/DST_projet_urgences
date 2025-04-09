import pandas as pd

def transformer_donnees_utilisateur(df_user, df_base):
    df_user["Delai_entree_IOA_heure"] = (df_user["Date_Heure_PEC_IOA"] - df_user["Date_Heure_Entree_Sejour"]).dt.total_seconds() / 3600
    df_user["Jour"] = df_user["Date_Heure_Entree_Sejour"].dt.dayofweek
    df_user["Heure_Entree"] = df_user["Date_Heure_Entree_Sejour"].dt.hour
    df_user["Date"] = df_user["Date_Heure_Entree_Sejour"].dt.date

    affluence = df_base.groupby(df_base["Date_Heure_Entree_Sejour"].dt.date).size().rename("nombre_patients_present")
    df_user = df_user.merge(affluence, left_on="Date", right_index=True, how="left")
    df_user["nombre_patients_present"].fillna(df_base.shape[0] / df_base["Date"].nunique(), inplace=True)

    return df_user