import pandas as pd
import numpy as np
from datetime import timedelta
import holidays

def preprocess_common(df):
    # Renommage
    df.columns = df.columns.str.replace('[/ ]', '_', regex=True)
    df.columns = df.columns.str.replace('é', 'e')
    df.columns = df.columns.str.replace('[()]', '', regex=True)

    # Nettoyage des chaînes
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Valeurs à remplacer
    df["Tri_IOA"] = df["Tri_IOA"].replace("#VALEURMULTI", np.nan)
    df["Motif_de_recours"] = df["Motif_de_recours"].replace("#VALEURMULTI", np.nan)
    df["Type_de_PEC"] = df["Type_de_PEC"].replace("#VALEURMULTI", np.nan)
    df["Discipline_Examen"] = df["Discipline_Examen"].replace(["007 - --", " - ", "004 - -"], np.nan)

    # Catégorisation des motifs
    remplacements = {'Cardio-vasculaire':['Douleur thoracique', 'Douleur thoracique syndrome coronaire aigu (SCA', 'Dyspnée/insuffisance cardiaque','Palpitations', 'Bradycardie', 'Bradycardie/hypotension/tachycardie', 'Membre douloureux chaud ou rouge phlébite','Malaise avec PC', 'Malaise sans PC', 'Malaise', 'Hypertension artérielle', 'Hypertension' , 'Tachycardie', 'Tachycardie tachyarythmie', 'Œdème des membres inférieurs insuffisance cardiaque', 'Hypotension artérielle', 'Hypotension', 'Hypotension / collapsus', 'Arrêt cardiorespiratoire', 'Dysfonction stimulateur/ défibrillateur cardiaque', 'Membre douloureux froid ou pâle ischémie', 'Bradycardie bradyarythmie', 'Membre inférieur chaud rouge : symptôme de maladie thrombo-embolique', 'Œdème des membres inférieurs', 'Membre inferieur froid sans pouls/ symptômes compatible avec ischémie de membre', 'Problème avec matériel implantable cardiologique' ],
'Environnemental' : ['Hypothermie', 'Gelure / Lésions liées au froid', 'Coup de chaleur / insolation'],
'Gastro-enterologie': ['Corps étranger dans œsophage/estomac/intestins', 'Douleur abdominale', 'Constipation', 'Maelena/rectorragies', 'Vomissements', 'Hernie, masse ou distension abdominale', 'Douleur anale', 'Diarrhée', 'Vomissement de sang/ hématémèse', 'Nausées et/ou vomissement', 'Vomissement de sang', 'Hoquet', 'Ictère', 'Corps étranger dans le rectum', 'Sang dans les selles / Méléna', 'Proctologie (thrombose, fissure, douleur…)', 'Masse ou distension abdominale', "Inappétence / perte d'appétit"],
'Genito-urinaire': ['Dysfonction de sonde urinaire/ sonde JJ/ stomie' , "Rétention d'urine/anurie", 'Douleur de la fosse lombaire/du flanc', 'Dysurie/brûlure mictionnelle / infection',  "Signe d'infection urinaire/ dysurie/ brûlure mictionnelle",  'Douleur de bourse/ orchite/torsion testicule','Douleur ou œdème des OGE', 'Ecoulement ou lésion cutanéo-muqueuse génitale', 'Hématurie', 'Douleur de la fosse lombaire ou flanc', "Rétention d'urine ou anurie" ],
'Intoxication':['Intoxication non médicamenteuse', 'Intoxication médicamenteuse', 'Intoxication médicamenteuse volontaire' , 'Intoxication non médicamenteuse volontaire' ,'Intoxication accidentelle et/ou iatrogène (occasionné par le traitement médical)', 'Comportement ébrieux / ivresse','Ébriété, ivresse', 'Demande de sevrage / toxicomanie', 'Etat de manque / sevrage / toxicomanie'],
'Neurologie':['Convulsions', 'Déficit moteur, sensitif, sensoriel ou du langage/AVC', 'Confusion/désorientation temporo spatiale', "Vertiges/trouble de l'équilibre", 'Céphalée', 'Altération de la conscience/coma', 'Altération de la conscience / coma', 'Faiblesse de membre/ déficit / symptômes pouvant évoquer un AVC', 'Perte de sensibilité / paresthésies', 'Confusion / Désorientation temporo spatiale', 'Etourdissement / vertiges', "Anomalie de la marche / Ataxie / trouble de l'équilibre", 'Tremblement'],
'Gynécologie': ['Anomalie vulvo-vaginale /corps étranger', 'Agression sexuelle / sévices', 'Agression sexuelle et sévices', 'Méno-metrorragie', 'Anomalie du sein', 'Rougeur ou tension du sein', 'Douleur pelvienne', 'Méno-metrorragie en dehors de la grossesse', 'Ecoulement ou lésion génitale' , 'Pertes vaginales'],
'Obstétrique': ['Problème de grossesse 1er et 2ème trimestre',  'Problème de post partum (allaitement)',  'Problème de post partum (allaitement, hémorragie...)', 'Problèmes de grossesse 2eme trimestre (saignement, douleur, perte de liquide amniotique...)', 'Problèmes de grossesse 3eme trimestre (saignement, douleur, perte de liquide amniotique...)', 'Problèmes de grossesse 1er trimestre (saignement, douleur...)', 'Accouchement imminent ou réalisé'],
'Ophtalmologie':['Trouble visuel /œil douloureux / cécité', 'Corps étranger / brûlure oculaire', 'Corps étranger, brûlure oculaire', 'Démangeaison / œil rouge', "Trouble visuel / diplopie / baisse de l'acuité", 'Douleur oculaire', 'Démangeaison, œil rouge' ],
'ORL' :['Douleur de gorge / angine / stomatite', 'Obstruction nasale / rhinite / sinusite', 'Problème de dent ou de gencive',"Pathologie de l'oreille / otite", 'Corps étranger ORL', 'Tuméfaction ORL ou cervicale', 'Epistaxis', "Trouble de l'audition/accouphènes", 'Otorrhée / écoulement oreille', "Acouphène ou trouble de l'audition", 'Douleur de gorge, difficultés à avaler, dysphagie', 'Otalgie', 'Tuméfaction ORL', 'Congestion nasale / rhume des foins', 'Tumeur ORL et complications'],
'Peau':['Plaie', 'Abcès ou infection localisée de la peau', 'Contusion, masse, ampoule, autres affections cutanées', 'Dermabrasion / plaie', 'Œdème / rougeur localisé / abcès', 'Erythème étendu et autres éruptions / oèdeme spontané', 'Corps étranger sous la peau' , 'Corps étranger - Peau', 'Morsure, piqûre, prurit, parasitose', 'Brûlure', 'Ecchymose / hématome spontané(e)','Hématome non traumatique', 'Piqure', 'Prurit', 'Parasitose', 'Morsure', 'Erythème étendu et autres éruptions'],
'Pediatrie': ['Diarrhée / vomissements du nourrisson (<24 mois)','Fièvre <3 mois' , 'Fièvre < 3 mois', 'Convulsion hyperthermique' ],
'Psychiatrie': ['Anxiété / dépression / consultation psychiatrique' ,'Idée / comportement suicidaire', 'Troubles du comportement / psychiatrie', 'Dépression / suicidaire', "Anxiété / Crise d'angoisse / spasmophilie",  'Comportement étrange / troubles du comportement social / anorexie / boulimie', 'Comportement violent / agitation', 'Insomnie / troubles du sommeil' , 'Hallucinations' ],
'Respiratoire': ['Dyspnée / insuffisance respiratoire', 'Dyspnée','Douleur thoracique / embolie / pneumopathie / pneumothorax', 'Toux / bronchite', 'Corps étranger voies aériennes', 'Hémoptysie', 'Dyspnée avec sifflement respiratoire', 'Asthme ou aggravation BPCO', 'Asthme', 'Toux', 'Sifflement respiratoire sans autres symptômes (bronchiolite, laryngite…)', 'Réaction allergique', 'Corps étranger voies digestives', 'Cyanose' ],
'Rhumatologie':['Douleur de membre/sciatique', 'Douleur rachidienne (cervicale, dorsale, lombaire)', 'Douleur articulaire/arthrose/arthrite', 'Douleur rachidienne (cervicale, dorsale ou lombaire)',  "Douleur au niveau d'un membre inférieur" , "Douleur au niveau d'un membre supérieur", 'Douleur diffuse / sans précision', 'Epanchement articulaire'],
'Traumatologie': ['Traumatisme abdomen/thorax/cervical', "Traumatisme d'épaule ou distal de membre", 'Traumatisme membre supérieur', 'Traumatisme multiple', 'Traumatisme crânien sans PC', 'Traumatisme de bassin/hanche/fémur/rachis', 'Traumatisme crânien' , 'Traumatisme maxillo-facial/oreille', 'Traumatisme oculaire',  'Traumatisme avec amputation', 'Traumatisme cervical (hors rachis)', 'Traumatisme membre inférieur' ,  'Traumatisme thoracique' ,  'Traumatisme abdominal' ,'Traumatisme du bassin', 'Traumatisme anal ou du rectum', 'Traumatisme rachidien ou médullaire',  'Traumatisme crânien avec PC' ,'Traumatisme maxillo-facial',  'Traumatisme OGE, priapisme' ,'Amputation' ],
'General': ['AEG / Asthénie', 'AEG/Asthénie' ,'Anomalie de résultat biologique', 'Renouvellement ordonnance','Hypoglycémie','Examen à des fins administratives / certificat / réquisition', 'Allergie', 'Hyperglycémie', 'Problème de suture et pansements chirurgicaux / pansements', 'Problème suite de soins (pansements)', 'Fièvre', "Demande d'hébergement pour raison sociale", 'Problème social', 'Problème technique (stomie, cicatrices post op…)', 'Exposition à une maladie contagieuse', 'AES et/ou liquide biologique', 'Problème de matériel médical',  'Examen à des fins administratives/certificat/ réquisitions' , 'Conseil / consultation / renouvellement ordonnance' , "Accident d'exposition au sang et/ou au liquides biologiques", 'Complication post opératoire', "Demande d'examen complémentaire", 'Soins palliatifs / fin de vie', "Perte d'autonomie", 'Pathologie rare et grave en poussée (ex drépanocytose)', 'Electrisation'],
}

    for categorie, valeurs in remplacements.items():
        df['Motif_de_recours'] = df['Motif_de_recours'].replace(valeurs, categorie)

    for col in ['Date_Heure_PEC_IOA', 'Date_Heure_PEC_MED']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'Date_Heure_PEC_IOA' in df.columns and 'Date_Heure_Entree_Sejour' in df.columns:
        df['Delai_entree_IOA_heure'] = (df['Date_Heure_PEC_IOA'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600
    if 'Date_Heure_PEC_MED' in df.columns and 'Date_Heure_PEC_IOA' in df.columns:
        df['Delai_IOA_MED_heure'] = (df['Date_Heure_PEC_MED'] - df['Date_Heure_PEC_IOA']).dt.total_seconds() / 3600
    if 'Date_Heure_PEC_MED' in df.columns and 'Date_Heure_Entree_Sejour' in df.columns:
        df['Delai_entree_MED_heure'] = (df['Date_Heure_PEC_MED'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600
    if 'Date_Heure_PEC_MED' in df.columns and 'Date_Heure_Sortie_Urgences' in df.columns:
        df['Delai_MED_sortie_heure'] = (df['Date_Heure_Sortie_Urgences'] - df['Date_Heure_PEC_MED']).dt.total_seconds() / 3600
    if 'Date_Heure_Entree_Sejour' in df.columns and 'Date_Heure_Sortie_Urgences' in df.columns:
        df['Duree_totale_heure'] = (df['Date_Heure_Sortie_Urgences'] - df['Date_Heure_Entree_Sejour']).dt.total_seconds() / 3600

    if 'Date_Heure_Entree_Sejour' in df.columns:
        df['Jour_Entree'] = df['Date_Heure_Entree_Sejour'].dt.day_name()
        df['Heure_Entree'] = df['Date_Heure_Entree_Sejour'].dt.hour
        df['Jour'] = df['Date_Heure_Entree_Sejour'].dt.dayofweek
        df['Mois'] = df['Date_Heure_Entree_Sejour'].dt.month
        df['Annee'] = df['Date_Heure_Entree_Sejour'].dt.year
        df['Semaine_Annee'] = df['Date_Heure_Entree_Sejour'].dt.isocalendar().week

        fr_holidays = holidays.FR()
        df['jour_ferie'] = df['Date_Heure_Entree_Sejour'].dt.date.apply(lambda d: d in fr_holidays).astype(int)

    return df

def preprocess_for_attente(df, input_row):
    df = preprocess_common(df.copy())
    df = df[df['Annee'].isin([2023, 2024])]
    df_input = preprocess_common(pd.DataFrame([input_row]))
    df = pd.concat([df, df_input], ignore_index=True)
    df['Salle_attente_IOA'] = df.get('Salle_attente_IOA', 0)
    df['Salle_attente_MED'] = df.get('Salle_attente_MED', 0)
    df['nombre_patients_present'] = df.get('nombre_patients_present', 50)
    return df.tail(1)

def preprocess_for_hospitalisation(df, input_row):
    df = preprocess_common(df.copy())
    if 'Orientation' in df.columns:
        df['Hospitalisation'] = df['Orientation'].apply(
            lambda x: 1 if isinstance(x, str) and 'HOSP' in x.upper() else 0
        )
    df = df[df['Annee'].isin([2023, 2024])]

    df_input = preprocess_common(pd.DataFrame([input_row]))
    df_input['Hospitalisation'] = 0
    df_input["Age_Moyen_Sejour_Annees"] = df_input["AGE"]

    if "Date_Heure_PEC_IOA" in df_input.columns and "Date_Heure_Entree_Sejour" in df_input.columns:
        df_input["Delai_entree_IOA_min"] = (df_input["Date_Heure_PEC_IOA"] - df_input["Date_Heure_Entree_Sejour"]).dt.total_seconds() / 60
    if "Date_Heure_PEC_MED" in df_input.columns and "Date_Heure_PEC_IOA" in df_input.columns:
        df_input["Delai_IOA_MED_min"] = (df_input["Date_Heure_PEC_MED"] - df_input["Date_Heure_PEC_IOA"]).dt.total_seconds() / 60
    if "Date_Heure_PEC_MED" in df_input.columns and "Date_Heure_Entree_Sejour" in df_input.columns:
        df_input["Delai_entree_MED_min"] = (df_input["Date_Heure_PEC_MED"] - df_input["Date_Heure_Entree_Sejour"]).dt.total_seconds() / 60
    if "Date_Heure_Sortie_Urgences" in df_input.columns and "Date_Heure_PEC_MED" in df_input.columns:
        df_input["Delai_MED_sortie_min"] = (df_input["Date_Heure_Sortie_Urgences"] - df_input["Date_Heure_PEC_MED"]).dt.total_seconds() / 60
    if "Date_Heure_Sortie_Urgences" in df_input.columns and "Date_Heure_Entree_Sejour" in df_input.columns:
        df_input["Duree_totale_min"] = (df_input["Date_Heure_Sortie_Urgences"] - df_input["Date_Heure_Entree_Sejour"]).dt.total_seconds() / 60

    return df_input


def preprocess_for_moyenne(df):
    df = preprocess_common(df.copy())
    df = df[df['Annee'].isin([2023, 2024])]
    duree_moyenne = df['Duree_totale_heure'].mean()
    if pd.api.types.is_timedelta64_dtype(duree_moyenne):
        return duree_moyenne.total_seconds() / 60
    else:
        return duree_moyenne