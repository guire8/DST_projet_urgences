# Projet Urgences — Présentation DataScientest

Cette application Streamlit permet de :
1. Visualiser et explorer les données des urgences
2. Appliquer les transformations sur le fichier source `liste_sejours_new.xlsx`
3. Tester différents modèles de machine learning entraînés (temps d’attente, hospitalisation, estimation moyenne)

## Structure

```
projet_urgences/
├── app.py
├── data/
│   └── liste_sejours_new.xlsx  # à placer ici
├── models/
│   └── model_*.pkl             # modèles entraînés à placer ici
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

## Lancement de l'application

```bash
streamlit run app.py
```

## Auteurs

Cansu, Marine, Pascal, Rémi – Projet DataScientest