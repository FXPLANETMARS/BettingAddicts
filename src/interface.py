# Fichier : src/interface.py

# Imports
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

from modele import EntraineurModele
from exceptions import DonneeInvalideError


# Project paths
DOSSIER_PROJET = Path(__file__).resolve().parent.parent

DATA_PATH = DOSSIER_PROJET / "donnees" / "sports_betting_predictive_analysis.csv"
MODEL_PATH = DOSSIER_PROJET / "modeles" / "modeles_paris.joblib"
METADATA_PATH = DOSSIER_PROJET / "modeles" / "metadonnees.json"
HISTORY_PATH = DOSSIER_PROJET / "historique.csv"

df = pd.read_csv(DATA_PATH)


# Load the trained model
@st.cache_resource
def charger_modele():
    return EntraineurModele.charger(
        str(MODEL_PATH),
        str(METADATA_PATH)
    )


modele = charger_modele()


# Load model metadata
with open(METADATA_PATH, encoding="utf-8") as fichier:
    metadonnees = json.load(fichier)

variables = metadonnees["variables"]


# Configure the application
st.set_page_config(
    page_title="BettingAddicts",
    page_icon="🏆",
    layout="centered"
)

# Display the application header
st.title("🏆 BettingAddicts")
st.caption("Sports match prediction platform")
st.divider()


# Sport selection
st.header("Choose your sport")

sports = sorted(df["Sport"].dropna().unique())

selected_sport = st.selectbox(
    "Select a sport:",
    sports
)

sport_data = df[df["Sport"] == selected_sport]


# Team selection
st.header("Choose your matchup")

teams = sorted(
    pd.concat([
        sport_data["Home_Team"],
        sport_data["Away_Team"]
    ]).dropna().unique()
)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home team:", teams)

with col2:
    away_team = st.selectbox("Away team:", teams)


if home_team == away_team:
    st.warning("Please select two different teams.")


# Match variables generated from metadata
st.divider()
st.header("Match information")

saisies = {}

for variable in variables:

    nom = variable["nom"]

    if nom == "Sport":
        saisies[nom] = selected_sport

    elif variable["type"] == "categorielle":
        saisies[nom] = st.selectbox(
            nom,
            variable["valeurs"]
        )

    else:
        saisies[nom] = st.number_input(
            nom,
            min_value=float(variable["minimum"]),
            max_value=float(variable["maximum"]),
            value=float(variable["minimum"]),
            step=1.0 if nom == "Mois" else 0.1
        )

# Prevent selecting the same team twice
if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    st.success(f"{home_team} vs {away_team}")

# Enter the betting odds
st.divider()

st.header("Enter match odds")

col1, col2 = st.columns(2)

with col1:
    home_odds = st.number_input(
        "Home team odds:",
        min_value=1.01,
        value=2.00,
        step=0.05
    )

with col2:
    away_odds = st.number_input(
        "Away team odds:",
        min_value=1.01,
        value=2.00,
        step=0.05
    )


# Introduction
st.subheader("Welcome")

st.write(
    "Analyze a sports matchup and explore its predicted outcome."
)

# Generate the prediction
st.divider()

if st.button("Predict match", use_container_width=True):

    if home_team == away_team:
        st.error("Please select two different teams.")

    else:
        try:
            # Get prediction from the trained model
            classe_predite, probabilites = modele.predire(saisies)

            # Display match summary
            st.subheader("Match summary")
            st.write(f"**{selected_sport}**")
            st.write(f"{home_team} vs {away_team}")

            # Display predicted result
            st.subheader("Predicted outcome")

            if classe_predite == "Home":
                st.success(f"Predicted winner: {home_team}")

            elif classe_predite == "Away":
                st.success(f"Predicted winner: {away_team}")

            else:
                st.info("Predicted outcome: Draw")

            # Display probabilities
            st.subheader("Predicted probabilities")

            labels = ["Home", "Draw", "Away"]

            valeurs = [
                probabilites.get("Home", 0) * 100,
                probabilites.get("Draw", 0) * 100,
                probabilites.get("Away", 0) * 100
            ]

            fig, ax = plt.subplots()

            ax.bar(labels, valeurs)

            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Match outcome probabilities")

            for i, valeur in enumerate(valeurs):
                ax.text(i, valeur + 1, f"{valeur:.1f}%", ha="center")

            st.pyplot(fig)

            plt.close(fig)

            # Save prediction to history
            historique = {
                "Sport": selected_sport,
                "Home_Team": home_team,
                "Away_Team": away_team,
                "Prediction": classe_predite,
                "Home_Probability": probabilites.get("Home", 0),
                "Draw_Probability": probabilites.get("Draw", 0),
                "Away_Probability": probabilites.get("Away", 0)
            }

            nouveau_match = pd.DataFrame([historique])

            if HISTORY_PATH.exists():
                ancien_historique = pd.read_csv(HISTORY_PATH)
                nouveau_match = pd.concat(
                    [ancien_historique, nouveau_match],
                    ignore_index=True
                )

            nouveau_match.to_csv(HISTORY_PATH, index=False)

            st.success("Prediction saved to history.")

        except DonneeInvalideError as erreur:
            st.error(str(erreur))

# Prediction history
st.divider()
st.header("Prediction history")

if HISTORY_PATH.exists():
    historique = pd.read_csv(HISTORY_PATH)

    st.dataframe(
        historique,
        use_container_width=True
    )

else:
    st.info("No predictions have been saved yet.")

