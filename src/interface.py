
# Importer les bibliothèques nécessaires
from pathlib import Path

import pandas as pd
import streamlit as st


# Charger le fichier CSV contenant les données
CHEMIN_CSV = (
    Path(__file__).resolve().parent.parent
    / "donnees"
    / "sports_betting_predictive_analysis.csv"
)

df = pd.read_csv(CHEMIN_CSV)


# Configurer l'application
st.set_page_config(
    page_title="BettingAddicts de Christian et Felix",
    page_icon="🏆",
    layout="centered"
)


# Ajouter le style de l'interface
st.markdown(
    """
    <style>
    .stApp {
        background-color: #101820;
    }

    h1, h2, h3, p, label {
        color: #F2F5F7;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initialiser la page actuelle
if "page" not in st.session_state:
    st.session_state.page = 1


# Afficher le titre de l'application
st.title("🏆 BettingAddicts de Christian et Felix")
st.caption("Sports prediction platform")
st.divider()


# PAGE 1 : Sélection du sport
if st.session_state.page == 1:

    st.caption("STEP 1 OF 3")
    st.progress(1 / 3)

    st.header("Choose your sport")
    st.write("Select the sport you want to analyze.")

    # Récupérer les sports disponibles dans le CSV
    sports = sorted(df["Sport"].dropna().unique())

    sport = st.selectbox(
        "Sport",
        sports
    )

    # Passer à la page suivante
    if st.button("Continue →", use_container_width=True):

        st.session_state.sport = sport
        st.session_state.page = 2
        st.rerun()


# PAGE 2 : Sélection des équipes
elif st.session_state.page == 2:

    st.caption("STEP 2 OF 3")
    st.progress(2 / 3)

    st.header("Choose your matchup")
    st.write(f"Selected sport: **{st.session_state.sport}**")

    # Filtrer les données selon le sport choisi
    donnees_sport = df[df["Sport"] == st.session_state.sport]

    # Récupérer toutes les équipes disponibles
    participants = sorted(
        set(donnees_sport["Home_Team"].dropna())
        | set(donnees_sport["Away_Team"].dropna())
    )

    # Choisir les deux participants
    home_team = st.selectbox(
        "Home team",
        participants
    )

    away_team = st.selectbox(
        "Away team",
        participants
    )

    col1, col2 = st.columns(2)

    # Revenir à la page précédente
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()

    # Vérifier les équipes et continuer
    with col2:
        if st.button("Continue →", use_container_width=True):

            if home_team == away_team:
                st.error("Choose two different participants.")

            else:
                st.session_state.home_team = home_team
                st.session_state.away_team = away_team

                st.session_state.page = 3
                st.rerun()


# PAGE 3 : Informations sur le match
elif st.session_state.page == 3:

    st.caption("STEP 3 OF 3")
    st.progress(1.0)

    st.header("Match odds")

    # Afficher les informations du match
    st.write(f"Sport: **{st.session_state.sport}**")
    st.write(f"Home team: **{st.session_state.home_team}**")
    st.write(f"Away team: **{st.session_state.away_team}**")

    st.divider()

    st.subheader("Enter match odds")

    # Entrer les cotes des deux équipes
    home_odds = st.number_input(
        "Home team odds",
        min_value=1.01,
        value=2.00,
        step=0.05
    )

    away_odds = st.number_input(
        "Away team odds",
        min_value=1.01,
        value=2.00,
        step=0.05
    )

    col1, col2 = st.columns(2)

    # Revenir à la sélection des équipes
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 2
            st.rerun()

    # Enregistrer les informations du match
    with col2:
        if st.button("Continue →", use_container_width=True):

            st.session_state.home_odds = home_odds
            st.session_state.away_odds = away_odds

            st.success("Match information saved!")
