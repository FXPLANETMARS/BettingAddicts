
from pathlib import Path

import pandas as pd
import streamlit as st


# Charger les données
CHEMIN_CSV = (
    Path(__file__).resolve().parent.parent
    / "donnees"
    / "sports_betting_predictive_analysis.csv"
)

df = pd.read_csv(CHEMIN_CSV)


# Configuration de l'application
st.set_page_config(
    page_title="BettingAddicts",
    page_icon="🏆",
    layout="centered"
)

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


# En-tête
st.title("🏆 BettingAddicts")
st.caption("Sports prediction platform")
st.divider()


# PAGE 1 : Sélection du sport
if st.session_state.page == 1:

    st.caption("STEP 1 OF 2")
    st.progress(0.5)

    st.header("Choose your sport")
    st.write("Select the sport you want to analyze.")

    sports = sorted(df["Sport"].dropna().unique())

    sport = st.selectbox(
        "Sport",
        sports
    )

    if st.button("Continue →", use_container_width=True):
        st.session_state.sport = sport
        st.session_state.page = 2
        st.rerun()


# PAGE 2 : Sélection des équipes
elif st.session_state.page == 2:

    st.caption("STEP 2 OF 2")
    st.progress(1.0)

    st.header("Choose your matchup")
    st.write(f"Selected sport: **{st.session_state.sport}**")

    donnees_sport = df[df["Sport"] == st.session_state.sport]

    participants = sorted(
        set(donnees_sport["Home_Team"].dropna())
        | set(donnees_sport["Away_Team"].dropna())
    )

    home_team = st.selectbox(
        "Home team",
        participants
    )

    away_team = st.selectbox(
        "Away team",
        participants
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()

    with col2:
        if st.button("Continue →", use_container_width=True):

            if home_team == away_team:
                st.error("Choose two different participants.")

            else:
                st.session_state.home_team = home_team
                st.session_state.away_team = away_team

                st.success("Match selected successfully!")
