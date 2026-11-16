import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the sports dataset
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "donnees"
    / "sports_betting_predictive_analysis.csv"
)

df = pd.read_csv(DATA_PATH)

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

st.write(f"You selected: {selected_sport}")

# Get the matches belonging to the selected sport
sport_data = df[df["Sport"] == selected_sport]

# Collect all available teams
teams = sorted(
    pd.concat([
        sport_data["Home_Team"],
        sport_data["Away_Team"]
    ]).dropna().unique()
)

# Select the two teams
st.header("Choose your matchup")

home_team = st.selectbox(
    "Home team:",
    teams
)

away_team = st.selectbox(
    "Away team:",
    teams
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
        st.subheader("Match summary")
        st.caption(f"Sport: {selected_sport}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=home_team,
                value=f"{home_odds:.2f}",
                help="Home team betting odds"
            )

        with col2:
            st.metric(
                label=away_team,
                value=f"{away_odds:.2f}",
                help="Away team betting odds"
            )

        # Display illustrative probabilities
        st.subheader("Illustrative win probabilities")

        teams = [home_team, away_team]
        probabilities = [60, 40]

        fig, ax = plt.subplots()

        ax.bar(teams, probabilities)

        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Match outcome probabilities")

        st.pyplot(fig)

        plt.close(fig)


