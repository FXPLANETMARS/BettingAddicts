"""Interface Streamlit de BettingAddicts.

L'utilisateur choisit d'abord un sport, ensuite les deux équipes, puis les
informations du match. L'interface appelle le modèle et affiche le résultat.
"""
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from exceptions import DonneeInvalideError, ModeleIntrouvableError
from historique import ajouter_historique, lire_historique
from modele import EntraineurModele


# -----------------------------------------------------------------------------
# Fichiers du projet
# -----------------------------------------------------------------------------
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DATA_PATH = DOSSIER_PROJET / "donnees" / "sports_betting_predictive_analysis.csv"
MODEL_PATH = DOSSIER_PROJET / "modeles" / "modeles_paris.joblib"
METADATA_PATH = DOSSIER_PROJET / "modeles" / "metadonnees.json"
HISTORY_PATH = DOSSIER_PROJET / "historique.csv"

st.set_page_config(
    page_title="BettingAddicts",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Chargement des ressources
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_donnees() -> pd.DataFrame:
    """Charge le CSV utilisé pour proposer les sports et les équipes."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def charger_metadonnees() -> dict:
    """Charge les métadonnées qui décrivent les champs attendus par le modèle."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

    with open(METADATA_PATH, encoding="utf-8") as fichier:
        return json.load(fichier)


@st.cache_resource(show_spinner="Loading prediction model...")
def charger_modele() -> EntraineurModele:
    """Charge le modèle déjà entraîné, sans lancer un nouvel entraînement."""
    return EntraineurModele.charger(str(MODEL_PATH), str(METADATA_PATH))


try:
    df = charger_donnees()
    metadonnees = charger_metadonnees()
    modele = charger_modele()
except (OSError, ValueError, KeyError, ModeleIntrouvableError) as erreur:
    st.error(f"The application could not load its files: {erreur}")
    st.info("Check that the dataset, saved model and metadata files are present.")
    st.stop()

colonnes_requises = {"Sport", "Home_Team", "Away_Team"}
colonnes_absentes = colonnes_requises - set(df.columns)
if colonnes_absentes:
    st.error(f"The dataset is missing required columns: {', '.join(sorted(colonnes_absentes))}.")
    st.stop()

variables = metadonnees.get("variables", [])
if not variables:
    st.error("The model metadata does not contain any input variables.")
    st.stop()


# -----------------------------------------------------------------------------
# Fonctions d'aide pour les étapes et les résultats
# -----------------------------------------------------------------------------
def valeur_initiale(variable: dict, sport_supporte_nul: bool) -> float:
    """Choisit une valeur de départ pratique qui respecte les bornes du JSON."""
    nom = variable["nom"]
    minimum = float(variable["minimum"])
    maximum = float(variable["maximum"])

    if nom in ("Home_Team_Odds", "Away_Team_Odds"):
        valeur = 2.0
    elif nom == "Draw_Odds":
        valeur = 3.0 if sport_supporte_nul else 0.0
    elif nom == "Mois":
        valeur = float(datetime.now().month)
    else:
        valeur = (minimum + maximum) / 2

    return min(max(valeur, minimum), maximum)


def libelle_resultat(classe: str, equipe_domicile: str, equipe_exterieure: str) -> str:
    """Transforme la classe du modèle en texte compréhensible par l'utilisateur."""
    if classe == "Home":
        return f"Home win — {equipe_domicile}"
    if classe == "Away":
        return f"Away win — {equipe_exterieure}"
    return "Draw"


def afficher_probabilites(resultat: dict) -> None:
    """Affiche les probabilités sous forme de métriques et de graphique."""
    probabilites = resultat["probabilites"]
    categories = [
        ("Home", resultat["home_team"]),
        ("Draw", "Draw"),
        ("Away", resultat["away_team"]),
    ]
    valeurs = [probabilites.get(cle, 0.0) * 100 for cle, _ in categories]

    colonnes = st.columns(3)
    for colonne, (_, nom), valeur in zip(colonnes, categories, valeurs):
        with colonne:
            st.metric(nom, f"{valeur:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    barres = ax.bar([nom for _, nom in categories], valeurs)
    ax.set_title("Predicted match outcome probabilities")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for barre, valeur in zip(barres, valeurs):
        ax.text(
            barre.get_x() + barre.get_width() / 2,
            min(valeur + 2, 96),
            f"{valeur:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def afficher_historique() -> None:
    """Charge l'historique avec le module dédié et l'affiche dans l'interface."""
    try:
        historique = lire_historique(str(HISTORY_PATH))
    except (OSError, ValueError, pd.errors.ParserError) as erreur:
        st.warning(f"The prediction history could not be read: {erreur}")
        return

    if historique is None or historique.empty:
        st.info("No predictions have been saved yet.")
        return

    historique_affiche = historique.copy()

    if "Date" in historique_affiche.columns:
        historique_affiche["Date"] = pd.to_datetime(
            historique_affiche["Date"],
            errors="coerce",
        )
        historique_affiche = historique_affiche.sort_values(
            "Date",
            ascending=False,
            na_position="last",
        )
        historique_affiche["Date"] = historique_affiche["Date"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    noms_colonnes = {
        "Home_Team_Odds": "Home odds",
        "Away_Team_Odds": "Away odds",
        "Draw_Odds": "Draw odds",
        "Mois": "Month",
        "Home_Team": "Home team",
        "Away_Team": "Away team",
        "Home_Probability": "Home probability",
        "Draw_Probability": "Draw probability",
        "Away_Probability": "Away probability",
        "Prediction": "Predicted outcome",
        "Confiance_pct": "Predicted-class probability (%)",
    }
    historique_affiche = historique_affiche.rename(columns=noms_colonnes)

    st.caption(
        f"{len(historique_affiche)} prediction(s) saved. Newest predictions appear first."
    )
    st.dataframe(
        historique_affiche,
        use_container_width=True,
        hide_index=True,
    )


def afficher_progression(etape: int) -> None:
    """Montre où l'utilisateur se trouve dans le parcours de prédiction."""
    st.progress(etape / 3)
    st.caption(f"Step {etape} of 3")


# -----------------------------------------------------------------------------
# En-tête et informations générales
# -----------------------------------------------------------------------------
st.title("🏆 BettingAddicts")
st.caption("Sports match prediction platform")
st.markdown(
    "Follow three simple steps to explore a matchup and see the model's prediction."
)
st.info(
    "Educational project only. Predictions are estimates from a simple model, "
    "not betting advice or a guarantee of a match result."
)

sports = sorted(df["Sport"].dropna().astype(str).unique().tolist())
if not sports:
    st.error("No sports were found in the dataset.")
    st.stop()

with st.sidebar:
    st.header("About the model")
    exactitude = metadonnees.get("exactitude")
    reference = metadonnees.get("reference")

    if exactitude is not None:
        st.metric("Model accuracy", f"{float(exactitude) * 100:.1f}%")
    if reference is not None:
        st.metric("Baseline accuracy", f"{float(reference) * 100:.1f}%")

    st.caption(
        "Inputs: sport, home odds, away odds, draw odds and match month. "
        "Team names label the matchup; they are not model inputs."
    )


# Garder l'étape courante entre les reruns automatiques de Streamlit.
if "etape" not in st.session_state:
    st.session_state["etape"] = 1
if "sport_choisi" not in st.session_state:
    st.session_state["sport_choisi"] = sports[0]

etape = st.session_state["etape"]
afficher_progression(etape)


# -----------------------------------------------------------------------------
# Étape 1 : choisir le sport
# -----------------------------------------------------------------------------
if etape == 1:
    st.header("1. Choose your sport")
    sport_precedent = st.session_state["sport_choisi"]
    index_sport = sports.index(sport_precedent) if sport_precedent in sports else 0

    with st.form("sport_form", border=True):
        sport_choisi = st.selectbox(
            "Which sport would you like to analyze?",
            sports,
            index=index_sport,
            key="sport_form_choice",
        )
        continuer = st.form_submit_button(
            "Continue to team selection",
            type="primary",
            use_container_width=True,
        )

    if continuer:
        if sport_choisi != sport_precedent:
            # Les équipes et les anciennes saisies ne doivent pas être réutilisées
            # si l'utilisateur change de sport.
            for cle in (
                "home_team_choice",
                "away_team_choice",
                "derniere_prediction",
            ):
                st.session_state.pop(cle, None)
            for variable in variables:
                st.session_state.pop(f"input_{variable['nom']}", None)

        st.session_state["sport_choisi"] = sport_choisi
        st.session_state["etape"] = 2
        st.rerun()


# -----------------------------------------------------------------------------
# Étape 2 : choisir les équipes
# -----------------------------------------------------------------------------
if etape == 2:
    sport_choisi = st.session_state["sport_choisi"]
    donnees_sport = df[df["Sport"].astype(str) == sport_choisi]
    equipes = sorted(
        set(donnees_sport["Home_Team"].dropna().astype(str))
        | set(donnees_sport["Away_Team"].dropna().astype(str))
    )

    st.header("2. Choose your matchup")
    st.caption(f"Selected sport: **{sport_choisi}**")

    if len(equipes) < 2:
        st.error("At least two teams are required for the selected sport.")
    else:
        ancien_domicile = st.session_state.get("home_team_choice")
        ancien_exterieur = st.session_state.get("away_team_choice")
        index_domicile = equipes.index(ancien_domicile) if ancien_domicile in equipes else 0
        index_exterieur = equipes.index(ancien_exterieur) if ancien_exterieur in equipes else min(1, len(equipes) - 1)

        with st.form("teams_form", border=True):
            colonne_domicile, colonne_exterieure = st.columns(2)

            with colonne_domicile:
                equipe_domicile = st.selectbox(
                    "Home team",
                    equipes,
                    index=index_domicile,
                    key="home_team_choice",
                )

            with colonne_exterieure:
                equipe_exterieure = st.selectbox(
                    "Away team",
                    equipes,
                    index=index_exterieur,
                    key="away_team_choice",
                )

            colonne_retour, colonne_suivante = st.columns(2)
            with colonne_retour:
                retour = st.form_submit_button("Back", use_container_width=True)
            with colonne_suivante:
                continuer = st.form_submit_button(
                    "Continue to match information",
                    type="primary",
                    use_container_width=True,
                )

        if retour:
            st.session_state["etape"] = 1
            st.rerun()

        if continuer:
            if equipe_domicile == equipe_exterieure:
                st.error("Please select two different teams.")
            else:
                st.session_state["home_team_choice"] = equipe_domicile
                st.session_state["away_team_choice"] = equipe_exterieure
                st.session_state.pop("derniere_prediction", None)
                st.session_state["etape"] = 3
                st.rerun()


# -----------------------------------------------------------------------------
# Étape 3 : saisir les informations et prédire
# -----------------------------------------------------------------------------
if etape == 3:
    sport_choisi = st.session_state["sport_choisi"]
    equipe_domicile = st.session_state.get("home_team_choice")
    equipe_exterieure = st.session_state.get("away_team_choice")

    # Empêcher l'accès à cette étape si la sélection des équipes est incomplète.
    if not equipe_domicile or not equipe_exterieure:
        st.warning("Choose the sport and both teams before entering match information.")
        st.session_state["etape"] = 2
        st.rerun()

    donnees_sport = df[df["Sport"].astype(str) == sport_choisi]
    if "Draw_Odds" in donnees_sport.columns:
        cotes_nul = pd.to_numeric(donnees_sport["Draw_Odds"], errors="coerce")
        sport_supporte_nul = bool((cotes_nul > 0).any())
    else:
        sport_supporte_nul = False

    st.header("3. Match information and prediction")
    st.markdown(f"**{sport_choisi}** · {equipe_domicile} vs {equipe_exterieure}")

    with st.form("prediction_form", clear_on_submit=False, border=True):
        st.markdown("**Enter the match information**")
        st.caption("Each model input appears once. The fields are built from the saved model metadata.")

        saisies = {"Sport": sport_choisi}
        colonnes_saisie = st.columns(2)

        index_champ = 0
        for variable in variables:
            nom = variable["nom"]
            if nom == "Sport":
                continue

            conteneur = colonnes_saisie[index_champ % 2]
            index_champ += 1
            with conteneur:
                if variable["type"] == "categorielle":
                    saisies[nom] = st.selectbox(
                        nom.replace("_", " "),
                        variable["valeurs"],
                        key=f"input_{nom}",
                    )
                else:
                    minimum = float(variable["minimum"])
                    maximum = float(variable["maximum"])
                    valeur = valeur_initiale(variable, sport_supporte_nul)
                    est_mois = nom == "Mois"
                    cote_nul_inapplicable = nom == "Draw_Odds" and not sport_supporte_nul

                    if cote_nul_inapplicable:
                        st.number_input(
                            "Draw odds (not applicable for this sport)",
                            min_value=minimum,
                            max_value=maximum,
                            value=0.0,
                            step=0.05,
                            format="%.2f",
                            disabled=True,
                            key=f"input_{nom}",
                        )
                        saisies[nom] = 0.0
                    else:
                        saisies[nom] = st.number_input(
                            nom.replace("_", " "),
                            min_value=minimum,
                            max_value=maximum,
                            value=valeur,
                            step=1.0 if est_mois else 0.05,
                            format="%.0f" if est_mois else "%.2f",
                            key=f"input_{nom}",
                        )

        st.caption(
            "Team names label the result. The model uses sport, odds and month; "
            "it does not use team identities as input features."
        )

        colonne_retour, colonne_prediction = st.columns(2)
        with colonne_retour:
            retour = st.form_submit_button("Back to teams", use_container_width=True)
        with colonne_prediction:
            prediction_demandee = st.form_submit_button(
                "Predict match",
                type="primary",
                use_container_width=True,
            )

    if retour:
        st.session_state.pop("derniere_prediction", None)
        st.session_state["etape"] = 2
        st.rerun()

    if prediction_demandee:
        try:
            classe_predite, probabilites = modele.predire(saisies)

            derniere_prediction = {
                "sport": sport_choisi,
                "home_team": equipe_domicile,
                "away_team": equipe_exterieure,
                "classe": classe_predite,
                "probabilites": probabilites,
                "sport_supporte_nul": sport_supporte_nul,
            }
            st.session_state["derniere_prediction"] = derniere_prediction

            # La confiance enregistrée correspond à la probabilité de la classe prédite.
            confiance = probabilites.get(classe_predite, 0.0) * 100
            details_historique = {
                "Home_Team": equipe_domicile,
                "Away_Team": equipe_exterieure,
                "Home_Probability": probabilites.get("Home", 0.0),
                "Draw_Probability": probabilites.get("Draw", 0.0),
                "Away_Probability": probabilites.get("Away", 0.0),
            }

            try:
                ajouter_historique(
                    str(HISTORY_PATH),
                    saisies,
                    classe_predite,
                    confiance,
                    details_historique,
                )
                st.success("Prediction saved to history.")
            except (OSError, ValueError, pd.errors.ParserError) as erreur:
                st.warning(f"The prediction was generated, but could not be saved: {erreur}")

        except DonneeInvalideError as erreur:
            st.error(f"Please check the match information: {erreur}")
        except (KeyError, TypeError, ValueError) as erreur:
            st.error(f"The prediction could not be generated from these inputs: {erreur}")

    resultat = st.session_state.get("derniere_prediction")
    if resultat is not None:
        st.divider()
        st.subheader("Prediction result")

        with st.container(border=True):
            libelle = libelle_resultat(
                resultat["classe"],
                resultat["home_team"],
                resultat["away_team"],
            )

            if resultat["classe"] == "Draw" and not resultat["sport_supporte_nul"]:
                st.warning(
                    "The model returned Draw, but this sport has no draw odds in the training "
                    "dataset. One model is shared across several sports, so this result is a "
                    "model limitation rather than a possible match outcome."
                )
            elif resultat["classe"] == "Draw":
                st.info(f"Predicted outcome: {libelle}")
            else:
                st.success(f"Predicted outcome: {libelle}")

            confiance = resultat["probabilites"].get(resultat["classe"], 0.0) * 100
            st.metric("Probability of the predicted outcome", f"{confiance:.1f}%")

            if not resultat["sport_supporte_nul"] and resultat["classe"] != "Draw":
                st.caption(
                    "The model may still assign probability to Draw, even though the training "
                    "dataset has no draw odds for this sport. Interpret that value as a model limitation."
                )

            st.markdown("**Outcome probabilities**")
            afficher_probabilites(resultat)

        if st.button("Start a new prediction", type="primary"):
            st.session_state.pop("derniere_prediction", None)
            st.session_state["etape"] = 1
            st.rerun()


# -----------------------------------------------------------------------------
# Historique accessible depuis toutes les étapes
# -----------------------------------------------------------------------------
with st.expander("Prediction history", expanded=False):
    afficher_historique()
