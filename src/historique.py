"""Historique des prédictions : écriture et lecture d'un fichier CSV."""
from datetime import datetime
from pathlib import Path

import pandas as pd

# Les caractéristiques réellement envoyées au modèle.
COLONNES_SAISIES = [
    "Sport",
    "Home_Team_Odds",
    "Away_Team_Odds",
    "Draw_Odds",
    "Mois",
]

# Informations utiles pour comprendre une prédiction après coup.
COLONNES_DETAILS = [
    "Home_Team",
    "Away_Team",
    "Home_Probability",
    "Draw_Probability",
    "Away_Probability",
]

COLONNES_HISTORIQUE = (
    COLONNES_SAISIES
    + COLONNES_DETAILS
    + ["Prediction", "Confiance_pct", "Date"]
)


def _normaliser_historique(tableau: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes absentes et retourne les colonnes dans l'ordre attendu.

    Cela permet de lire l'ancien historique, qui ne contenait pas encore
    les cotes, le mois ni la date, sans supprimer ses anciennes prédictions.
    """
    tableau = tableau.copy()

    for colonne in COLONNES_HISTORIQUE:
        if colonne not in tableau.columns:
            tableau[colonne] = pd.NA

    return tableau[COLONNES_HISTORIQUE]


def ajouter_historique(
    chemin: str,
    saisies: dict[str, object],
    classe: str,
    confiance: float,
    details: dict[str, object] | None = None,
) -> None:
    """Ajoute une prédiction au CSV sans effacer les entrées précédentes.

    Args:
        chemin: Chemin du fichier CSV d'historique.
        saisies: Valeurs des caractéristiques transmises au modèle.
        classe: Classe prédite (Home, Draw ou Away).
        confiance: Probabilité de la classe prédite, en pourcentage.
        details: Équipes et probabilités des trois résultats, facultatives.
    """
    chemin_fichier = Path(chemin)
    chemin_fichier.parent.mkdir(parents=True, exist_ok=True)

    nouvelle_entree = {
        colonne: saisies.get(colonne, pd.NA)
        for colonne in COLONNES_SAISIES
    }
    nouvelle_entree.update({
        colonne: (details or {}).get(colonne, pd.NA)
        for colonne in COLONNES_DETAILS
    })
    nouvelle_entree.update({
        "Prediction": classe,
        "Confiance_pct": round(float(confiance), 1),
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    nouvelle_ligne = pd.DataFrame(
        [nouvelle_entree],
        columns=COLONNES_HISTORIQUE,
    )

    if chemin_fichier.exists():
        ancien_historique = pd.read_csv(chemin_fichier)
        ancien_historique = _normaliser_historique(ancien_historique)
        # Reconstruire le tableau évite les avertissements de concaténation
        # lorsque l'ancien fichier ne possède pas encore certaines colonnes.
        anciennes_lignes = ancien_historique.to_dict(orient="records")
        historique_complet = pd.DataFrame(
            anciennes_lignes + [nouvelle_entree],
            columns=COLONNES_HISTORIQUE,
        )
    else:
        historique_complet = nouvelle_ligne

    # pandas gère correctement les nombres, les virgules et les valeurs manquantes.
    historique_complet.to_csv(chemin_fichier, index=False, encoding="utf-8")


def lire_historique(chemin: str) -> pd.DataFrame | None:
    """Retourne l'historique sous forme de tableau, ou None s'il est absent."""
    chemin_fichier = Path(chemin)

    if not chemin_fichier.exists():
        return None

    historique = pd.read_csv(chemin_fichier)
    return _normaliser_historique(historique)
