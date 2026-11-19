"""Historique des prédictions : écriture et lecture d'un fichier CSV."""
import os
from datetime import datetime

import pandas as pd

COLONNES_SAISIES = ["Sport", "Home_Team_Odds", "Away_Team_Odds", "Draw_Odds", "Mois"]


def ajouter_historique(chemin: str, saisies: dict[str, str], classe: str, confiance: float) -> None:
    """Ajoute une ligne (saisies, prédiction, confiance en %, date) à la fin du fichier CSV d'historique."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    nouveau = not os.path.exists(chemin)  # Fichier neuf : il faut écrire la ligne d'en-têtes
    with open(chemin, "a", encoding="utf-8") as fichier:  # "a" = ajouter à la fin sans effacer
        if nouveau:
            fichier.write(",".join(COLONNES_SAISIES + ["Prediction", "Confiance_pct", "Date"]) + "\n")
        valeurs = [saisies[colonne] for colonne in COLONNES_SAISIES]
        valeurs = valeurs + [classe, f"{confiance:.1f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        fichier.write(",".join(valeurs) + "\n")


def lire_historique(chemin: str) -> pd.DataFrame | None:
    """Retourne l'historique sous forme de tableau, ou None s'il n'existe pas encore."""
    if not os.path.exists(chemin):
        return None
    return pd.read_csv(chemin)
