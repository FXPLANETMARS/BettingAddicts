import os
import tempfile  # Dossier temporaire : les tests ne laissent aucun fichier derrière eux

import pytest
from historique import ajouter_historique, lire_historique

SAISIES = {"Sport": "Tennis", "Home_Team_Odds": "1.5", "Away_Team_Odds": "3.0", "Draw_Odds": "0.0", "Mois": "6"}


def test_lire_historique_absent():  # Pas de fichier : on retourne None
    assert lire_historique("nexiste_pas.csv") is None


def test_ajouter_puis_lire_historique():
    with tempfile.TemporaryDirectory() as dossier:
        chemin = os.path.join(dossier, "historique.csv")
        ajouter_historique(chemin, SAISIES, "Home", 52.0)
        ajouter_historique(chemin, SAISIES, "Away", 47.5)  # Le 2e ajout n'efface pas le 1er
        tableau = lire_historique(chemin)
        assert len(tableau) == 2
        assert tableau["Prediction"].iloc[0] == "Home"
        assert float(tableau["Confiance_pct"].iloc[1]) == pytest.approx(47.5)
