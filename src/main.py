"""Point d'entrée : entraîne le modèle s'il n'existe pas encore, puis lance l'interface Streamlit."""
import os
import sys

from donnees import JeuDeDonnees, VariableNumerique, VariableCategorielle, tracer_resultats
from modele import EntraineurModele

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Dossier du projet
CHEMIN_CSV = os.path.join(RACINE, "donnees", "sports_betting_predictive_analysis.csv")
CHEMIN_MODELE = os.path.join(RACINE, "modeles", "modeles_paris.joblib")
CHEMIN_JSON = os.path.join(RACINE, "modeles", "metadonnees.json")
CHEMIN_GRAPHIQUE = os.path.join(RACINE, "modeles", "exploration.png")
CHEMIN_INTERFACE = os.path.join(RACINE, "src", "interface.py")


def lancer_entrainement() -> None:
    """Nettoie les données, entraîne et évalue le modèle, puis sauvegarde le modèle et ses métadonnées."""
    os.makedirs(os.path.join(RACINE, "modeles"), exist_ok=True)

    variables = [  # Les 5 caractéristiques
        VariableCategorielle("Sport", ["Baseball", "Basketball", "Football", "Hockey", "Tennis"]),
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
        VariableNumerique("Draw_Odds"),
        VariableNumerique("Mois"),
    ]

    dataset = JeuDeDonnees(CHEMIN_CSV, variables)
    dataset.nettoyer()
    tracer_resultats(dataset.df, CHEMIN_GRAPHIQUE)
    dataset.valider()
    x_train, x_test, y_train, y_test = dataset.separer_train_test("Resultat_Simplifie")

    modele = EntraineurModele(variables)
    modele.entrainer(x_train, y_train)
    exactitude = modele.evaluer(x_test, y_test)
    majoritaire, reference = modele.calculer_reference(y_train, y_test)

    modele.sauvegarder(CHEMIN_MODELE)
    modele.sauvegarder_json(CHEMIN_JSON, dataset.decrire_variables(), exactitude, reference)

    print(f"Exactitude : {exactitude * 100:.1f} %")
    print(f"Reference (toujours {majoritaire}) : {reference * 100:.1f} %")


def modele_existe() -> bool:
    """Vrai si le modèle et ses métadonnées ont déjà été sauvegardés."""
    return os.path.exists(CHEMIN_MODELE) and os.path.exists(CHEMIN_JSON)


def main() -> None:
    """Entraîne le modèle s'il est absent (jamais de ré-entraînement inutile), puis lance l'interface."""
    if not modele_existe():
        print("Modele absent : entrainement en cours...")
        lancer_entrainement()
    os.system(f'"{sys.executable}" -m streamlit run "{CHEMIN_INTERFACE}"')


if __name__ == "__main__":
    main()
