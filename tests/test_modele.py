import pandas as pd  # Sert à créer des Series
import pytest
from sklearn.metrics import accuracy_score  # L'exactitude de scikit-learn, pour comparer
from donnees import VariableCategorielle, VariableNumerique
from exceptions import DonneeInvalideError, ModeleIntrouvableError
from modele import EntraineurModele


def creer_modele_entraine():  # Fonction d'aide (pas un test) qui entraîne un petit modèle
    variables = [VariableCategorielle("Sport", ["A", "B"]), VariableNumerique("cote", 1.0, 5.0)]  # 2 variables
    x = [[1, 0, 1.5], [0, 1, 4.0], [1, 0, 1.2], [0, 1, 4.5], [1, 0, 1.8], [0, 1, 3.9]]  # 6 matchs inventés, déjà encodés (2 cases pour le sport, puis la cote)
    y = pd.Series(["Home", "Away", "Home", "Away", "Home", "Away"])  # Leurs 6 résultats
    modele = EntraineurModele(variables)
    modele.entrainer(x, y)  # Entraîne le modèle
    return modele, x, y  # Retourne les trois


def test_predire_probabilites_totalisent_un():
    modele, x, y = creer_modele_entraine()  # Récupère les trois dans chaque test
    classe, probabilites = modele.predire({"Sport": "A", "cote": "1.4"})
    assert classe in ["Home", "Away"]  # Vérifie une classe valide
    assert sum(probabilites.values()) == pytest.approx(1.0)  # Vérifie que les probabilités totalisent 1


def test_predire_saisie_non_numerique():  # "beaucoup" lève DonneeInvalideError
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": "beaucoup"})


def test_predire_champ_vide():  # "" lève DonneeInvalideError
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": ""})


def test_evaluer_egal_a_sklearn():
    modele, x, y = creer_modele_entraine()
    attendu = accuracy_score(y, modele.algorithme.predict(x))  # L'exactitude calculée par scikit-learn
    assert modele.evaluer(x, y) == pytest.approx(attendu)  # Vérifie que notre calcul donne la même chose


def test_reference_classe_majoritaire():
    modele, x, y = creer_modele_entraine()
    # Avec ["Home", "Home", "Away"] à l'entraînement, la classe majoritaire est « Home », et elle a raison pour 1 match sur 2 dans le test (0.5)
    classe, score = modele.calculer_reference(pd.Series(["Home", "Home", "Away"]), pd.Series(["Home", "Away"]))
    assert classe == "Home"
    assert score == pytest.approx(0.5)


def test_charger_modele_absent():  # Charger des fichiers inexistants lève ModeleIntrouvableError
    with pytest.raises(ModeleIntrouvableError):
        EntraineurModele.charger("nexiste_pas.joblib", "nexiste_pas.json")