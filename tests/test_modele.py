import pandas as pd  # Sert à créer des Series
import pytest
from sklearn.metrics import accuracy_score  # L'exactitude de scikit-learn, pour comparer
from donnees import VariableCategorielle, VariableNumerique
from exceptions import DonneeInvalideError, ModeleIntrouvableError
from modele import EntraineurModele


def creer_modele_entraine():  # Fonction d'aide (pas un test) qui entraîne un petit modèle
    variables = [VariableCategorielle("Sport", ["A", "B"]), VariableNumerique("cote", 1.0, 5.0)]
    x = [[1, 0, 1.5], [0, 1, 4.0], [1, 0, 1.2], [0, 1, 4.5], [1, 0, 1.8],
         [0, 1, 3.9]]  # 6 matchs inventés, déjà encodés
    y = pd.Series(["Home", "Away", "Home", "Away", "Home", "Away"])  # Leurs 6 résultats
    modele = EntraineurModele(variables)
    modele.entrainer(x, y)
    return modele, x, y


def test_predire_probabilites_totalisent_un():
    modele, x, y = creer_modele_entraine()
    classe, probabilites = modele.predire({"Sport": "A", "cote": "1.4"})
    assert classe in ["Home", "Away"]  # Classe valide
    assert sum(probabilites.values()) == pytest.approx(1.0)  # Les probabilités totalisent 1


def test_predire_saisie_non_numerique():  # "beaucoup" lève DonneeInvalideError
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": "beaucoup"})


def test_predire_champ_vide():  # "" lève DonneeInvalideError
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": ""})


def test_predire_hors_intervalle():  # 9 dépasse l'intervalle 1 à 5
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": "9"})


def test_evaluer_egal_a_sklearn():
    modele, x, y = creer_modele_entraine()
    attendu = accuracy_score(y, modele.algorithme.predict(x))  # L'exactitude calculée par scikit-learn
    assert modele.evaluer(x, y) == pytest.approx(attendu)  # Notre calcul donne la même chose


def test_reference_classe_majoritaire():
    modele, x, y = creer_modele_entraine()
    # « Home » est la plus fréquente (2 contre 1) et a raison pour 1 match sur 2 dans le test
    classe, score = modele.calculer_reference(pd.Series(["Home", "Home", "Away"]), pd.Series(["Home", "Away"]))
    assert classe == "Home"
    assert score == pytest.approx(0.5)


def test_charger_modele_absent():  # Des fichiers inexistants lèvent ModeleIntrouvableError
    with pytest.raises(ModeleIntrouvableError):
        EntraineurModele.charger("nexiste_pas.joblib", "nexiste_pas.json")
