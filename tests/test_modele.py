import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from donnees import VariableCategorielle, VariableNumerique
from exceptions import DonneeInvalideError, ModeleIntrouvableError
from modele import EntraineurModele


def creer_modele_entraine():
    variables = [VariableCategorielle("Sport", ["A", "B"]), VariableNumerique("cote", 1.0, 5.0)]
    x = [[1, 0, 1.5], [0, 1, 4.0], [1, 0, 1.2], [0, 1, 4.5], [1, 0, 1.8], [0, 1, 3.9]]
    y = pd.Series(["Home", "Away", "Home", "Away", "Home", "Away"])
    modele = EntraineurModele(variables)
    modele.entrainer(x, y)
    return modele, x, y


def test_predire_probabilites_totalisent_un():
    modele, x, y = creer_modele_entraine()
    classe, probabilites = modele.predire({"Sport": "A", "cote": "1.4"})
    assert classe in ["Home", "Away"]
    assert sum(probabilites.values()) == pytest.approx(1.0)


def test_predire_saisie_non_numerique():
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": "beaucoup"})


def test_predire_champ_vide():
    modele, x, y = creer_modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"Sport": "A", "cote": ""})


def test_evaluer_egal_a_sklearn():
    modele, x, y = creer_modele_entraine()
    attendu = accuracy_score(y, modele.algorithme.predict(x))
    assert modele.evaluer(x, y) == pytest.approx(attendu)


def test_reference_classe_majoritaire():
    modele, x, y = creer_modele_entraine()
    classe, score = modele.calculer_reference(pd.Series(["Home", "Home", "Away"]), pd.Series(["Home", "Away"]))
    assert classe == "Home"
    assert score == pytest.approx(0.5)


def test_charger_modele_absent():
    with pytest.raises(ModeleIntrouvableError):
        EntraineurModele.charger("nexiste_pas.joblib", "nexiste_pas.json")