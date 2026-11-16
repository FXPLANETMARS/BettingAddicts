import pytest
from donnees import VariableNumerique, VariableCategorielle
from exceptions import DonneeInvalideError


def test_numerique_valeur_normale():
    variable = VariableNumerique("cote", 1.0, 5.0)
    assert variable.valider_saisie("2.5") == pytest.approx(2.5)


def test_numerique_limites_acceptees():
    variable = VariableNumerique("cote", 1.0, 5.0)
    assert variable.valider_saisie("1.0") == pytest.approx(1.0)
    assert variable.valider_saisie("5.0") == pytest.approx(5.0)


@pytest.mark.parametrize("saisie", ["abc", "9", "0.5", "nan"])
def test_numerique_saisie_refusee(saisie):
    variable = VariableNumerique("cote", 1.0, 5.0)
    with pytest.raises(DonneeInvalideError):
        variable.valider_saisie(saisie)


def test_categorielle_valeur_permise():
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.valider_saisie("Tennis") == "Tennis"


def test_categorielle_valeur_refusee():
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    with pytest.raises(DonneeInvalideError):
        variable.valider_saisie("Curling")


def test_encoder_categorielle():
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.encoder(["Tennis", "Hockey"]) == [[0.0, 1.0], [1.0, 0.0]]


def test_encoder_numerique():
    variable = VariableNumerique("cote")
    resultat = variable.encoder(["2.5"])
    assert resultat[0][0] == pytest.approx(2.5)