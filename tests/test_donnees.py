import pytest  # Importe pytest
from donnees import VariableNumerique, VariableCategorielle  # Importe les classes de variables testées
from exceptions import DonneeInvalideError  # Importe notre exception personnalisée


def test_numerique_valeur_normale():  # Une fonction qui commence par test_ est un test
    variable = VariableNumerique("cote", 1.0, 5.0)  # Crée une variable avec l'intervalle 1 à 5
    assert variable.valider_saisie("2.5") == pytest.approx(2.5)  # Vérifie que "2.5" devient 2.5 (approx évite les erreurs de précision des flottants)


def test_numerique_limites_acceptees():  # Vérifie que les bornes 1.0 et 5.0 sont acceptées
    variable = VariableNumerique("cote", 1.0, 5.0)
    assert variable.valider_saisie("1.0") == pytest.approx(1.0)
    assert variable.valider_saisie("5.0") == pytest.approx(5.0)


@pytest.mark.parametrize("saisie", ["abc", "9", "0.5", "nan"])  # Exécute le test 4 fois, une fois par valeur de la liste
def test_numerique_saisie_refusee(saisie):
    variable = VariableNumerique("cote", 1.0, 5.0)
    with pytest.raises(DonneeInvalideError):  # Le test réussit seulement si le bloc lève cette erreur
        variable.valider_saisie(saisie)  # La saisie refusée (texte, trop grande, trop petite, nan)


def test_categorielle_valeur_permise():  # « Tennis » est retourné tel quel
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.valider_saisie("Tennis") == "Tennis"


def test_categorielle_valeur_refusee():  # « Curling » lève DonneeInvalideError
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    with pytest.raises(DonneeInvalideError):
        variable.valider_saisie("Curling")


def test_encoder_categorielle():  # Vérifie l'encodage de deux sports avec ["Hockey", "Tennis"]
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.encoder(["Tennis", "Hockey"]) == [[0.0, 1.0], [1.0, 0.0]]


def test_encoder_numerique():  # Vérifie que "2.5" devient [[2.5]]
    variable = VariableNumerique("cote")
    resultat = variable.encoder(["2.5"])
    assert resultat[0][0] == pytest.approx(2.5)  # On compare resultat[0][0] avec approx