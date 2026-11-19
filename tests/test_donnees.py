import pandas as pd  # Sert à créer une colonne pour decrire()
import pytest  # Importe pytest
from donnees import Variable, VariableNumerique, VariableCategorielle, creer_variable  # Classes testées
from exceptions import DonneeInvalideError  # Notre exception personnalisée


def test_numerique_valeur_normale():  # Une fonction qui commence par test_ est un test
    variable = VariableNumerique("cote", 1.0, 5.0)  # Variable avec l'intervalle 1 à 5
    assert variable.valider_saisie("2.5") == pytest.approx(2.5)  # approx évite les erreurs de précision des flottants


def test_numerique_limites_acceptees():  # Les bornes 1.0 et 5.0 sont acceptées
    variable = VariableNumerique("cote", 1.0, 5.0)
    assert variable.valider_saisie("1.0") == pytest.approx(1.0)
    assert variable.valider_saisie("5.0") == pytest.approx(5.0)


@pytest.mark.parametrize("saisie", ["abc", "9", "0.5", "nan"])  # Le test est exécuté 4 fois, une fois par valeur
def test_numerique_saisie_refusee(saisie):
    variable = VariableNumerique("cote", 1.0, 5.0)
    with pytest.raises(DonneeInvalideError):  # Réussit seulement si le bloc lève cette erreur
        variable.valider_saisie(saisie)  # Texte, trop grande, trop petite, nan


def test_categorielle_valeur_permise():  # « Tennis » est retourné tel quel
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.valider_saisie("Tennis") == "Tennis"


def test_categorielle_valeur_refusee():  # « Curling » lève DonneeInvalideError
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    with pytest.raises(DonneeInvalideError):
        variable.valider_saisie("Curling")


def test_encoder_categorielle():  # Encodage de deux sports avec ["Hockey", "Tennis"]
    variable = VariableCategorielle("Sport", ["Hockey", "Tennis"])
    assert variable.encoder(["Tennis", "Hockey"]) == [[0.0, 1.0], [1.0, 0.0]]


def test_encoder_numerique():  # "2.5" devient [[2.5]]
    variable = VariableNumerique("cote")
    resultat = variable.encoder(["2.5"])
    assert resultat[0][0] == pytest.approx(2.5)


def test_variable_abstraite_non_instanciable():  # On ne peut pas créer une Variable directement
    with pytest.raises(TypeError):
        Variable("x")


def test_str_variable():  # __str__ affiche le nom de la classe et de la variable
    assert str(VariableNumerique("Mois")) == "VariableNumerique(Mois)"


def test_decrire_puis_creer_variable():  # La fiche permet de recréer la variable avec son intervalle
    fiche = VariableNumerique("cote").decrire(pd.Series([1.2, 4.0, 2.5]))
    variable = creer_variable(fiche)
    assert variable.minimum == pytest.approx(1.2)
    assert variable.maximum == pytest.approx(4.0)
