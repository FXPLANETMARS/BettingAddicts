import os

import pandas as pd
import matplotlib.pyplot as plt
from exceptions import DonneeInvalideError, ModeleIntrouvableError
from sklearn.model_selection import train_test_split


class Variable:
    def __init__(self, nom, type_donnee=str):
        self.nom = nom
        self.type_donnee = type_donnee

    def valider_saisie(self, valeur):   ##Verifie si un cellule du dataset contient une valeur valide
        if pd.isna(valeur): ##Fonction de panda qui permet de savoir si une cellule est vide
            return None

        try:    ##Essaie de convertir la valeur dans le type attendu
            return self.type_donnee(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(f"La valeur '{valeur}' n'est pas compatible avec la variable '{self.nom}'.")


class VariableNumerique(Variable):
    def __init__(self, nom, minimum=None, maximum=None):
        super().__init__(nom, float)
        self.minimum = minimum
        self.maximum = maximum

    def valider_saisie(self, valeur):
        if pd.isna(valeur):
            return None

        try:
            nombre = float(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(f"La valeur '{valeur}' n'est pas numérique pour '{self.nom}'.")

        if self.minimum is not None and not self.minimum <= nombre <= self.maximum:
            raise DonneeInvalideError(f"'{self.nom}' doit être entre {self.minimum} et {self.maximum}.")

        return nombre

    def decrire(self, colonne):
        return {"nom": self.nom, "type": "numerique",
                "minimum": float(colonne.min()), "maximum": float(colonne.max())}

    def encoder(self, valeurs):
        resultat = []
        for valeur in valeurs:
            resultat.append([float(valeur)])
        return resultat

class VariableCategorielle(Variable):  ##Sous classe pour les categories (ex: le sport)
    def __init__(self, nom, valeurs_permises):
        super().__init__(nom, str)
        self.valeurs_permises = valeurs_permises    ##Liste des valeurs acceptees (ex: ["Tennis", "Hockey"])

    def valider_saisie(self, valeur):
        if pd.isna(valeur):
            return None

        if valeur not in self.valeurs_permises:
            raise DonneeInvalideError(f"La valeur '{valeur}' n'est pas permise pour '{self.nom}'.")

        return valeur

    def decrire(self, colonne):
        return {"nom": self.nom, "type": "categorielle", "valeurs": self.valeurs_permises}

    def encoder(self, valeurs):
        resultat = []
        for valeur in valeurs:
            ligne = []
            for permise in self.valeurs_permises:
                if valeur == permise:
                    ligne.append(1.0)
                else:
                    ligne.append(0.0)
            resultat.append(ligne)
        return resultat

def creer_variable(description):
    if description["type"] == "numerique":
        return VariableNumerique(description["nom"], description["minimum"], description["maximum"])
    return VariableCategorielle(description["nom"], description["valeurs"])

class JeuDeDonnees: ##Classe qui gere le dataset CSV
    def __init__(self, chemin_fichier, variables=None):
        self.chemin_fichier = chemin_fichier
        self.variables = variables or []
        self.df = self.charger()    ## Des que l'objet est cree, il est charge pour ouvrir le fichier et
                                    ## stocker le tableau dans la variable DataFrame as df

    def charger(self):
        chemins = []

        if os.path.isabs(self.chemin_fichier):
            chemins.append(self.chemin_fichier)
        else:
            chemins.extend([
                os.path.join(os.getcwd(), self.chemin_fichier),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), self.chemin_fichier),
                os.path.join(os.path.dirname(__file__), self.chemin_fichier),
            ])

        for chemin in chemins:
            if os.path.exists(chemin):
                self.chemin_fichier = chemin
                return pd.read_csv(chemin)  ## methode de pandas qui permet de lire le dataset CSV et qui le
                                            ## transforme en tableau

        raise FileNotFoundError(f"Le fichier n'existe pas : {self.chemin_fichier}")

    def nettoyer(self):
        self.df = self.df.copy()    ##Evite de modifier sur l'original par accident

        for colonne in self.df.columns:
            self.df[colonne] = self.df[colonne].map(lambda x: str(x).strip() if pd.notna(x) else x)

        self.df = self.df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})  ##.replace(...) remplace les
                                                                                            ## mots vides par le format
                                                                                            ## officiel de pandas pour
                                                                                            ## les donnees manquantes

        self.df = self.df.dropna(subset=["Home_Team_Odds", "Away_Team_Odds", "Sport"])
        ## Supprime seulement les lignes où une colonne essentielle est vide
        ## (Draw_Odds est souvent vide, car le nul n'existe pas dans certains sports)

        self.df["Draw_Odds"] = self.df["Draw_Odds"].fillna("0.0")

        self.df["Mois"] = pd.to_datetime(self.df["Date"]).dt.month

        self.df["Resultat_Simplifie"] = self.df.apply(
            lambda ligne: "Home" if (ligne["Actual_Winner"] == ligne["Home_Team"])
            else ("Away" if (ligne["Actual_Winner"] == ligne["Away_Team"]) else ("Draw")),
            axis=1
        )

        return self.df

    def valider(self):
        if not self.variables:  ##Si aucune variables a surveiller retourne le tableau df sans rien modifier
            return self.df

        for variable in self.variables:
            if variable.nom not in self.df.columns:
                raise DonneeInvalideError(f"La colonne '{variable.nom}' est absente du dataset.")

            self.df[variable.nom] = self.df[variable.nom].map(variable.valider_saisie)  ## .map(...) prend chaque case et
                                                                                        ## l'envoie a .valider_saisie()

        return self.df  ## Retourne la colonne nettoyee (tableau)

    def separer_train_test(self, cible, test_size=0.2, random_state=42):
        if cible not in self.df.columns:
            raise KeyError(f"La cible '{cible}' n'existe pas dans le dataset.")

        colonnes_encodees = [var.encoder(self.df[var.nom]) for var in self.variables]
        X = []
        for i in range(len(self.df)):
            ligne = []
            for colonne in colonnes_encodees:
                ligne = ligne + colonne[i]
            X.append(ligne)

        y = self.df[cible]  ## Prend la colonne "reponse" que le modele doit pouvoir predire

        return train_test_split(X, y, test_size=test_size, random_state=random_state)   ## coupe les donnees en
        ## 4 paquets, 80% entrainement, 20% test. Le random_state garanti que le melange des lignes reste le meme a
        ## chaque execution

    def decrire_variables(self):
        description = []
        for variable in self.variables:
            description.append(variable.decrire(self.df[variable.nom]))
        return description

def tracer_resultats(df, chemin_image):
    comptes = df["Resultat_Simplifie"].value_counts()
    plt.bar(comptes.index, comptes.values)
    plt.title("Nombre de matchs selon le résultat")
    plt.xlabel("Résultat")
    plt.ylabel("Nombre de matchs")
    plt.savefig(chemin_image, dpi=150)
    plt.show()

if __name__ == "__main__":
    dataset = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv")
    print(dataset.df.columns.tolist())
    print(dataset.df.head())

    variables = [
        VariableCategorielle("Sport", ["Baseball", "Basketball", "Football", "Hockey", "Tennis"]),
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
        VariableNumerique("Draw_Odds"),
        VariableNumerique("Mois"),
    ]

    dataset2 = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv", variables)

    dataset2.nettoyer()
    tracer_resultats(dataset2.df, "modeles/exploration.png")
    print(len(dataset2.df))
    print(dataset2.df["Resultat_Simplifie"].value_counts())
    dataset2.valider()
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")
    print(len(x_train), len(x_test))
