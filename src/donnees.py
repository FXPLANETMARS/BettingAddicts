import os

import pandas as pd
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
            raise ValueError(f"La valeur '{valeur}' n'est pas compatible avec la variable '{self.nom}'.")


class VariableNumerique(Variable):  ##Sous classe specialisee pour les chiffres
    def __init__(self, nom):
        super().__init__(nom, float)    ##Force le type de donnee a etre un float au lieu de du texte

    def valider_saisie(self, valeur):
        if pd.isna(valeur):
            return None

        try:
            return float(valeur)
        except (TypeError, ValueError):
            raise ValueError(f"La valeur '{valeur}' n'est pas numérique pour '{self.nom}'.")


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

        self.df = self.df.dropna()  ## Supprime toutes les lignes du tableau qui contiennent au moins une case vide pour
                                    ## eviter de faire crasher l'algo. Chose qui est arrivee dans le passe.

        self.df["Resultat_Simplifie"] = self.df.apply(
            lambda ligne: "Home" if ligne["Actual_Winner"] == ligne["Home_Team"] else "Away",
            axis=1
        )

        return self.df

    def valider(self):
        if not self.variables:  ##Si aucune variables a surveiller retourne le tableau df sans rien modifier
            return self.df

        for variable in self.variables:
            if variable.nom not in self.df.columns:
                raise ValueError(f"La colonne '{variable.nom}' est absente du dataset.")

            self.df[variable.nom] = self.df[variable.nom].map(variable.valider_saisie)  ## .map(...) prend chaque case et
                                                                                        ## l'envoie a .valider_saisie()

        return self.df  ## Retourne la colonne nettoyee (tableau)

    def separer_train_test(self, cible, test_size=0.2, random_state=42):
        if cible not in self.df.columns:
            raise KeyError(f"La cible '{cible}' n'existe pas dans le dataset.")

        colonnes_to_keep = [var.nom for var in self.variables]
        X = self.df[colonnes_to_keep]   ## Prend uniquement les colonnes numeriques d'entrees (cotes) pour former les
                                        ## caracteristiques qu'analysera le modele
        y = self.df[cible]  ## Prend la colonne "reponse" que le modele doit pouvoir predire

        return train_test_split(X, y, test_size=test_size, random_state=random_state)   ## coupe les donnees en
        ## 4 paquets, 80% entrainement, 20% test. Le random_state garanti que le melange des lignes reste le meme a
        ## chaque execution


if __name__ == "__main__":
    dataset = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv")
    print(dataset.df.head())

    variables = [
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
    ]

    dataset2 = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv", variables)
    dataset2.nettoyer()
    dataset2.valider()
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")
    print(x_train.shape, x_test.shape)
