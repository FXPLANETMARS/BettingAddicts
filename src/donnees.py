import os

import pandas as pd
from sklearn.model_selection import train_test_split


class Variable:
    def __init__(self, nom, type_donnee=str):
        self.nom = nom
        self.type_donnee = type_donnee

    def valider_saisie(self, valeur):
        if pd.isna(valeur):
            return None

        try:
            return self.type_donnee(valeur)
        except (TypeError, ValueError):
            raise ValueError(f"La valeur '{valeur}' n'est pas compatible avec la variable '{self.nom}'.")


class VariableNumerique(Variable):
    def __init__(self, nom):
        super().__init__(nom, float)

    def valider_saisie(self, valeur):
        if pd.isna(valeur):
            return None

        try:
            return float(valeur)
        except (TypeError, ValueError):
            raise ValueError(f"La valeur '{valeur}' n'est pas numérique pour '{self.nom}'.")


class JeuDeDonnees:
    def __init__(self, chemin_fichier, variables=None):
        self.chemin_fichier = chemin_fichier
        self.variables = variables or []
        self.df = self.charger()

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
                return pd.read_csv(chemin)

        raise FileNotFoundError(f"Le fichier n'existe pas : {self.chemin_fichier}")

    def nettoyer(self):
        self.df = self.df.copy()

        for colonne in self.df.columns:
            self.df[colonne] = self.df[colonne].map(lambda x: str(x).strip() if pd.notna(x) else x)

        self.df = self.df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})
        return self.df

    def valider(self):
        if not self.variables:
            return self.df

        for variable in self.variables:
            if variable.nom not in self.df.columns:
                raise ValueError(f"La colonne '{variable.nom}' est absente du dataset.")

            self.df[variable.nom] = self.df[variable.nom].map(variable.valider_saisie)

        return self.df

    def separer_train_test(self, cible, test_size=0.2, random_state=42):
        if cible not in self.df.columns:
            raise KeyError(f"La cible '{cible}' n'existe pas dans le dataset.")

        X = self.df.drop(columns=[cible])
        y = self.df[cible]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)


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
    X_train, X_test, y_train, y_test = dataset2.separer_train_test("Actual_Winner")
    print(X_train.shape, X_test.shape)
