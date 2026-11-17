class DonneeInvalideError(Exception):  # Nouvelle classe d'erreur, enfant de Exception
    pass  # Rien d'ajouté : seul le nom de l'exception compte


class ModeleIntrouvableError(Exception):  # Même chose, pour signaler un fichier de modèle manquant
    pass