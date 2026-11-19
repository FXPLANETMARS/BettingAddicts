"""Exceptions personnalisées du projet."""


class DonneeInvalideError(Exception):
    """Levée quand une saisie ou une donnée ne respecte pas les règles attendues."""
    pass


class ModeleIntrouvableError(Exception):
    """Levée quand le modèle sauvegardé ou ses métadonnées sont introuvables."""
    pass
