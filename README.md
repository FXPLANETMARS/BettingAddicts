# BettingAddicts — prédire le résultat d'un match à partir des cotes

Projet 2 d'INF1007 (A2026), par Christian et Felix.

> ⚠️ **Exercice pédagogique.** Cet outil n'est PAS une aide à la décision pour parier.

## Ce que fait l'application
L'utilisateur choisit un sport, entre les cotes du match (domicile, visiteur, nul) et le mois.
Le modèle prédit le résultat le plus probable (victoire à domicile, victoire du visiteur ou match nul),
affiche la probabilité de chacun dans un graphique et garde un historique des prédictions.

## Données
- **Fichier :** `donnees/sports_betting_predictive_analysis.csv` (1369 lignes, 10 colonnes ; 1223 lignes après nettoyage).
- **Source et licence :** À COMPLÉTER (lien de la page du jeu de données et licence indiquée sur cette page).
- Aucune donnée personnelle. Les noms d'équipes semblent générés artificiellement (à vérifier auprès de la source).

## Installation et lancement
```
pip install -r requirements.txt
python src/main.py
```
`main.py` entraîne le modèle s'il n'existe pas encore (il n'est jamais ré-entraîné inutilement),
puis lance l'interface Streamlit dans le navigateur.

Tests et couverture :
```
pytest
coverage run -m pytest
coverage report
```
**Couverture mesurée :** 70 % (`coverage run --source=src -m pytest`, puis
`coverage report --omit="src/interface.py,src/main.py"`). Il s'agit de la couverture de la logique :
l'interface et le point d'entrée sont exclus, car les tests portent sur la logique hors interface.
Détail : `donnees.py` 59 %, `modele.py` 81 %, `historique.py` 100 %, `exceptions.py` 100 %.

## Structure du code
```
src/
├── exceptions.py   DonneeInvalideError, ModeleIntrouvableError
├── donnees.py      Variable (abstraite) → VariableNumerique / VariableCategorielle ; JeuDeDonnees
├── modele.py       EntraineurModele : entraîner, évaluer, prédire, sauvegarder, charger
├── historique.py   historique des prédictions (fichier CSV)
├── interface.py    interface Streamlit (appelle la logique, ne la contient pas)
└── main.py         point d'entrée
tests/              test_donnees.py, test_modele.py, test_historique.py
modeles/            modèle sauvegardé (.joblib), métadonnées (.json), graphique d'exploration, historique
```

## Modèle
- **Cible :** `Resultat_Simplifie` à 3 classes (Home / Away / Draw).
- **5 caractéristiques :** `Sport` (catégorielle), `Home_Team_Odds`, `Away_Team_Odds`, `Draw_Odds`, `Mois` (numériques).
- **Modèle :** régression logistique. **Métrique :** exactitude (proportion de bonnes réponses).
- **Séparation :** 80 % entraînement (978 matchs) / 20 % test (245 matchs), `random_state=42`.

| | Exactitude (test) |
|---|---|
| Modèle | **43,7 %** |
| Référence (toujours « Home », la classe la plus fréquente à l'entraînement) | 40,4 % |
| Hasard sur 3 classes | environ 33 % |

**Limites (analyse honnête).** Le gain sur la référence est modeste (+3,3 points) et nous n'avons pas testé s'il est significatif.
Sur les 245 matchs de test, le modèle prédit « Home » 192 fois. Il retrouve 86 des 99 victoires à domicile,
mais seulement 9 des 109 victoires du visiteur et 12 des 37 matchs nuls : il se trompe surtout sur « Away ».
Les cotes contiennent déjà l'information des sites de paris, et le choix des équipes n'entre pas dans le modèle.
Nous n'avons pas produit de matrice de confusion.

## Outils du chapitre 7
- **Décorateur `@chronometrer`** (`modele.py`) : affiche la durée de l'entraînement sans modifier la méthode `entrainer`.
- **Fonctions `lambda`** (`donnees.py`, `nettoyer`) : une petite fonction sans nom qui retire les espaces de chaque case,
  et une autre qui calcule le résultat (Home / Away / Draw) de chaque match.

## Documentation officielle consultée
À COMPLÉTER avec les pages que vous avez réellement consultées (au moins 3 : pandas, scikit-learn, Streamlit). Exemples de pages possibles :
- pandas : https://pandas.pydata.org/docs/user_guide/10min.html
- scikit-learn : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Streamlit : https://docs.streamlit.io
