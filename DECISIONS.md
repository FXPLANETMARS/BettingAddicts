# Fiche de décisions

Les colonnes « Options envisagées » reprennent ce qui a réellement été discuté. Les cases **À COMPLÉTER** sont à remplir
par l'équipe (en particulier les validations avec l'auxiliaire : n'inscrire une date que si la validation a eu lieu).

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire (date) |
|---|---|---|---|
| Domaine | À COMPLÉTER | Sport (paris sportifs) : prédire le résultat d'un match à partir des cotes. À COMPLÉTER (pourquoi ce domaine) | À COMPLÉTER |
| Jeu de données | À COMPLÉTER | `sports_betting_predictive_analysis.csv` : 1369 lignes (≥ 500), colonnes numériques et catégorielle, cible claire. Licence : À COMPLÉTER | À COMPLÉTER |
| Variable cible | 2 classes (Home / Away) ; 3 classes (Home / Away / Draw) | 3 classes : les nuls (environ 13 % des matchs) étaient classés « Away » à tort avec 2 classes | À COMPLÉTER |
| Métrique | À COMPLÉTER (autres : F1) | Exactitude, comparée à une référence (classe majoritaire). Simple à expliquer à un non-informaticien | À COMPLÉTER |
| Modèle | À COMPLÉTER (autres : arbre de décision, k plus proches voisins) | Régression logistique : simple, donne des probabilités pour chaque résultat | À COMPLÉTER |
| Technologie d'interface | À COMPLÉTER | Streamlit : fonctionne en local, champs adaptés au type, graphique Matplotlib intégré | À COMPLÉTER |
| Caractéristiques d'entrée | 2 cotes seulement ; 5 caractéristiques | 5 : sport, 3 cotes, mois (exigence : au moins 5, dont une catégorielle) | À COMPLÉTER |
| Nettoyage des valeurs manquantes | Supprimer toute ligne incomplète (443 lignes restantes) ; ne regarder que les colonnes essentielles | Suppression seulement si cote domicile, cote visiteur ou sport manque (1223 lignes). `Draw_Odds` vide = pas de nul possible, mis à 0 | À COMPLÉTER |
| Encodage du sport | Une case par sport (0/1) | Une case par sport permis, écrite avec des boucles `for` (vu en cours) | À COMPLÉTER |
| Séparation des données | À COMPLÉTER | 80 % / 20 % avec `random_state=42` : résultats reproductibles | À COMPLÉTER |
