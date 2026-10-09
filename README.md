# UE8 : Prédire la direction du Bitcoin avec le machine learning

Projet individuel bonus, échéance du 9 octobre 2026. Auteur : à compléter.

## Objectif

À la clôture du jour J, prédire si le cours de clôture de J+1 sera strictement
supérieur à celui de J. La classe 1 signifie « hausse », la classe 0 « baisse
ou stabilité ». Il s'agit de **classification**, pas de prédiction du prix exact.
Toutes les variables du jour J supposent que cette journée est terminée.

## Installation et exécution

Depuis le dossier du projet, avec Python 3.10 ou plus récent :

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

Sous macOS/Linux, remplacer `.venv\Scripts\python.exe` par `.venv/bin/python`.
`requirements-lock.txt` conserve les versions exactes utilisées lors de la
vérification avec Python 3.13. Pour reproduire cet environnement, installer
ce fichier à la place de `requirements.txt`.
Les chemins par défaut sont relatifs au script, même si on le lance depuis un
autre dossier. Pour utiliser un autre fichier :

```powershell
.venv\Scripts\python.exe main.py --data "chemin/bitcoin.csv" --output "results"
```

## Organisation

```text
main.py                 Lance l'expérience et enregistre le rapport
bitcoin_ml/
    __init__.py
    data.py             Lecture, contrôles, variables et découpage temporel
    models.py           Apprentissage, sélection et mesures de performance
    diagnostics.py      Validation temporelle des modèles fixes sur cinq plis
    plots.py            Création des graphiques
data/bitcoin.csv         Dataset fourni pour l'exercice
results/                Rapport, scores, prédictions de test et graphiques
requirements.txt        Dépendances Python
tests/test_data.py       Vérification de l'absence de fuite temporelle
```

## Données et méthode

Le CSV fourni contient des données publiques de marché : Date, Open, High,
Low, Close, Adj Close et Volume. La provenance amont exacte n'est pas certifiée
par ce projet. La colonne Adj Close n'est pas utilisée. Le programme contrôle
les dates, les valeurs manquantes, les prix et la continuité quotidienne. Il
refuse les données incohérentes au lieu d'inventer des cours manquants.

Les variables décrivent les rendements sur 1, 7 et 14 jours, l'amplitude de la
journée, la variation ouverture/clôture, l'écart aux moyennes mobiles sur 7 et
30 jours, la volatilité sur 7 jours et le logarithme du volume. Les premières
lignes sans historique suffisant et la dernière sans cible connue sont exclues.

Trois approches sont comparées : une référence prédisant la classe majoritaire,
une régression logistique avec standardisation et une forêt aléatoire de
profondeur limitée. Les paramètres sont fixés avant l'évaluation.

Les données sont séparées chronologiquement en environ 60 % d'apprentissage,
20 % de validation et 20 % de test. Une observation est retirée avant chaque
frontière, car sa cible dépend du lendemain. La standardisation est apprise
sur les données d'entraînement grâce à un Pipeline. Le modèle est choisi sur
la ROC-AUC de validation, puis réentraîné sur apprentissage et validation.
Le test final évalue seulement ce choix et la référence majoritaire.

## Lire les résultats

Pour lancer les quatre tests de cohérence temporelle :

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- `rapport.txt` : périodes, effectifs, scores et interprétation.
- `metrics.csv` : exactitude, exactitude équilibrée, précision, rappel, F1 de la hausse et ROC-AUC.
- `validation_temporelle.csv` : dates des cinq plis, scores et tailles des ensembles.
- `predictions.csv` : cible, classe prédite et probabilité de hausse sur le test.
  La date correspond au jour J ; la cible concerne le lendemain.
- Les cinq PNG montrent l'historique, la comparaison, l'évaluation finale,
  les corrélations et classes du train, puis la stabilité temporelle.

Une AUC proche de 0,5 indique une faible discrimination. L'exactitude doit être
comparée à la référence majoritaire. Un écart important entre apprentissage et
validation suggère du surapprentissage. De mauvais résultats restent un résultat
valide : on ne change pas les paramètres après avoir observé le test pour
améliorer artificiellement les scores.

### Résultats de l'exécution sur le CSV fourni

Le fichier contient 2 713 jours, du 17 septembre 2014 au 19 février 2022.
La forêt aléatoire est retenue avec une AUC de validation de 0,558. Sur les
537 observations de test, son AUC vaut 0,531 et son exactitude 52,9 %, contre
53,8 % pour la référence majoritaire. Elle ne dépasse donc pas la référence
en exactitude. Son AUC d'apprentissage de 0,740, nettement supérieure à celle
de validation, suggère du surapprentissage. Le signal prédictif reste faible ;
ces résultats ne démontrent pas une prévision fiable du marché.

Le projet ne mesure pas une performance de trading. Il ignore notamment les
frais. Le diagnostic sur cinq périodes historiques ne garantit pas la stabilité
future. Toute nouvelle optimisation nécessiterait un nouveau test final réservé
à l'avance : le test actuel a déjà été consulté dans la première version.

## Liens avec le cours UE8

Ajouts guidés par les fiches F02 (prévention des fuites), F03 (métriques) et
F04 (validation et surapprentissage), ainsi que les notebooks Iris de l'activité 1
(`classification_report`) et Housing V2 de l'activité 2 (validation croisée).

- **Validation croisée adaptée au temps** : `TimeSeriesSplit(n_splits=5, gap=1)`
  sur les 60 % initiaux uniquement. Chaque modèle apprend sur un passé croissant
  et valide sur le bloc suivant ; chaque Pipeline réapprend son scaler dans le pli.
  Les scores moyens et leur écart-type décrivent la stabilité entre périodes,
  sans être un intervalle de confiance. Ce diagnostic ne choisit pas le modèle.
- **Métriques par classe** : précision = part des hausses prédites qui sont réelles ;
  rappel = part des hausses réelles détectées. Le F1 combine précision et rappel.
  Le rapport inclut aussi les scores de baisse/stabilité et les effectifs réels.
- **Exploration sur train** : corrélations entre variables et équilibre des classes.
  Une corrélation ne démontre pas de causalité et n'entraîne pas automatiquement
  la suppression d'une variable.
- **Maîtrise de la complexité** : profondeur et taille minimale des feuilles
  limitées, régularisation L2 par défaut de la régression logistique. La validation
  croisée mesure les écarts de généralisation ; elle ne supprime pas le surapprentissage.

Le cours présente également GridSearchCV. Il reste ici une piste d'extension :
les paramètres et le choix initial sont conservés après consultation du test.
La démarche couvre chargement, préparation, variables, entraînement et évaluation.
L'intégration en production n'est pas réalisée : les résultats ne la justifient pas.

## Inspiration et adaptations

[Exemple GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/bitcoin-price-prediction-using-machine-learning-in-python/).
L'idée de classification directionnelle est reprise ; le code est organisé en
modules. Les adaptations portent sur le découpage chronologique, la séparation
validation/test, la standardisation sans fuite, l'exclusion de la dernière cible
inconnue et l'ajout d'une référence naïve. Aucun résultat annoncé dans l'article
n'est utilisé comme résultat de cette expérience.

Avant l'envoi au professeur, compléter le nom de l'auteur, lire les résultats
et pouvoir expliquer les choix. Joindre les sources, le CSV, les dépendances,
ce README et les résultats ; exclure le dossier `.venv`.
