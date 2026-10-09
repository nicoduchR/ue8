"""Comparer les modèles sur validation et évaluer le choix sur un test intact."""

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def measure(model, x, y):
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "f1": f1_score(y, prediction, zero_division=0),
        "precision": precision_score(y, prediction, zero_division=0),
        "rappel": recall_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }


def build_models():
    """Configurations fixées ; chaque appel crée des modèles non entraînés."""
    return {
        "Reference majoritaire": DummyClassifier(strategy="prior"),
        "Regression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Foret aleatoire": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    }


def train_and_evaluate(splits):
    models = build_models()
    x_train, y_train = splits["train"]
    rows = []
    for name, model in models.items():
        # Le scaler de la régression est ajusté exclusivement sur l'apprentissage.
        model.fit(x_train, y_train)
        for period in ("train", "validation"):
            rows.append({"modele": name, "periode": period,
                         **measure(model, *splits[period])})
    validation = [r for r in rows if r["periode"] == "validation"]
    selected = max(validation, key=lambda row: row["roc_auc"])["modele"]

    # Réentraînement sur les deux périodes déjà observées. Les lignes purgées
    # restent exclues ; le jeu de test n'intervient jamais dans le choix.
    x_fit = pd.concat([splits["train"][0], splits["validation"][0]])
    y_fit = pd.concat([splits["train"][1], splits["validation"][1]])
    x_test, y_test = splits["test"]
    predictions = pd.DataFrame({"reel": y_test})
    for name in dict.fromkeys(["Reference majoritaire", selected]):
        model = clone(models[name]).fit(x_fit, y_fit)
        rows.append({"modele": name, "periode": "test", **measure(model, x_test, y_test)})
        if name == selected:
            predictions["prediction"] = model.predict(x_test)
            predictions["probabilite_hausse"] = model.predict_proba(x_test)[:, 1]
    scores = pd.DataFrame(rows)
    test = scores[scores["periode"] == "test"].set_index("modele")
    auc = test.loc[selected, "roc_auc"]
    accuracy = test.loc[selected, "accuracy"]
    baseline = test.loc["Reference majoritaire", "accuracy"]
    report = (
        f"Modèle choisi sur l'AUC de validation : {selected}.\n\n"
        + scores.round(4).to_string(index=False)
        + f"\n\nSur le test : AUC = {auc:.3f} ; exactitude = {accuracy:.1%}.\n"
        + f"Exactitude de la référence majoritaire : {baseline:.1%}.\n"
        + "\nMétriques par classe sur le test (support = effectif réel) :\n"
        + classification_report(
            predictions["reel"], predictions["prediction"], labels=[0, 1],
            target_names=["Baisse/stable", "Hausse"], digits=3, zero_division=0,
        )
        + "Une AUC de 0,5 correspond à un classement sans pouvoir discriminant.\n"
        + "Ces scores décrivent une seule période historique ; ils ne prouvent pas "
        "une capacité stable à prévoir le marché.\n"
        + "Le modèle prédit une direction, pas un prix ni la rentabilité d'une stratégie.\n"
        + "Limites : aucun coût de transaction, aucune information macroéconomique, "
        "pas d'intervalle de confiance ; une seule période de test final.\n"
    )
    return scores, predictions, selected, report
