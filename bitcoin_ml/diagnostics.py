"""Diagnostic temporel des modèles fixes, exclusivement dans le train initial.

Ce diagnostic ajouté après la première évaluation ne modifie pas la sélection.
Il ne transforme pas le test déjà consulté en nouveau test indépendant.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from bitcoin_ml.models import build_models, measure


def temporal_folds(x):
    """Fenêtre d'apprentissage croissante, cinq validations et un jour de purge."""
    return TimeSeriesSplit(n_splits=5, gap=1).split(x)


def evaluate_temporal_stability(x, y):
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(temporal_folds(x), start=1):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[valid_idx].nunique() < 2:
            raise ValueError(f"Le pli {fold} ne contient pas les deux classes.")
        for name, estimator in build_models().items():
            # Le Pipeline réapprend son scaler dans CHAQUE pli.
            model = clone(estimator).fit(x.iloc[train_idx], y.iloc[train_idx])
            rows.append({
                "pli": fold, "modele": name,
                "debut_train": x.index[train_idx[0]],
                "fin_train": x.index[train_idx[-1]],
                "debut_validation": x.index[valid_idx[0]],
                "fin_validation": x.index[valid_idx[-1]],
                "n_train": len(train_idx), "n_validation": len(valid_idx),
                "auc_train": measure(model, x.iloc[train_idx], y.iloc[train_idx])["roc_auc"],
                **measure(model, x.iloc[valid_idx], y.iloc[valid_idx]),
            })
    return pd.DataFrame(rows)


def summarize_stability(scores):
    summary = scores.groupby("modele")["roc_auc"].agg(["mean", "std", "min", "max"])
    return (
        "\nDiagnostic : cinq validations temporelles dans le train initial\n"
        + summary.round(4).to_string()
        + "\nLa dispersion décrit les différences entre périodes ; ce n'est pas "
        "un intervalle de confiance.\n"
        + "Ce diagnostic ne modifie ni les hyperparamètres ni le modèle choisi. "
        "Le test a déjà été consulté lors de la première version du projet.\n"
    )
