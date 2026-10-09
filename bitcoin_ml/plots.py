"""Graphiques enregistrés en PNG, sans fenêtre interactive bloquante."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_figures(prices, splits, scores, predictions, selected, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prices.index, prices["Close"], color="#245b8a", linewidth=1)
    for period, color in [("validation", "#efb366"), ("test", "#7cc7a3")]:
        dates = splits[period][0].index
        ax.axvspan(dates.min(), dates.max(), color=color, alpha=0.35, label=period)
    ax.set(title="Bitcoin : cours de clôture et découpage chronologique", ylabel="Prix (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "01_prix_et_periodes.png", dpi=160)
    plt.close(fig)
    save_model_figures(scores, predictions, selected, output)


def save_diagnostics(train, stability, output):
    """Exploration limitée au train et stabilité des modèles sur cinq périodes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    x, y = train
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    corr = x.corr()
    heatmap = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    axes[0].set_yticks(range(len(corr)), corr.columns)
    axes[0].grid(False)
    axes[0].set_title("Corrélations entre variables — train")
    fig.colorbar(heatmap, ax=axes[0], shrink=0.75)
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    axes[1].bar(["Baisse/stable", "Hausse"], counts, color=["#245b8a", "#e4a04b"])
    for position, count in enumerate(counts):
        axes[1].text(position, count, f"{count}\n({count / len(y):.1%})", ha="center", va="bottom")
    axes[1].set(title="Répartition des classes — train", ylabel="Observations", ylim=(0, counts.max() * 1.2))
    fig.tight_layout()
    fig.savefig(output / "04_exploration_train.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, group in stability.groupby("modele", sort=False):
        ax.plot(group["pli"], group["roc_auc"], marker="o", label=name)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(title="Stabilité sur cinq validations chronologiques dans le train",
           xlabel="Pli (du plus ancien au plus récent)", ylabel="ROC-AUC de validation",
           xticks=[1, 2, 3, 4, 5], ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "05_stabilite_temporelle.png", dpi=160)
    plt.close(fig)


def save_model_figures(scores, predictions, selected, output):
    fig, ax = plt.subplots(figsize=(9, 5))
    table = scores[scores["periode"] != "test"].pivot(index="modele", columns="periode", values="roc_auc")
    table.plot.bar(ax=ax, rot=0, color=["#245b8a", "#e4a04b"])
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(title="AUC : apprentissage et validation", ylabel="ROC-AUC", xlabel="", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output / "02_comparaison_modeles.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        predictions["reel"], predictions["prediction"], labels=[0, 1],
        display_labels=["Baisse/stable", "Hausse"], ax=axes[0], colorbar=False, cmap="Blues",
    )
    axes[0].set(title="Matrice de confusion — test", xlabel="Classe prédite", ylabel="Classe réelle")
    axes[0].grid(False)
    RocCurveDisplay.from_predictions(predictions["reel"], predictions["probabilite_hausse"], ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Courbe ROC — test", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    fig.suptitle(selected)
    fig.tight_layout()
    fig.savefig(output / "03_evaluation_test.png", dpi=160)
    plt.close(fig)
