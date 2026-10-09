"""Point d'entrée du projet UE8 : python main.py."""

import argparse
from pathlib import Path

from bitcoin_ml.data import load_data, prepare_features, split_data
from bitcoin_ml.diagnostics import evaluate_temporal_stability, summarize_stability
from bitcoin_ml.models import train_and_evaluate
from bitcoin_ml.plots import save_figures, save_diagnostics


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="UE8 — Prédire la hausse du Bitcoin à J+1")
    parser.add_argument("--data", type=Path, default=root / "data" / "bitcoin.csv")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prices = load_data(args.data)
    features, target = prepare_features(prices)
    splits = split_data(features, target)
    stability = evaluate_temporal_stability(*splits["train"])
    stability.to_csv(args.output / "validation_temporelle.csv", index=False)
    save_diagnostics(splits["train"], stability, args.output)
    scores, predictions, selected, report = train_and_evaluate(splits)
    scores.to_csv(args.output / "metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index_label="Date")
    save_figures(prices, splits, scores, predictions, selected, args.output)

    summary = (
        "UE8 — Classification du mouvement du Bitcoin à J+1\n\n"
        f"Données : {len(prices)} jours, du {prices.index.min():%d/%m/%Y} "
        f"au {prices.index.max():%d/%m/%Y}.\n"
        f"Observations exploitables : {len(features)}.\n"
    )
    for name, (x, y) in splits.items():
        summary += (
            f"{name} : {len(x)} observations, {x.index.min():%d/%m/%Y} "
            f"au {x.index.max():%d/%m/%Y}, hausses : {y.mean():.1%}.\n"
        )
    summary += "\n" + report
    summary += summarize_stability(stability)
    (args.output / "rapport.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nRésultats et graphiques : {args.output.resolve()}")


if __name__ == "__main__":
    main()
