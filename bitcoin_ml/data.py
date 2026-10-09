"""Lecture, contrôles et variables disponibles à la clôture du jour J."""

import numpy as np
import pandas as pd


def load_data(path):
    data = pd.read_csv(path)
    columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(["Date", *columns]) - set(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Dates absentes ou dupliquées dans le dataset.")
    data = data.set_index("Date").sort_index()[columns]
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Le dataset contient des valeurs manquantes ou infinies.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Les prix doivent être strictement positifs.")
    if (data["Volume"] < 0).any():
        raise ValueError("Les volumes doivent être positifs ou nuls.")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("Prix High incohérent.")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Prix Low incohérent.")
    # Refuser les trous : la prochaine ligne doit bien représenter J+1.
    if not data.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("La série doit contenir une observation par jour sans trou.")
    if len(data) < 200:
        raise ValueError("Au moins 200 jours sont nécessaires pour cette expérience.")
    return data


def prepare_features(prices):
    close = prices["Close"]
    returns = close.pct_change(fill_method=None)
    x = pd.DataFrame(index=prices.index)
    x["rendement_1j"] = returns
    x["rendement_7j"] = close.pct_change(7, fill_method=None)
    x["rendement_14j"] = close.pct_change(14, fill_method=None)
    x["amplitude"] = (prices["High"] - prices["Low"]) / close
    x["variation_intrajour"] = (close - prices["Open"]) / prices["Open"]
    x["ecart_moyenne_7j"] = close / close.rolling(7).mean() - 1
    x["ecart_moyenne_30j"] = close / close.rolling(30).mean() - 1
    x["volatilite_7j"] = returns.rolling(7).std()
    x["volume_log"] = np.log1p(prices["Volume"])
    # shift(-1) est réservé à la cible, jamais aux variables explicatives.
    # La dernière ligne n'a pas de lendemain connu : elle doit être exclue.
    next_close = close.shift(-1)
    valid = x.notna().all(axis=1) & next_close.notna()
    y = (next_close > close).astype(int).rename("hausse_demain")
    return x.loc[valid], y.loc[valid]


def split_data(x, y):
    # Ordre temporel conservé : 60 % apprentissage, 20 % validation, 20 % test.
    # Retirer une ligne avant chaque frontière évite que sa cible utilise
    # le premier jour de la période suivante.
    first, second = int(len(x) * 0.6), int(len(x) * 0.8)
    splits = {
        "train": (x.iloc[: first - 1], y.iloc[: first - 1]),
        "validation": (x.iloc[first : second - 1], y.iloc[first : second - 1]),
        "test": (x.iloc[second:], y.iloc[second:]),
    }
    for name, (_, labels) in splits.items():
        if labels.nunique() != 2:
            raise ValueError(f"Les deux classes doivent être présentes dans {name}.")
    return splits
