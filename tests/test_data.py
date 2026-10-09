"""Vérifications des risques de fuite temporelle, sans entraîner les modèles."""

import unittest
from pathlib import Path

import pandas as pd

from bitcoin_ml.data import load_data, prepare_features, split_data
from bitcoin_ml.diagnostics import temporal_folds


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = load_data(Path(__file__).resolve().parents[1] / "data" / "bitcoin.csv")

    def test_target_is_next_day_and_last_day_excluded(self):
        x, y = prepare_features(self.prices)
        self.assertNotIn(self.prices.index[-1], x.index)
        expected = (self.prices["Close"].shift(-1) > self.prices["Close"]).loc[y.index].astype(int)
        pd.testing.assert_series_equal(y, expected, check_names=False)

    def test_future_prices_do_not_change_past_features(self):
        cutoff = self.prices.index[len(self.prices) // 2]
        modified = self.prices.copy()
        modified.loc[modified.index > cutoff, ["Open", "High", "Low", "Close"]] *= 2
        original_x, _ = prepare_features(self.prices)
        modified_x, _ = prepare_features(modified)
        pd.testing.assert_frame_equal(original_x.loc[:cutoff], modified_x.loc[:cutoff])

    def test_labels_do_not_cross_split_boundaries(self):
        splits = split_data(*prepare_features(self.prices))
        for previous, following in [("train", "validation"), ("validation", "test")]:
            last_label_date = splits[previous][0].index.max() + pd.Timedelta(days=1)
            self.assertLess(last_label_date, splits[following][0].index.min())

    def test_cross_validation_stays_in_train_with_a_gap(self):
        x, y = prepare_features(self.prices)
        train_x, _ = split_data(x, y)["train"]
        folds = list(temporal_folds(train_x))
        self.assertEqual(len(folds), 5)
        for training, validation in folds:
            self.assertLess(train_x.index[training[-1]] + pd.Timedelta(days=1),
                            train_x.index[validation[0]])
            self.assertLessEqual(train_x.index[validation[-1]], train_x.index.max())


if __name__ == "__main__":
    unittest.main()
