import numpy as np
import pytest
from reinsure_pricing.treaties import ExcessOfLoss, StopLoss


class TestExcessOfLoss:

    def test_no_loss(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([0.0])) == 0.0

    def test_loss_below_retention(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([500_000.0])) == 0.0

    def test_loss_exactly_at_retention(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([1_000_000.0])) == 0.0

    def test_loss_within_layer(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([3_000_000.0])) == 2_000_000.0

    def test_loss_exhausts_limit(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([6_000_000.0])) == 5_000_000.0

    def test_loss_above_exhaustion(self):
        # ceded loss should be capped at limit
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        assert treaty.apply(np.array([20_000_000.0])) == 5_000_000.0

    def test_rejects_negative_retention(self):
        with pytest.raises(ValueError):
            ExcessOfLoss(retention=-1, limit=5_000_000)


class TestStopLoss:

    def test_no_aggregate_loss(self):
        treaty = StopLoss(attachment=10_000_000, cap=5_000_000)
        assert treaty.apply(0.0) == 0.0

    def test_below_attachment(self):
        treaty = StopLoss(attachment=10_000_000, cap=5_000_000)
        assert treaty.apply(8_000_000.0) == 0.0

    def test_within_layer(self):
        treaty = StopLoss(attachment=10_000_000, cap=5_000_000)
        assert treaty.apply(12_000_000.0) == 2_000_000.0

    def test_exhausts_cap(self):
        treaty = StopLoss(attachment=10_000_000, cap=5_000_000)
        assert treaty.apply(20_000_000.0) == 5_000_000.0