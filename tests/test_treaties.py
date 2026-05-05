from reinsure_pricing.treaties import ExcessOfLoss, StopLoss, ReinstatementProvision
import numpy as np
import pytest

class TestAAL:

    def test_aal_caps_annual_ceded(self):
        # with AAL of 6M, annual ceded should never exceed 6M
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000,
                              aggregate_limit=6_000_000)
        losses = np.array([3_000_000, 3_000_000, 3_000_000])
        result = treaty.apply_detailed(losses)
        assert result.annual_ceded <= 6_000_000

    def test_aal_partial_cession(self):
        # claim straddling AAL should be partially ceded
        treaty = ExcessOfLoss(retention=0, limit=5_000_000,
                              aggregate_limit=7_000_000)
        losses = np.array([5_000_000, 5_000_000, 5_000_000])
        result = treaty.apply_detailed(losses)
        # first two claims cede 5M each = 10M > AAL 7M
        # third claim should cede 0, second should be partial
        assert abs(result.annual_ceded - 7_000_000) < 1

    def test_aal_exhausted_flag(self):
        treaty = ExcessOfLoss(retention=0, limit=5_000_000,
                              aggregate_limit=4_000_000)
        losses = np.array([5_000_000, 5_000_000])
        result = treaty.apply_detailed(losses)
        assert result.aal_exhausted is True

    def test_no_aal_not_exhausted(self):
        treaty = ExcessOfLoss(retention=1_000_000, limit=5_000_000)
        losses = np.array([2_000_000])
        result = treaty.apply_detailed(losses)
        assert result.aal_exhausted is False

    def test_aal_per_claim_sum_equals_annual(self):
        treaty = ExcessOfLoss(retention=500_000, limit=3_000_000,
                              aggregate_limit=5_000_000)
        losses = np.array([2_000_000, 2_000_000, 2_000_000, 2_000_000])
        result = treaty.apply_detailed(losses)
        assert abs(result.ceded_per_claim.sum() - result.annual_ceded) < 1


class TestAAD:

    def test_aad_absorbs_small_years(self):
        # if annual ceded < AAD, reinsurer pays nothing
        treaty = ExcessOfLoss(retention=500_000, limit=5_000_000,
                              aggregate_deductible=3_000_000)
        losses = np.array([1_000_000, 1_000_000])
        result = treaty.apply_detailed(losses)
        assert result.annual_ceded == 0.0

    def test_aad_excess_paid(self):
        # reinsurer pays only excess over AAD
        treaty = ExcessOfLoss(retention=0, limit=10_000_000,
                              aggregate_deductible=2_000_000)
        losses = np.array([5_000_000])
        result = treaty.apply_detailed(losses)
        assert abs(result.annual_ceded - 3_000_000) < 1

    def test_aad_remaining(self):
        treaty = ExcessOfLoss(retention=0, limit=10_000_000,
                              aggregate_deductible=5_000_000)
        losses = np.array([2_000_000])
        result = treaty.apply_detailed(losses)
        assert result.aad_remaining == 3_000_000


class TestReinstatementProvision:

    def test_no_reinstatements_no_premium(self):
        rp = ReinstatementProvision(n_free=1, n_paid=1,
                                    original_premium=500_000)
        assert rp.reinstatement_premium(0) == 0.0

    def test_free_reinstatement_no_premium(self):
        rp = ReinstatementProvision(n_free=1, n_paid=1,
                                    original_premium=500_000)
        # first exhaustion is free
        assert rp.reinstatement_premium(1) == 0.0

    def test_paid_reinstatement_charges_premium(self):
        rp = ReinstatementProvision(n_free=1, n_paid=1,
                                    original_premium=500_000)
        # second exhaustion triggers paid reinstatement
        assert rp.reinstatement_premium(2) > 0.0

    def test_max_reinstatements_cap(self):
        rp = ReinstatementProvision(n_free=1, n_paid=1,
                                    original_premium=500_000)
        # 3, 5, 10 exhaustions should all give the same premium
        # because only 1 paid reinstatement is allowed regardless
        p3  = rp.reinstatement_premium(3)
        p5  = rp.reinstatement_premium(5)
        p10 = rp.reinstatement_premium(10)
        assert p3 == p5 == p10

    def test_net_cost_less_than_ceded(self):
        rp = ReinstatementProvision(n_free=1, n_paid=1,
                                    original_premium=500_000)
        net = rp.net_cost(ceded_loss=2_000_000, n_reinstatements=2)
        assert net < 2_000_000

    def test_rejects_negative_premium(self):
        with pytest.raises(ValueError):
            ReinstatementProvision(original_premium=-1)