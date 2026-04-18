import unittest

from calculations import (
    compute_loss_ratio,
    compute_mark_to_market_value,
    compute_profit_ratio,
    compute_scaled_entry_amount,
    compute_shares_to_sell,
    evaluate_hedge_plan,
    find_minimum_hedge_plan,
)


class EntrySizingTests(unittest.TestCase):
    def test_scaled_entry_amount_grows_linearly_from_reference_balance(self) -> None:
        self.assertAlmostEqual(
            compute_scaled_entry_amount(
                base_entry_amount=3.0,
                current_total_equity=49.0,
                reference_total_equity=49.0,
            ),
            3.0,
        )
        self.assertAlmostEqual(
            compute_scaled_entry_amount(
                base_entry_amount=3.0,
                current_total_equity=98.0,
                reference_total_equity=49.0,
            ),
            6.0,
        )


class HedgeCalculationTests(unittest.TestCase):
    def test_evaluate_hedge_plan_classifies_full_two_way_lock(self) -> None:
        plan = evaluate_hedge_plan(
            entry_cost=5.0,
            entry_shares=6.17,
            hedge_price=0.07,
            hedge_amount=0.93,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.classification, "LOCK_BOTH_PROFIT")
        self.assertGreater(plan.entry_net, 0.0)
        self.assertGreater(plan.hedge_net, 0.0)

    def test_binary_search_finds_minimum_break_even_hedge(self) -> None:
        plan = find_minimum_hedge_plan(
            entry_cost=5.0,
            entry_shares=6.17,
            hedge_price=0.07,
            max_hedge_amount=3.0,
            minimum_hedge_amount=0.01,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(plan.entry_net, -1e-6)
        self.assertGreaterEqual(plan.hedge_net, -1e-6)
        self.assertAlmostEqual(plan.hedge_amount, 0.38)

    def test_binary_search_returns_none_when_hedge_can_never_lock(self) -> None:
        plan = find_minimum_hedge_plan(
            entry_cost=5.0,
            entry_shares=5.55,
            hedge_price=0.60,
            max_hedge_amount=10.0,
            minimum_hedge_amount=1.0,
        )

        self.assertIsNone(plan)


class FloatingMetricTests(unittest.TestCase):
    def test_profit_and_loss_ratio_use_mark_to_market_value(self) -> None:
        current_value = compute_mark_to_market_value(10.0, 0.80)
        self.assertAlmostEqual(current_value, 8.0)
        self.assertAlmostEqual(compute_profit_ratio(5.0, current_value), 0.60)
        self.assertAlmostEqual(compute_loss_ratio(10.0, current_value), 0.20)

    def test_partial_cut_loss_sell_uses_requested_percentage(self) -> None:
        self.assertAlmostEqual(compute_shares_to_sell(25.0, 80.0), 20.0)


if __name__ == "__main__":
    unittest.main()
