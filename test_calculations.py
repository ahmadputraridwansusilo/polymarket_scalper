import unittest

from calculations import compute_phase1_lock_plan


class Phase1LockPlanTests(unittest.TestCase):
    def test_profitable_lock_budget_stays_strict_during_phase1(self) -> None:
        plan = compute_phase1_lock_plan(
            base_amount=2.55,
            entry_price=0.51,
            hedge_price=0.49,
            target_minimum_profit=0.1275,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertAlmostEqual(plan.max_hedge_budget, 2.3225)
        self.assertAlmostEqual(plan.hedge_cost, 2.45)
        self.assertLess(plan.max_hedge_budget, plan.hedge_cost)

    def test_risk_off_budget_can_finish_full_hedge_near_phase2(self) -> None:
        plan = compute_phase1_lock_plan(
            base_amount=2.55,
            entry_price=0.51,
            entry_shares=5.0,
            hedge_price=0.495,
            current_hedge_shares=0.0,
            current_hedge_cost=0.0,
            target_minimum_profit=0.1275,
            max_acceptable_loss=0.1275,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertAlmostEqual(plan.max_hedge_budget, 2.5775)
        self.assertAlmostEqual(plan.remaining_hedge_budget, 2.5775)
        self.assertAlmostEqual(plan.hedge_cost, 2.475)
        self.assertAlmostEqual(plan.guaranteed_profit, -0.025)

    def test_remaining_budget_respects_existing_partial_hedge(self) -> None:
        plan = compute_phase1_lock_plan(
            base_amount=2.55,
            entry_price=0.51,
            entry_shares=5.0,
            hedge_price=0.48,
            current_hedge_shares=2.0,
            current_hedge_cost=0.94,
            target_minimum_profit=0.1275,
            max_acceptable_loss=0.1275,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertAlmostEqual(plan.hedge_shares_needed, 3.0)
        self.assertAlmostEqual(plan.hedge_cost, 1.44)
        self.assertAlmostEqual(plan.remaining_hedge_budget, 1.6375)

    def test_stoploss_budget_matches_seventy_percent_loss_threshold(self) -> None:
        stoploss_loss = 2.55 * 0.70
        safe_plan = compute_phase1_lock_plan(
            base_amount=2.55,
            entry_price=0.51,
            entry_shares=5.0,
            hedge_price=0.84,
            target_minimum_profit=0.1275,
            max_acceptable_loss=stoploss_loss,
        )
        breached_plan = compute_phase1_lock_plan(
            base_amount=2.55,
            entry_price=0.51,
            entry_shares=5.0,
            hedge_price=0.85,
            target_minimum_profit=0.1275,
            max_acceptable_loss=stoploss_loss,
        )

        self.assertIsNotNone(safe_plan)
        self.assertIsNotNone(breached_plan)
        assert safe_plan is not None and breached_plan is not None
        self.assertAlmostEqual(safe_plan.max_hedge_budget, 4.235)
        self.assertLessEqual(safe_plan.hedge_cost, safe_plan.max_hedge_budget)
        self.assertGreater(breached_plan.hedge_cost, breached_plan.max_hedge_budget)


if __name__ == "__main__":
    unittest.main()
