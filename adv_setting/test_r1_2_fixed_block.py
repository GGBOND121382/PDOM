"""R1-2 fixed-block baseline regression tests. No Bodyfat file required."""
from __future__ import annotations

import math
from tempfile import TemporaryDirectory
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from loop_PTF_PBDOO_LR import (
    PTF_PBDO2_Convex_Config,
    corollary1_params,
    fixed_block_optimized_eta_params_convex,
    fixed_block_params_convex,
    run_ptf_pbd02_convex,
)
from loop_PTF_PDOCO_LR_strong import fixed_block_params_strong
from validate_r1_9_runs import expected_keys
import adv_setting_common as C


class FixedBlockComparisonTests(unittest.TestCase):
    def setUp(self):
        self.P = np.full((4, 4), 0.25)
        self.params = dict(T=10000, d=5, n=4, epsilon=0.5,
                           r=10.0, C_norm=10.0, P=self.P,
                           L=2.0, M=3.0)

    def test_historical_convex_fixed_eta_formula_is_retained_but_not_active(self):
        p = self.params
        tau, xi, eta, Ctau = fixed_block_params_convex(**p)
        self.assertEqual(tau, math.ceil(Ctau))
        self.assertAlmostEqual(eta, p['C_norm'] / (p['L'] * math.sqrt(tau * p['T'])))
        K = (p['M'] + p['L']*p['r']) / p['r']
        A = p['d']**2*p['M']**2 * (1+p['d']**2/(p['n']*tau*p['epsilon']**2))
        B = p['L']*p['d']**2*p['M']/p['epsilon']
        self.assertAlmostEqual(xi, min(max((eta*A/K)**(1/3), math.sqrt(eta*B/K)), 0.9*p['r']))
        self.assertNotIn(C.LEGACY_FIXED_BLOCK_ALPHA0, C.ALGORITHMS_ALPHA0)
        self.assertNotIn(C.LEGACY_FIXED_BLOCK_ALPHA0.name, C.CONVEX_FTRL_ALGORITHMS)

    def test_strong_theorem_2_xi_at_fixed_tau(self):
        p = dict(self.params, alpha=1.0)
        tau, xi, Ctau = fixed_block_params_strong(**p)
        self.assertEqual(tau, math.ceil(Ctau))
        B = (p['d']**2*p['M']**2*p['r']*math.log(p['T'])
             /(p['alpha']*(p['M']+p['L']*p['r'])*p['T'])
             * (1 + p['d']**2/(p['n']*tau*p['epsilon']**2)))**(1/3)
        P = math.sqrt(p['L']*p['d']**2*p['M']*p['r']*math.log(p['T'])
                      /(p['alpha']*p['epsilon']*(p['M']+p['L']*p['r'])*p['T']))
        self.assertAlmostEqual(xi, min(max(B, P), 0.9*p['r']))

    def test_convex_optimized_eta_rebalances_at_the_fixed_tau(self):
        for epsilon in (0.5, 1e-5):
            p = dict(self.params, epsilon=epsilon)
            tau, xi, eta, Ctau = fixed_block_optimized_eta_params_convex(**p)
            self.assertEqual(tau, math.ceil(Ctau))
            old_tau, _, old_eta, old_Ctau = fixed_block_params_convex(**p)
            self.assertEqual((tau, Ctau), (old_tau, old_Ctau))
            K = p["M"] / p["r"] + p["L"]
            A = p["d"]**2 * p["M"]**2 * (1 + p["d"]**2 / (p["n"] * tau * epsilon**2))
            B = p["L"] * p["d"]**2 * p["M"] / epsilon
            F = p["C_norm"]**2 / (p["T"] * K**2)
            uncapped = max((F * A)**0.25, (F * B)**(1/3))
            self.assertEqual(uncapped > 0.9 * p["r"], epsilon == 1e-5)
            self.assertAlmostEqual(xi, min(uncapped, 0.9 * p["r"]))
            self.assertAlmostEqual(eta, p["C_norm"] / math.sqrt(
                p["T"] * (A / xi**2 + B / xi + tau * p["L"]**2)
            ))
            self.assertLess(eta, old_eta)

    def test_convex_schedules_agree_when_corollary_uses_the_fixed_tau(self):
        p = dict(self.params, epsilon=1e5)
        self.assertEqual(corollary1_params(**p), fixed_block_optimized_eta_params_convex(**p))

    def test_algorithm1_mixing_and_ftrl_projection(self):
        P = 0.5 * np.eye(4) + 0.5 * self.P
        cfg = PTF_PBDO2_Convex_Config(n=4, d=2, T=8, tau=2, eta=0.01,
                                      xi=0.2, epsilon=1., M=1., P=P, R=10.)
        res = run_ptf_pbd02_convex(lambda t, i, x: 1., cfg)
        self.assertAlmostEqual(res["sigma2"], 0.5)
        self.assertAlmostEqual(res["theta"], 1 / (1 + math.sqrt(1 - 0.5**2)))
        self.assertGreater(np.linalg.norm(res["z_batches"]), 0)
        from optimization_utils.adv_setting_utils import l2_ball_projection
        expected = l2_ball_projection(-cfg.eta * res["z_batches"] / 2, 9.8)
        np.testing.assert_allclose(res["x_batches"], expected)

    def test_only_unaffected_runs_are_reused(self):
        with TemporaryDirectory() as root:
            source, target = Path(root) / "old", Path(root) / "new"
            experiment = "impact_epsilon_and_T"
            t, n, eps, topo = 1000, 9, 1.0, "grid"
            ftrl = next(a for a in C.ALGORITHMS_ALPHA0 if a.name == "PTF-PBDOO-cor1")
            source_rows = [
                dict(experiment=experiment, algorithm=alg, alpha_reg=0, run=0,
                     seed=C.deterministic_seed(9, C.make_run_key(
                         experiment=experiment, algorithm=alg, alpha_reg=0, T=t,
                         epsilon=eps, num_nodes=n, topology=topo), 0),
                     T=t, epsilon=eps, num_nodes=n, topology=topo,
                     avg_regret_per_round=10., avg_loss_per_round=11., mse=12.)
                for alg in ("unaffected", ftrl.name)
            ]
            filename = f"{experiment}_alpha0_runs.csv"
            C.write_csv(source / filename, source_rows)
            def skip(*args, **kwargs):
                self.fail("unaffected old row was not reused")
            def run(*args, **kwargs):
                return 1., 2., 3.
            specs = [C.AlgorithmSpec("unaffected", skip, True, 0.),
                     C.AlgorithmSpec(ftrl.name, run, True, 0.)]
            args = dict(name=experiment, settings=[dict(T=t, epsilon=eps, num_nodes=n, topology=topo)],
                        algorithms=specs, alpha_label="alpha0", runs=1,
                        base_seed=9, X=np.zeros((1, 1)), y=np.zeros(1))
            with self.assertRaisesRegex(ValueError, "Old convex FTRL"):
                C.run_experiment(**args, output_dir=source)
            C.run_experiment(**args, output_dir=target, reuse_unaffected_from=source)
            a, b = C.read_csv(target / filename)
            self.assertEqual(float(a["avg_regret_per_round"]), 10.)
            self.assertEqual(float(b["avg_regret_per_round"]), 1.)
            self.assertEqual(b["implementation"], C.CONVEX_FTRL_REVISION)
            self.assertEqual(len(C.read_csv(source / filename)), 2)
    def test_one_vanilla_block_baseline_is_expected_in_each_convexity_case(self):
        for alpha, baseline in (
            ("alpha0", C.VANILLA_BLOCK_ALPHA0),
            ("alpha1", C.VANILLA_BLOCK_ALPHA1),
        ):
            original = expected_keys("impact_epsilon_and_T", alpha, [100], [1.0])
            added = expected_keys(
                "impact_epsilon_and_T", alpha, [100], [1.0],
                include_vanilla_block=True,
            )
            self.assertEqual(len(original), 3)
            self.assertEqual(len(added), 4)
            self.assertEqual(len(added - original), 1)
            self.assertIn(baseline.name, {key[1] for key in added})
            self.assertNotIn(C.LEGACY_FIXED_BLOCK_ALPHA0.name, {key[1] for key in added})

    def test_full_comparison_adds_exactly_one_vanilla_baseline_per_private_point(self):
        for experiment in ("impact_epsilon_and_T", "impact_topology", "impact_num_learners"):
            for alpha, proposed, baseline in (
                ("alpha0", "PTF-PBDOO-cor1", C.VANILLA_BLOCK_ALPHA0.name),
                ("alpha1", "PTF-PBDOO-strong", C.VANILLA_BLOCK_ALPHA1.name),
            ):
                old = expected_keys(experiment, alpha, [1000, 3000], [0.1, 1, 10])
                new = expected_keys(
                    experiment, alpha, [1000, 3000], [0.1, 1, 10],
                    include_vanilla_block=True,
                )
                self.assertTrue(old <= new)
                expected = {
                    (key[0], baseline, *key[2:])
                    for key in old if key[1] == proposed
                }
                self.assertEqual(new - old, expected)

    def test_reuses_corrected_convex_ftrl_rows_and_ignores_obsolete_baseline(self):
        with TemporaryDirectory() as root:
            source, target = Path(root) / "old", Path(root) / "new"
            experiment, t, n, eps, topo = "impact_epsilon_and_T", 1000, 9, 1.0, "grid"
            active_names = (
                "unaffected",
                "PTF-PBDOO-cor1",
                C.VANILLA_BLOCK_ALPHA0.name,
            )
            rows = []
            for algorithm in active_names + (C.LEGACY_FIXED_BLOCK_ALPHA0.name,):
                key = C.make_run_key(
                    experiment=experiment, algorithm=algorithm, alpha_reg=0.0,
                    T=t, epsilon=eps, num_nodes=n, topology=topo,
                )
                row = dict(
                    experiment=experiment, algorithm=algorithm, alpha_reg=0.0,
                    run=0, seed=C.deterministic_seed(9, key, 0), T=t, epsilon=eps,
                    num_nodes=n, topology=topo, avg_regret_per_round=10.0,
                    avg_loss_per_round=11.0, mse=12.0,
                )
                if algorithm in C.CONVEX_FTRL_ALGORITHMS:
                    row["implementation"] = C.CONVEX_FTRL_REVISION
                rows.append(row)
            C.write_csv(source / f"{experiment}_alpha0_runs.csv", rows)

            def fail_if_run(*args, **kwargs):
                self.fail("a matching corrected result should have been reused")

            algorithms = [
                C.AlgorithmSpec("unaffected", fail_if_run, True, 0.0),
                C.AlgorithmSpec("PTF-PBDOO-cor1", fail_if_run, True, 0.0),
                C.AlgorithmSpec(C.VANILLA_BLOCK_ALPHA0.name, fail_if_run, True, 0.0),
            ]
            C.run_experiment(
                name=experiment,
                settings=[dict(T=t, num_nodes=n, topology=topo, epsilon=eps)],
                algorithms=algorithms, alpha_label="alpha0", runs=1, base_seed=9,
                X=np.zeros((1, 1)), y=np.zeros(1), output_dir=target,
                reuse_unaffected_from=source,
            )
            out = C.read_csv(target / f"{experiment}_alpha0_runs.csv")
            self.assertEqual({r["algorithm"] for r in out}, set(active_names))
            self.assertEqual(len(out), 3)
            self.assertEqual(len(C.read_csv(source / f"{experiment}_alpha0_runs.csv")), 4)

    def test_floor_rejects_too_short_horizon(self):
        with self.assertRaisesRegex(ValueError, 'ceil\\(C_tau\\) <= T'):
            fixed_block_params_convex(**dict(self.params, T=1))
        with self.assertRaisesRegex(ValueError, 'ceil\\(C_tau\\) <= T'):
            fixed_block_optimized_eta_params_convex(**dict(self.params, T=1))
        with self.assertRaisesRegex(ValueError, 'ceil\\(C_tau\\) <= T'):
            fixed_block_params_strong(**dict(self.params, T=2, alpha=1.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
