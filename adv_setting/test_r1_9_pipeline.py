from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import adv_setting_common as C
from loop_PBDOO_LR import Algo7Config, run_algorithm7
from loop_PBDOO_LR_strong import algorithm1_dp_opbf
from loop_PTF_PBDOO_LR import PTF_PBDO2_Convex_Config, corollary1_params, run_ptf_pbd02_convex
from loop_PTF_PDOCO_LR_strong import (
    PTF_PBDO2_StronglyConvex_Config,
    corollary2_params,
    run_ptf_pbd02_strongly_convex,
)


class R19PipelineTests(unittest.TestCase):
    def test_deterministic_seed_is_stable_and_setting_specific(self) -> None:
        key_a = C.make_run_key(
            experiment="impact_epsilon_and_T", algorithm="BDOO", alpha_reg=0,
            T=1000, epsilon=None, num_nodes=9, topology="grid",
        )
        key_b = C.make_run_key(
            experiment="impact_epsilon_and_T", algorithm="BDOO", alpha_reg=0,
            T=3000, epsilon=None, num_nodes=9, topology="grid",
        )
        self.assertEqual(C.deterministic_seed(20261109, key_a, 0), C.deterministic_seed(20261109, key_a, 0))
        self.assertNotEqual(C.deterministic_seed(20261109, key_a, 0), C.deterministic_seed(20261109, key_a, 1))
        self.assertNotEqual(C.deterministic_seed(20261109, key_a, 0), C.deterministic_seed(20261109, key_b, 0))

    def test_run_experiment_checkpoints_and_resumes_idempotently(self) -> None:
        calls = {"count": 0}

        def flaky_runner(X, y, *, T, num_nodes, topology, rng):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("synthetic interruption")
            return float(rng.normal()), float(rng.normal()), float(abs(rng.normal()))

        alg = C.AlgorithmSpec("Dummy", flaky_runner, uses_epsilon=False, alpha_reg=0.0)
        settings = [
            {"T": 10, "num_nodes": 2, "topology": "grid", "epsilon": 0.1},
            {"T": 10, "num_nodes": 2, "topology": "grid", "epsilon": 1.0},
        ]
        X = np.zeros((3, 2), dtype=float)
        y = np.zeros(3, dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                C.run_experiment(
                    name="synthetic", settings=settings, algorithms=[alg], alpha_label="alpha0",
                    runs=3, base_seed=77, X=X, y=y, output_dir=out,
                )

            run_path = out / "synthetic_alpha0_runs.csv"
            partial = C.read_csv(run_path)
            self.assertEqual(len(partial), 1, "completed run must be checkpointed before the next run")
            self.assertEqual(partial[0]["run"], "0")

            # The second invocation fills only the missing run IDs.
            C.run_experiment(
                name="synthetic", settings=settings, algorithms=[alg], alpha_label="alpha0",
                runs=3, base_seed=77, X=X, y=y, output_dir=out,
            )
            rows_3 = C.read_csv(run_path)
            self.assertEqual(len(rows_3), 3, "epsilon-independent baseline must be run once")
            self.assertEqual(sorted(int(r["run"]) for r in rows_3), [0, 1, 2])
            self.assertEqual(len({r["seed"] for r in rows_3}), 3)

            # Repeating the same command must not append another batch.
            C.run_experiment(
                name="synthetic", settings=settings, algorithms=[alg], alpha_label="alpha0",
                runs=3, base_seed=77, X=X, y=y, output_dir=out,
            )
            self.assertEqual(len(C.read_csv(run_path)), 3)

            # Increasing the target from 3 to 5 adds only run IDs 3 and 4.
            C.run_experiment(
                name="synthetic", settings=settings, algorithms=[alg], alpha_label="alpha0",
                runs=5, base_seed=77, X=X, y=y, output_dir=out,
            )
            rows_5 = C.read_csv(run_path)
            self.assertEqual(len(rows_5), 5)
            self.assertEqual(sorted(int(r["run"]) for r in rows_5), [0, 1, 2, 3, 4])
            self.assertEqual(C.validate_run_rows(rows_5, expected_runs=5, base_seed=77), [])

    def test_convex_corollary_matches_manuscript_formula(self) -> None:
        T, d, n, eps, r, R, L, M = 10000, 5, 4, 0.5, 10.0, 10.0, 2.0, 3.0
        P = np.ones((n, n), dtype=float) / n
        tau, xi, eta, c_tau = corollary1_params(
            T=T, d=d, n=n, epsilon=eps, r=r, C_norm=R, P=P, L=L, M=M,
        )
        tau_expr = (d**4 * M**2 * (M + L*r)**2 * T / (n * L**4 * r**2 * R**2 * eps**2)) ** (1 / 3)
        expected_tau = int(np.ceil(max(c_tau, tau_expr)))
        xi_1 = (d**2 * M**2 * r**2 * R**2 / (T * (M + L*r)**2) *
                (1 + d**2 / (n * expected_tau * eps**2))) ** 0.25
        xi_2 = (L * d**2 * M * r**2 * R**2 / (eps * T * (M + L*r)**2)) ** (1 / 3)
        expected_xi = min(max(xi_1, xi_2), 0.9*r)
        expected_eta = R / np.sqrt(T * (
            d**2*M**2/expected_xi**2 * (1 + d**2/(n*expected_tau*eps**2))
            + expected_tau*L**2 + L*d**2*M/(eps*expected_xi)
        ))
        self.assertEqual(tau, expected_tau)
        self.assertAlmostEqual(xi, expected_xi, places=12)
        self.assertAlmostEqual(eta, expected_eta, places=12)

    def test_strongly_convex_corollary_matches_manuscript_formula(self) -> None:
        T, d, n, eps, alpha, r, R, L, M = 10000, 5, 4, 0.5, 1.0, 10.0, 10.0, 2.0, 3.0
        P = np.ones((n, n), dtype=float) / n
        tau, xi, c_tau = corollary2_params(
            T=T, d=d, n=n, epsilon=eps, alpha=alpha, r=r, C_norm=R, P=P, L=L, M=M,
        )
        lnT = np.log(T)
        tau_expr = (alpha**2*d**4*M**2*(M+L*r)**2*T**2 /
                    (n*r**2*eps**2*(L**2+alpha**2*R**2)**3*lnT**2)) ** 0.25
        expected_tau = int(np.ceil(max(tau_expr, c_tau)))
        xi_1 = (d**2*M**2*r*lnT/(alpha*(M+L*r)*T) *
                (1+d**2/(n*expected_tau*eps**2))) ** (1/3)
        xi_2 = (L*d**2*M*r*lnT/(alpha*eps*(M+L*r)*T)) ** 0.5
        expected_xi = min(max(xi_1, xi_2), 0.9*r)
        self.assertEqual(tau, expected_tau)
        self.assertAlmostEqual(xi, expected_xi, places=12)

    def test_internal_randomness_uses_supplied_rng(self) -> None:
        n, d, T = 2, 3, 4
        P = np.ones((n, n), dtype=float) / n
        zero_loss = lambda t, i, x: 0.0

        def convex_actions(seed: int) -> np.ndarray:
            cfg = PTF_PBDO2_Convex_Config(
                n=n, d=d, T=T, tau=2, eta=0.01, xi=0.1, epsilon=1.0, M=1.0,
                P=P, R=1.0, r=1.0, seed=0, rng=np.random.default_rng(seed), store_actions=True,
            )
            return run_ptf_pbd02_convex(zero_loss, cfg)["actions"]

        a1, a2, a1_repeat = convex_actions(11), convex_actions(12), convex_actions(11)
        self.assertFalse(np.allclose(a1, a2))
        np.testing.assert_allclose(a1, a1_repeat)

        def strong_actions(seed: int) -> np.ndarray:
            cfg = PTF_PBDO2_StronglyConvex_Config(
                n=n, d=d, T=T, tau=2, xi=0.1, epsilon=1.0, M=1.0, alpha=1.0,
                P=P, R=1.0, r=1.0, seed=0, rng=np.random.default_rng(seed), store_actions=True,
            )
            return run_ptf_pbd02_strongly_convex(zero_loss, cfg)["actions"]

        s1, s2, s1_repeat = strong_actions(21), strong_actions(22), strong_actions(21)
        self.assertFalse(np.allclose(s1, s2))
        np.testing.assert_allclose(s1, s1_repeat)

        def pbddo_actions(seed: int) -> np.ndarray:
            cfg = Algo7Config(
                n=n, d=d, T=T, delta=0.1, epsilon=1.0, zeta=1.0, C=1.0,
                seed=0, rng=np.random.default_rng(seed),
            )
            out = run_algorithm7(
                cost_oracle=lambda t, i, x: 0.0,
                A_seq=[np.eye(n) for _ in range(T)],
                cfg=cfg,
            )
            return out["x_hist"]

        p1, p2, p1_repeat = pbddo_actions(31), pbddo_actions(32), pbddo_actions(31)
        self.assertFalse(np.allclose(p1, p2))
        np.testing.assert_allclose(p1, p1_repeat)

        def strong_pbddo_actions(seed: int) -> np.ndarray:
            _, hist = algorithm1_dp_opbf(
                W=P,
                loss_fn=lambda i, t, x: 0.0,
                T=T,
                alpha=lambda t: 0.1,
                delta=0.1,
                sigma=lambda t: 0.1,
                x0=np.zeros((n, d)),
                radius=1.0,
                rng=np.random.default_rng(seed),
                return_history=True,
            )
            return hist

        q1, q2, q1_repeat = strong_pbddo_actions(41), strong_pbddo_actions(42), strong_pbddo_actions(41)
        self.assertFalse(np.allclose(q1, q2))
        np.testing.assert_allclose(q1, q1_repeat)

    def test_all_six_experiment_wrappers_use_distinct_run_rngs(self) -> None:
        data_rng = np.random.default_rng(2026)
        X = data_rng.normal(size=(32, 3))
        y = data_rng.normal(size=32)
        for algorithms in (C.ALGORITHMS_ALPHA0, C.ALGORITHMS_ALPHA1):
            for alg in algorithms:
                outputs = []
                for seed in (101, 102):
                    kwargs = {
                        "T": 120,
                        "num_nodes": 9,
                        "topology": "grid",
                        "rng": np.random.default_rng(seed),
                    }
                    if alg.uses_epsilon:
                        kwargs["epsilon"] = 1.0
                    with redirect_stdout(io.StringIO()):
                        outputs.append(np.asarray(alg.runner(X, y, **kwargs), dtype=float))
                self.assertFalse(
                    np.allclose(outputs[0], outputs[1]),
                    f"{alg.name} did not respond to distinct run RNGs",
                )

    def test_summary_uses_sample_standard_deviation(self) -> None:
        key = {
            "experiment": "synthetic", "algorithm": "A", "alpha_reg": 0.0,
            "T": 10, "epsilon": 1.0, "num_nodes": 2, "topology": "grid",
        }
        rows = []
        for run_id, value in enumerate(range(1, 11)):
            rows.append({
                **key, "run": run_id, "seed": run_id + 100,
                "avg_regret_per_round": value,
                "avg_loss_per_round": value + 1,
                "mse": value + 2,
            })
        summary = C.summarize_from_runs(rows)
        self.assertEqual(summary[0]["num_runs"], 10)
        self.assertAlmostEqual(summary[0]["avg_regret_mean"], np.mean(np.arange(1, 11)))
        self.assertAlmostEqual(summary[0]["avg_regret_std"], np.std(np.arange(1, 11), ddof=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
