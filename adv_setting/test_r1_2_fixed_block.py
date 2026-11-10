"""R1-2 fixed-block baseline regression tests. No Bodyfat file required."""
from __future__ import annotations

import math
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from loop_PTF_PBDOO_LR import fixed_block_params_convex
from loop_PTF_PDOCO_LR_strong import fixed_block_params_strong
from validate_r1_9_runs import expected_keys
import adv_setting_common as C


class FixedBlockComparisonTests(unittest.TestCase):
    def setUp(self):
        self.P = np.full((4, 4), 0.25)
        self.params = dict(T=10000, d=5, n=4, epsilon=0.5,
                           r=10.0, C_norm=10.0, P=self.P,
                           L=2.0, M=3.0)

    def test_convex_original_step_and_theorem_1_xi_balance(self):
        p = self.params
        tau, xi, eta, Ctau = fixed_block_params_convex(**p)
        self.assertEqual(tau, math.ceil(Ctau))
        self.assertAlmostEqual(eta, p['C_norm'] / (p['L'] * math.sqrt(tau * p['T'])))
        K = (p['M'] + p['L']*p['r']) / p['r']
        A = p['d']**2*p['M']**2 * (1+p['d']**2/(p['n']*tau*p['epsilon']**2))
        B = p['L']*p['d']**2*p['M']/p['epsilon']
        self.assertAlmostEqual(xi, min(max((eta*A/K)**(1/3), math.sqrt(eta*B/K)), 0.9*p['r']))

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

    def test_existing_three_unchanged_optional_fourth_in_validation(self):
        original = expected_keys('impact_epsilon_and_T', 'alpha0', [100], [1.0])
        added = expected_keys('impact_epsilon_and_T', 'alpha0', [100], [1.0], include_fixed_block=True)
        self.assertEqual(len(original), 3)
        self.assertEqual(len(added), 4)
        self.assertEqual(len(added - original), 1)
        self.assertIn('PTF-PBDOO-fixed-block-cor1', {spec.name for spec in [C.FIXED_BLOCK_ALPHA0]})
        self.assertIn('PTF-PBDOO-fixed-block-strong', {spec.name for spec in [C.FIXED_BLOCK_ALPHA1]})

    def test_floor_rejects_too_short_horizon(self):
        with self.assertRaisesRegex(ValueError, 'ceil\\(C_tau\\) <= T'):
            fixed_block_params_convex(**dict(self.params, T=1))
        with self.assertRaisesRegex(ValueError, 'ceil\\(C_tau\\) <= T'):
            fixed_block_params_strong(**dict(self.params, T=2, alpha=1.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
