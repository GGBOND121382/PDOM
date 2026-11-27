# Convex PBDOO learning diagnostics — trend analysis

## 1. Center model vs zero model (real learning gain)
- Candidate A (zeta=3e-05): center beats zero model in 0/18 settings; worst setting impact_topology eps=1.0 topo=complete T=100000: gain -39.11.
- Candidate B (zeta=0.0001): center beats zero model in 0/18 settings; worst setting impact_topology eps=1.0 topo=complete T=100000: gain -38.6.
- Candidate C (zeta=0.0003): center beats zero model in 0/18 settings; worst setting impact_topology eps=1.0 topo=complete T=100000: gain -36.74.

## 2. Center regret vs T
- Candidate A (eps=1, grid, n=9): improves with T (T=100000: 9.293, T=300000: 8.92, T=1000000: 8.765); strictly monotone: True.
- Candidate B (eps=1, grid, n=9): improves with T (T=100000: 8.702, T=300000: 8.051, T=1000000: 7.462); strictly monotone: True.
- Candidate C (eps=1, grid, n=9): improves with T (T=100000: 7.714, T=300000: 7.22, T=1000000: 5.769); strictly monotone: True.

## 3. Query-regret decomposition (learning vs exploration-loss reduction)
- Candidate A: T 1e5 -> 1e6, query change -2.358 = center change -0.5285 + exploration change -1.829.
- Candidate B: T 1e5 -> 1e6, query change -5.33 = center change -1.24 + exploration change -4.089.
- Candidate C: T 1e5 -> 1e6, query change -10.46 = center change -1.945 + exploration change -8.518.

## 4. Noise domination and projection saturation (strict privacy budget)
- Candidate A, eps=0.1: noise scale T=100000: 2.616e+04, T=300000: 3.142e+04, T=1000000: 3.84e+04; saturation ratio T=100000: 0, T=300000: 0, T=1000000: 0; center regret T=100000: 7.279, T=300000: 9.409, T=1000000: 15.86.
- Candidate B, eps=0.1: noise scale T=100000: 1.751e+04, T=300000: 2.103e+04, T=1000000: 2.57e+04; saturation ratio T=100000: 0.6561, T=300000: 0.169, T=1000000: 0.1912; center regret T=100000: 6.92, T=300000: 10.81, T=1000000: 26.69.
- Candidate C, eps=0.1: noise scale T=100000: 1.437e+04, T=300000: 1.458e+04, T=1000000: 1.782e+04; saturation ratio T=100000: 1, T=300000: 1, T=1000000: 1; center regret T=100000: 7.459, T=300000: 9.147, T=1000000: 17.2.

## 5. Trends vs theory
- See figures; monotonicity is not enforced and curves are reported as measured.

## Matched-seed comparison vs historical parameters and PB2D-FTRL
- Candidate A: better query regret than final-tuned-v1 in 15/54 matched rows, than trend-v1 in 2/54, than PB2D-FTRL in 0/54.
- Candidate B: better query regret than final-tuned-v1 in 7/54 matched rows, than trend-v1 in 3/54, than PB2D-FTRL in 0/54.
- Candidate C: better query regret than final-tuned-v1 in 5/54 matched rows, than trend-v1 in 3/54, than PB2D-FTRL in 0/54.

All values are measured results with matched seeds; no seed selection, no parameter selection, no data adjustment.