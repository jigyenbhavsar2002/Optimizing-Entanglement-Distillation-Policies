# Optimizing Entanglement Distillation Policies via Markov Decision Process Formulation

Code accompanying the paper *"Optimizing Entanglement Distillation Policies via
Markov Decision Process Formulation"* (add venue / arXiv link / DOI once available).

This repository formulates the problem of choosing when to generate vs.
distill entangled pairs across `m` quantum memories as a finite-state,
undiscounted (γ = 1) Markov decision process, solves it exactly via value
iteration, and compares the resulting optimal policy against three standard
heuristics from the entanglement-purification literature: **pumping**,
**nested (recurrence) purification**, and **greedy** (minimum
fidelity-difference matching).

## Repository structure

```
src/
  mdp.py            core MDP construction (distillation map, state/action
                     space, transition model) -- shared by everything below
  optimal_policy.py  value iteration, γ = 1  (Sec. III / Algorithm 1)
  baselines.py        pumping / nested / greedy policy builders (Sec. IV)
  policy_eval.py       evaluate a fixed policy two ways: (1) absorption
                        probability + exact linear solve, (2) value
                        iteration with the action fixed -- both verified to
                        agree to 5+ decimal places
scripts/
  reproduce_figures.py  sweeps (p, f0), builds the advantage heatmap
                          (T_min(baselines) - T_opt) / T_min(baselines)
notebooks/
  walkthrough.ipynb     narrated example: build an MDP, solve it, compare
                          to baselines, inspect the optimal policy table
tests/
  test_policies.py      sanity checks, including two paper claims verified
                          computationally: the m=2 optimal policy reduces
                          exactly to pumping (independent of p), and the
                          state space is finite iff f_T < f_infty(f0)
data/results/           cached grid outputs (JSON) so figures can be
                          regenerated without rerunning every sweep
figures/                 output directory for regenerated plots
docs/                    supplementary derivations (twirling / Werner-state
                          preservation, finite-state-space proof)
```

## Installation

```bash
git clone https://github.com/<your-username>/entanglement-distillation-mdp.git
cd entanglement-distillation-mdp
pip install -r requirements.txt
```

Requires Python ≥ 3.9, `numpy`, `matplotlib`. `pytest` is only needed to run
the test suite.

## Quick start

```python
from src.mdp import build_mdp
from src.optimal_policy import value_iteration

mdp = build_mdp(p=0.9, m=4, f0=0.75, delta=0.04)   # fT = f0 + delta
V, policy = value_iteration(mdp)

init_id = mdp["state2id"][tuple([0] * mdp["m"])]
print("Expected number of rounds to reach target fidelity:", -V[init_id])
```

## Reproducing the paper's figures

| Paper figure/table | Script | Notes |
|---|---|---|
| Fig. 2 (example policy table) | `src/optimal_policy.py` (run directly) | prints the policy for a single `(p, m, f0, delta)` |
| Fig. 3 (T_opt vs. f0, state-space discontinuities) | `scripts/reproduce_figures.py` (adapt grid) | see `tests/test_policies.py::test_finite_state_space_boundary` for the underlying finiteness check |
| Fig. 4 / 6 (baseline comparisons) | `scripts/reproduce_figures.py` | per-baseline expected waiting times |
| Fig. 7 (m-dependence, nested-purification limit) | `scripts/reproduce_figures.py` (adapt `m` sweep) | |
| Advantage heatmap, `(T_min(baselines) - T_opt) / T_min(baselines)` | `scripts/reproduce_figures.py` | |

```bash
python scripts/reproduce_figures.py
```

writes `data/results/advantage_grid.json` and `figures/advantage_heatmap.png`.

## Running the tests

```bash
pytest tests/ -v
```

## Notes on the two baseline-evaluation methods

`policy_eval.py` provides two ways to compute a fixed policy's expected
waiting time, both giving identical answers (see
`tests/test_policies.py::test_exact_and_vi_policy_evaluation_agree`):

- `exact_policy_evaluation` — computes the absorption probability `h(s)`
  (does this policy reach the target with probability 1 from state `s`?)
  via fixed-point iteration, then solves for the exact expected time on the
  "proper" subset directly via `numpy.linalg.solve`. This is what was used
  to generate the paper's reported numbers; it is fast (a single linear
  solve per policy).
- `fixed_policy_value_iteration` — evaluates the same fixed policy using
  the identical value-iteration recursion (γ = 1) as `optimal_policy.py`,
  just without maximizing over actions. Structurally uniform with the
  optimal-policy solver, at the cost of being slower (pure-Python sweeps).

Both are provided so results can be cross-checked against each other.

## Citing this work

See [`CITATION.cff`](CITATION.cff), or cite:

```bibtex
@article{yourbibkey,
  title   = {Optimizing Entanglement Distillation Policies via Markov Decision Process Formulation},
  author  = {Your Name and Coauthors},
  journal = {Journal name},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}
```

## License

MIT (see [`LICENSE`](LICENSE)) — see also the Zenodo archive: [DOI badge/link once created].
