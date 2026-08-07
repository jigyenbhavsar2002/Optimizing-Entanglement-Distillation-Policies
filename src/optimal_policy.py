"""
Optimal policy via value iteration, gamma = 1 (undiscounted).
"""

import numpy as np

from .mdp import build_mdp


def value_iteration(mdp, gamma=1, theta=1e-6):
    """Standard value iteration. Returns (V, policy) arrays indexed by state id.
    """
    states, state2id, nS = mdp["states"], mdp["state2id"], mdp["nS"]
    P, available_actions = mdp["P"], mdp["available_actions"]

    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    while True:
        delta = 0.0
        for s_id, s in enumerate(states):
            acts = available_actions[s]
            if not acts:
                continue
            v = V[s_id]
            best_Q, best_a = -np.inf, None
            for a_id in acts:
                Q = 0.0
                for prob, ns, r, done in P[s_id][a_id]:
                    Q += prob * (r + gamma * V[ns] * (not done))
                if Q > best_Q:
                    best_Q, best_a = Q, a_id
            V[s_id] = best_Q
            policy[s_id] = best_a
            delta = max(delta, abs(v - V[s_id]))
        if delta < theta:
            break
    return V, policy


def expected_waiting_time(p, m, f0, delta, gamma=1, theta=1e-6):
    """Convenience wrapper: builds the MDP, solves it, and returns
    T_opt = -V(all-empty state)."""
    mdp = build_mdp(p, m, f0, delta)
    V, policy = value_iteration(mdp, gamma=gamma, theta=theta)
    init_id = mdp["state2id"][tuple([0] * m)]
    return -V[init_id], mdp, V, policy


if __name__ == "__main__":
    T_opt, mdp, V, policy = expected_waiting_time(p=0.9, m=4, f0=0.75, delta=0.04)
    print(f"K = {mdp['F']} reachable fidelity levels")
    print(f"T_opt (all-empty state) = {T_opt:.6f}")

