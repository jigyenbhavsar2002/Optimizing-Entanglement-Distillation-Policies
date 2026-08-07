"""
Evaluate a FIXED policy's expected waiting time.
"""

import numpy as np


def evaluate_policy(mdp, policy, gamma=1, theta=1e-6, max_iter=3000,
                     probe_iters=50, probe_tol=1e-3):
    """Evaluate a fixed policy via value iteration

    Deadlocking states -- those from which `policy` does not reach the
    target with probability 1 -- cause the gamma=1 recursion to fail to
    converge (V decreases without bound). We detect this by continuing to
    sweep for `probe_iters` iterations after nominal convergence and
    flagging any state whose value is still moving by more than
    `probe_tol` as a deadlock (V = -inf there).

    Returns (V, proper_mask): V is indexed by state id (V[s] = -inf for
    deadlocking states), and proper_mask[s] is True iff state s reaches
    the target with probability 1 under `policy`.
    """
    states, nS, P = mdp["states"], mdp["nS"], mdp["P"]
    TARGET = mdp["TARGET"]
    target_mask = np.array([TARGET in s for s in states])

    V = np.zeros(nS)
    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s_id, s in enumerate(states):
            if target_mask[s_id]:
                continue
            a_id = policy[s_id]
            Q = 0.0
            for prob, ns, r, done in P[s_id][a_id]:
                Q += prob * (r + gamma * V[ns] * (not done))
            delta = max(delta, abs(V[s_id] - Q))
            V_new[s_id] = Q
        V = V_new
        if delta < theta:
            break

    V_probe = V.copy()
    for _ in range(probe_iters):
        for s_id, s in enumerate(states):
            if target_mask[s_id]:
                continue
            a_id = policy[s_id]
            Q = 0.0
            for prob, ns, r, done in P[s_id][a_id]:
                Q += prob * (r + gamma * V_probe[ns] * (not done))
            V_probe[s_id] = Q
    diverging = np.abs(V_probe - V) > probe_tol

    V_final = V.copy()
    V_final[diverging] = -np.inf
    proper_mask = (~diverging) | target_mask
    return V_final, proper_mask


def expected_waiting_time(mdp, policy, **kwargs):
    """Returns T = -V(all-empty state), or np.inf if that state deadlocks
    under this policy."""
    V, proper_mask = evaluate_policy(mdp, policy, **kwargs)
    init_id = mdp["state2id"][tuple([0] * mdp["m"])]
    return np.inf if not proper_mask[init_id] else -V[init_id]
