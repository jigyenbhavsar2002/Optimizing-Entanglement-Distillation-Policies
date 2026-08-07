"""
Core MDP construction for the entanglement-distillation problem:
distillation fidelity map, exact reachable fidelity-level closure,
state/action spaces, and transition model
"""

import itertools
from collections import defaultdict
from itertools import combinations

import numpy as np


def distillation(f1, f2):
    """2-to-1 Deutsch/BBPSSW distillation map

    Returns (f_out, p_success) for two Werner states of fidelity f1, f2.
    """
    f_dist = (f1 * f2 + ((1 - f1) * (1 - f2)) / 9) / (
        f1 * f2 + (f1 * (1 - f2) + f2 * (1 - f1)) / 3 + (5 * (1 - f1) * (1 - f2)) / 9
    )
    p_dist = f1 * f2 + (f1 * (1 - f2) + f2 * (1 - f1)) / 3 + (5 * (1 - f1) * (1 - f2)) / 9
    return f_dist, p_dist


def build_mdp(p, m, f0, delta):
    """Build the full MDP for given generation probability p, number of
    memories m, initial fidelity f0, and target gap delta (fT = f0 + delta).

    Returns a dict with the reachable fidelity levels, state/action spaces,
    and the transition table P, keyed the same way throughout the repo:
        P[state_id][action_id] -> list of (prob, next_state_id, reward, done)
    """
    ft = f0 + delta

    # ---- exact reachable fidelity-level closure (see Appendix D: finite
    #      iff ft < f_infty(f0)) ----
    S, Sf = [], []
    Sn = [f0]
    while Sn:
        new_state, target_state = [], []
        for i in Sn:
            fself = distillation(i, i)[0]
            (new_state if fself < ft else target_state).append(fself)
            for j in S:
                fcomb = distillation(i, j)[0]
                (new_state if fcomb < ft else target_state).append(fcomb)
        for i, j in combinations(Sn, 2):
            fcomb = distillation(i, j)[0]
            (new_state if fcomb < ft else target_state).append(fcomb)
        Sf.extend(target_state)
        S.extend(Sn)
        Sn = new_state

    f = sorted(S)
    F = len(f)
    TARGET = F + 1
    F0_INDEX = 1  # index 1 always corresponds to f[0] == f0
    fidelities = list(range(F + 2))  # 0 = empty, 1..F = fidelity idx, F+1 = target

    # ---- state space ----
    states = list(itertools.product(fidelities, repeat=m))
    state2id = {s: idx for idx, s in enumerate(states)}
    nS = len(states)

    # ---- action space: all partial matchings (disjoint pairs to distill)
    #      x all generation patterns on the remaining free memories ----
    def get_all_partial_matchings(indices):
        if len(indices) < 2:
            return [frozenset()]
        matchings = [frozenset()]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pair = (indices[i], indices[j])
                rest = [indices[k] for k in range(len(indices)) if k != i and k != j]
                for sub in get_all_partial_matchings(rest):
                    matchings.append(frozenset([pair]) | sub)
        return list({frozenset(mm) for mm in matchings})

    def build_action_space(m):
        matchings = get_all_partial_matchings(list(range(m)))
        actions = []
        for matching in matchings:
            distilled = set(i for pair in matching for i in pair)
            free = [k for k in range(m) if k not in distilled]
            for pattern in itertools.product([0, 1], repeat=len(free)):
                g = [0] * m
                for idx, val in zip(free, pattern):
                    g[idx] = val
                actions.append((tuple(g), matching))
        seen, unique = set(), []
        for a in actions:
            key = (a[0], a[1])
            if key not in seen:
                seen.add(key)
                unique.append(a)
        action2id = {(a[0], a[1]): idx for idx, a in enumerate(unique)}
        return unique, action2id, len(unique)

    actions, action2id, nA = build_action_space(m)
    IDLE_ACTION_ID = action2id[(tuple([0] * m), frozenset())]

    # ---- available actions per state ----
    available_actions = {}
    for s in states:
        if TARGET in s:
            available_actions[s] = [IDLE_ACTION_ID]
            continue
        acts = []
        for a_id, (g, matching) in enumerate(actions):
            valid = True
            for (i, j) in matching:
                if s[i] == 0 or s[j] == 0 or s[i] == TARGET or s[j] == TARGET:
                    valid = False
                    break
            if not valid:
                continue
            for idx in range(m):
                if g[idx] == 1 and s[idx] != 0:
                    valid = False
                    break
            if valid:
                acts.append(a_id)
        available_actions[s] = acts

    REWARD_GEN = -1
    REWARD_DISTILL = -2
    REWARD_TARGET = 0

    # ---- transition model: distillation (all matched pairs resolved
    #      simultaneously) then generation (all flagged empty memories) ----
    def combined_transition(state, action):
        g_mask, matching = action
        if not matching:
            distill_outcomes = [(list(state), 1.0)]
        else:
            distill_outcomes = [(list(state), 1.0)]
            for (i, j) in list(matching):
                new_outcomes = []
                for s_list, prob in distill_outcomes:
                    if s_list[i] == 0 or s_list[j] == 0:
                        new_outcomes.append((s_list, prob))
                        continue
                    control = i if s_list[i] >= s_list[j] else j
                    other = j if control == i else i
                    f1v = f[s_list[control] - 1]
                    f2v = f[s_list[other] - 1]
                    f_new, p_succ = distillation(f1v, f2v)
                    new_idx = (TARGET if f_new >= ft
                               else int(np.argmin(np.abs(np.array(f) - f_new))) + 1)
                    s_succ = list(s_list); s_succ[control] = new_idx; s_succ[other] = 0
                    s_fail = list(s_list); s_fail[control] = 0; s_fail[other] = 0
                    new_outcomes.append((s_succ, prob * p_succ))
                    new_outcomes.append((s_fail, prob * (1 - p_succ)))
                distill_outcomes = new_outcomes

        next_states = defaultdict(float)
        for s_mid, p_mid in distill_outcomes:
            zero_indices = [i for i in range(m) if s_mid[i] == 0 and g_mask[i] == 1]
            n = len(zero_indices)
            for mask in range(1 << n):
                prob = p_mid
                s_list = list(s_mid)
                for bit, idx in enumerate(zero_indices):
                    if (mask >> bit) & 1:
                        s_list[idx] = 1; prob *= p
                    else:
                        prob *= (1 - p)
                next_states[tuple(s_list)] += prob
        return next_states

    P = {s_id: {} for s_id in range(nS)}
    for s in states:
        s_id = state2id[s]
        if TARGET in s:
            P[s_id][IDLE_ACTION_ID] = [(1.0, s_id, REWARD_TARGET, True)]
            continue
        for a_id in available_actions[s]:
            action = actions[a_id]
            trans = combined_transition(s, action)
            lst = []
            for ns, prob in trans.items():
                if prob < 1e-15:
                    continue
                done = TARGET in ns
                reward = REWARD_TARGET if done else (REWARD_DISTILL if action[1] else REWARD_GEN)
                lst.append((prob, state2id[ns], reward, done))
            P[s_id][a_id] = lst

    return dict(
        p=p, m=m, f0=f0, delta=delta, ft=ft,
        f=f, F=F, TARGET=TARGET, F0_INDEX=F0_INDEX, fidelities=fidelities,
        states=states, state2id=state2id, nS=nS,
        actions=actions, action2id=action2id, nA=nA, IDLE_ACTION_ID=IDLE_ACTION_ID,
        available_actions=available_actions, P=P,
        REWARD_GEN=REWARD_GEN, REWARD_DISTILL=REWARD_DISTILL, REWARD_TARGET=REWARD_TARGET,
    )


def f_infty(f0):
    """Closed-form fixed point of pure pumping (Eq. 7): F = D(F, f0).
    The state space built by build_mdp() is finite iff ft < f_infty(f0)
    (see Appendix D)."""
    return (-3 + 6 * f0 + np.sqrt(7 - 26 * f0 + 28 * f0 ** 2)) / (-2 + 8 * f0)

