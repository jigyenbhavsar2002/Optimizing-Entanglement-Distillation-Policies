"""
Baseline policies: entanglement pumping, nested, and greedy.

Each build_*_policy(mdp) function returns a fixed policy array (action id
per state id), to be evaluated by policy_eval.py.
"""

from collections import defaultdict

import numpy as np


def build_pumping_policy(mdp):
    """Entanglement pumping: each elementary (f0-fidelity) pair
    is greedily matched with the highest-fidelity remaining unmatched
    active link; leftover f0 pairs (no higher-fidelity partner available)
    are matched among themselves. Generate on all empty memories."""
    m, TARGET, F0_INDEX = mdp["m"], mdp["TARGET"], mdp["F0_INDEX"]
    states, action2id, actions = mdp["states"], mdp["action2id"], mdp["actions"]
    available_actions, IDLE_ACTION_ID = mdp["available_actions"], mdp["IDLE_ACTION_ID"]
    nS = mdp["nS"]
    policy = np.full(nS, -1, dtype=int)

    for s_id, s in enumerate(states):
        if TARGET in s:
            policy[s_id] = IDLE_ACTION_ID
            continue
        valid = [i for i in range(m) if 0 < s[i] < TARGET]
        f0_slots = [i for i in valid if s[i] == F0_INDEX]
        other_slots = sorted([i for i in valid if s[i] > F0_INDEX],
                              key=lambda i: s[i], reverse=True)
        matching_pairs, used_slots = [], set()
        if f0_slots:
            for f0_slot, partner in zip(f0_slots, other_slots):
                pair = (min(f0_slot, partner), max(f0_slot, partner))
                matching_pairs.append(pair)
                used_slots.add(f0_slot); used_slots.add(partner)
            remaining_f0 = [i for i in f0_slots if i not in used_slots]
            for k in range(0, len(remaining_f0) - 1, 2):
                a, b = remaining_f0[k], remaining_f0[k + 1]
                matching_pairs.append((min(a, b), max(a, b)))
                used_slots.add(a); used_slots.add(b)
        matching = frozenset(matching_pairs)
        g = tuple(1 if s[idx] == 0 else 0 for idx in range(m))
        key = (g, matching)
        chosen = action2id.get(key, None)
        if chosen is None or chosen not in available_actions[s]:
            chosen = None
            if matching:
                for a_id in available_actions[s]:
                    if actions[a_id][1] == matching:
                        chosen = a_id
                        break
            if chosen is None:
                chosen = action2id.get((g, frozenset()), available_actions[s][0])
        policy[s_id] = chosen
    return policy


def build_nested_policy(mdp):
    """nested purification: distill ONLY identical-
    fidelity pairs. Generate on all empty memories. Unmatched occupied
    links are left idle."""
    m, TARGET = mdp["m"], mdp["TARGET"]
    states, action2id = mdp["states"], mdp["action2id"]
    available_actions, IDLE_ACTION_ID = mdp["available_actions"], mdp["IDLE_ACTION_ID"]
    nS = mdp["nS"]
    policy = np.full(nS, -1, dtype=int)

    for s_id, s in enumerate(states):
        if TARGET in s:
            policy[s_id] = IDLE_ACTION_ID
            continue
        groups = defaultdict(list)
        for slot_idx, fid in enumerate(s):
            if 0 < fid < TARGET:
                groups[fid].append(slot_idx)
        matching_pairs = []
        for fid_val in groups:
            slots = groups[fid_val]
            for k in range(0, len(slots) - 1, 2):
                matching_pairs.append((slots[k], slots[k + 1]))
        matching = frozenset(matching_pairs)
        g = tuple(1 if s[idx] == 0 else 0 for idx in range(m))
        key = (g, matching)
        chosen = action2id.get(key, None)
        if chosen is None or chosen not in available_actions[s]:
            chosen = action2id.get((g, frozenset()), available_actions[s][0])
        policy[s_id] = chosen
    return policy


def build_greedy_policy(mdp):
    """Greedy distillation: repeatedly match the two active links with the smallest
    fidelity difference, until fewer than two remain. Generate on all
    empty memories."""
    m, TARGET, f = mdp["m"], mdp["TARGET"], mdp["f"]
    states, action2id, actions = mdp["states"], mdp["action2id"], mdp["actions"]
    available_actions, IDLE_ACTION_ID = mdp["available_actions"], mdp["IDLE_ACTION_ID"]
    nS = mdp["nS"]
    policy = np.full(nS, -1, dtype=int)

    for s_id, s in enumerate(states):
        if TARGET in s:
            policy[s_id] = IDLE_ACTION_ID
            continue
        valid = [i for i in range(m) if 0 < s[i] < TARGET]
        matching_pairs, used_slots = [], set()
        remaining = list(valid)
        while len(remaining) >= 2:
            best_pair, best_diff = None, float("inf")
            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    idx1, idx2 = remaining[a], remaining[b]
                    diff = abs(f[s[idx1] - 1] - f[s[idx2] - 1])
                    if diff < best_diff:
                        best_diff, best_pair = diff, (idx1, idx2)
            i, j = best_pair
            matching_pairs.append((min(i, j), max(i, j)))
            used_slots.add(i); used_slots.add(j)
            remaining = [r for r in remaining if r not in used_slots]
        matching = frozenset(matching_pairs)
        g = tuple(1 if (s[idx] == 0 and idx not in used_slots) else 0 for idx in range(m))
        key = (g, matching)
        chosen = action2id.get(key, None)
        if chosen is None or chosen not in available_actions[s]:
            chosen = None
            if matching:
                for a_id in available_actions[s]:
                    if actions[a_id][1] == matching:
                        chosen = a_id
                        break
            if chosen is None:
                g_gen = tuple(1 if s[idx] == 0 else 0 for idx in range(m))
                chosen = action2id.get((g_gen, frozenset()), available_actions[s][0])
        policy[s_id] = chosen
    return policy
