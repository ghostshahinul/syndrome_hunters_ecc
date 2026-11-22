"""An exact syndrome join for explanations with at most two DEM faults."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numba import njit

from src.decoders.faultjoin.faults import Fault


class JoinIndex(NamedTuple):
    salt: np.ndarray
    buckets: np.ndarray
    heads: np.ndarray
    links: np.ndarray
    site_starts: np.ndarray
    sites: np.ndarray
    frames: np.ndarray
    odds: np.ndarray
    offsets: np.ndarray
    incident: np.ndarray
    keys: np.ndarray
    decisions: np.ndarray
    zero: np.uint64
    max_hits: int


def make_join(faults: list[Fault], n: int) -> JoinIndex:
    # Hashes select a short bucket. Every candidate is checked against the
    # complete detector sets, so a collision can only affect the work performed.
    size = 1 << max(1, (2 * len(faults)).bit_length())
    salt = np.random.default_rng(314159).integers(size, size=n, dtype=np.int32)
    buckets = np.zeros(len(faults), np.int32)
    heads = np.full(size, -1, np.int32)
    links = np.empty(len(faults), np.int32)
    site_starts = np.cumsum([0] + [len(f.detectors) for f in faults], dtype=np.int32)
    sites = np.array([d for f in faults for d in f.detectors], np.int32)
    incident = [[] for _ in range(n)]
    tabulate = n <= 64 and len(faults) <= 350
    signatures = []
    frames = np.array([f.logical for f in faults], np.uint64)
    odds = np.array([f.probability / (1 - f.probability) for f in faults])
    for i, f in enumerate(faults):
        signatures.append(sum(1 << d for d in f.detectors) if tabulate else f.detectors)
        for d in f.detectors:
            incident[d].append(i)
            buckets[i] ^= salt[d]
        links[i] = heads[buckets[i]]
        heads[buckets[i]] = i
    offsets = np.cumsum([0] + [len(ids) for ids in incident], dtype=np.int32)
    flat_incident = np.array([i for ids in incident for i in ids], np.int32)

    # Small circuits can tabulate the entire order-two posterior once. Large
    # circuits precompute only the empty syndrome and use the pivot join below.
    mass = defaultdict(lambda: defaultdict(float))
    mass[0][0] = 1.0  # no-fault explanation, dividing out the common prior
    same_signature = defaultdict(list)
    for i, f in enumerate(faults):
        sig = signatures[i]
        if tabulate or not sig:
            mass[sig if tabulate else 0][f.logical] += odds[i]
        partners = range(i) if tabulate else same_signature[sig]
        for j in partners:
            key = sig ^ signatures[j] if tabulate else 0
            mass[key][f.logical ^ faults[j].logical] += odds[i] * odds[j]
        same_signature[sig].append(i)

    def winner(votes):
        return min(votes, key=lambda frame: (-votes[frame], frame))

    keys = sorted(mass) if tabulate else []
    return JoinIndex(
        salt, buckets, heads, links, site_starts, sites, frames, odds, offsets, flat_incident,
        np.array(keys, np.uint64), np.array([winner(mass[k]) for k in keys], np.uint64),
        np.uint64(winner(mass[0])), 2 * max((len(f.detectors) for f in faults), default=0),
    )


@njit(cache=True)
def _matches(hits, count, sites, i, end_i, j, end_j):
    """Compare sorted hits with the exact symmetric difference of two faults."""
    seen = 0
    while i < end_i or j < end_j:
        if j == end_j or (i < end_i and sites[i] < sites[j]):
            d = sites[i]
            i += 1
        elif i == end_i or sites[j] < sites[i]:
            d = sites[j]
            j += 1
        else:
            i += 1
            j += 1
            continue
        if seen >= count or hits[seen] != d:
            return False
        seen += 1
    return seen == count


@njit(cache=True)
def join_decision(hits, count, index):
    if count == 0:
        return True, index.zero
    if count > index.max_hits:
        return False, np.uint64(0)
    if len(index.keys):
        bits = np.uint64(0)
        for j in range(count):
            bits |= np.uint64(1) << np.uint64(hits[j])
        pos = np.searchsorted(index.keys, bits)
        if pos < len(index.keys) and index.keys[pos] == bits:
            return True, index.decisions[pos]
        return False, np.uint64(0)

    bucket = 0
    pivot = hits[0]
    for j in range(count):
        d = hits[j]
        bucket ^= index.salt[d]
        if index.offsets[d + 1] - index.offsets[d] < index.offsets[pivot + 1] - index.offsets[pivot]:
            pivot = d

    votes = {np.uint64(0): 0.0}
    b = index.heads[bucket]
    while b >= 0:
        if _matches(hits, count, index.sites, 0, 0, index.site_starts[b], index.site_starts[b + 1]):
            frame = index.frames[b]
            votes[frame] = votes.get(frame, 0.0) + index.odds[b]
        b = index.links[b]
    # The pivot fired, so exactly one member of a two-fault explanation
    # contains it. Enumerating this member counts each unordered pair once.
    for k in range(index.offsets[pivot], index.offsets[pivot + 1]):
        a = index.incident[k]
        b = index.heads[bucket ^ index.buckets[a]]
        while b >= 0:
            if _matches(hits, count, index.sites, index.site_starts[a], index.site_starts[a + 1],
                        index.site_starts[b], index.site_starts[b + 1]):
                frame = index.frames[a] ^ index.frames[b]
                votes[frame] = votes.get(frame, 0.0) + index.odds[a] * index.odds[b]
            b = index.links[b]
    best, best_mass = np.uint64(0), 0.0
    for frame, value in votes.items():
        if value > best_mass or (value == best_mass and frame < best):
            best, best_mass = frame, value
    return best_mass > 0, best
