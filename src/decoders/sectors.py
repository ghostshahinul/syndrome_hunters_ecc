"""Project each physical fault once per connected detector sector."""
from collections import defaultdict

import stim


def sector_model(dem, odds_scale=1.0, *, center=False):
    """Recombine parts of one Bernoulli fault, retaining its logical frame.

    At unit scale the distribution is identical. Setting center=True removes
    deterministic reference shifts, which the caller must apply at both ends.
    """
    flat = dem.flattened()
    parent = list(range(dem.num_detectors))

    def root(site):
        while site != parent[site]:
            parent[site] = site = parent[parent[site]]
        return site

    for instruction in flat:
        if instruction.type == "error":
            for part in instruction.target_groups():
                sites = [t.val for t in part if t.is_relative_detector_id()]
                if len(sites) == 2:
                    parent[root(sites[1])] = root(sites[0])

    labels = [root(site) for site in range(len(parent))]
    result = stim.DetectorErrorModel()
    for instruction in flat:
        if instruction.type != "error":
            result.append(instruction)
            continue
        groups = instruction.target_groups()
        sectors = defaultdict(set)
        for part in groups:
            sector = next((labels[t.val] for t in part if t.is_relative_detector_id()), -1)
            sectors[sector].symmetric_difference_update(part)
        if all(sum(t.is_relative_detector_id() for t in part) <= 2 for part in sectors.values()):
            groups = [sorted(sectors[k], key=lambda t: (t.is_logical_observable_id(), t.val))
                      for k in sorted(sectors) if sectors[k]]
        # A logical-only part belongs to the same physical fault. Native
        # correlated matching expects every separator-delimited edge to have a detector.
        edge_groups = []
        standalone_frame = set()
        for part in groups:
            if any(t.is_relative_detector_id() for t in part):
                edge_groups.append(part)
            else:
                standalone_frame.symmetric_difference_update(part)
        if standalone_frame and edge_groups:
            edge_groups[0] = sorted(set(edge_groups[0]) ^ standalone_frame,
                                    key=lambda t: (t.is_logical_observable_id(), t.val))
        groups = edge_groups or ([sorted(standalone_frame, key=lambda t: t.val)] if standalone_frame else [])
        targets = []
        for part in groups:
            if targets:
                targets.append(stim.target_separator())
            targets.extend(part)
        probability, = instruction.args_copy()
        if center and probability > .5:
            probability = 1 - probability
        if targets and probability:
            if odds_scale != 1 and 0 < probability < .5:
                odds = probability / (1 - probability) * odds_scale ** len(groups)
                probability = odds / (1 + odds)
            result.append("error", probability, targets)
    # Centering a deterministic fault may remove the only mention of a detector or observable.
    if result.num_detectors < dem.num_detectors:
        result.append("detector", [], [stim.target_relative_detector_id(dem.num_detectors - 1)])
    if result.num_observables < dem.num_observables:
        result.append("logical_observable", [], [stim.target_logical_observable_id(dem.num_observables - 1)])
    return result
