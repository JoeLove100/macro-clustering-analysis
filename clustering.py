"""
Macro clustering analysis

This module exposes a collection of functions to replicate the heirarchical clustering analysis carried out
by SBV Research in the 2020 paper 'Asset Allocation via Clustering'
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple

Grouping = Dict[int, List[str]]


def _corr_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:

    return np.sqrt(2 * (-corr_matrix.subtract(1)))


def _get_min_indices(distances: pd.DataFrame) -> Tuple[int, int]:

    np.fill_diagonal(distances.values, math.inf)  # otherwise returns 0 elements from diagonal
    min_distance = distances.min().min()
    min_indices = (distances == min_distance).dot(distances.columns)
    min_indices = min_indices[min_indices != 0]
    return min_indices.index[0], min_indices.iloc[0]


def _get_avg_distance(distances: pd.DataFrame,
                      asset: str,
                      group_assets: List[str]) -> float:

    avg_distance = distances.loc[asset, group_assets].sum() / len(group_assets)
    if asset in group_assets:
        avg_distance *= len(group_assets) / (len(group_assets) - 1)

    return avg_distance


def _get_silhouette_cost(distances: pd.DataFrame,
                         grouping: Grouping) -> float:

    all_silhouettes = []

    for i, assets in grouping.items():

        if len(assets) == 1:
            all_silhouettes.append(0)
        else:
            for asset in assets:
                in_group = min_out_group = math.inf
                for j in list(grouping):
                    avg_distance = _get_avg_distance(distances, asset, grouping[j])
                    if i == j:
                        in_group = avg_distance
                    else:
                        min_out_group = min(min_out_group, avg_distance)

                silhouette = (min_out_group - in_group) / max(min_out_group, in_group)
                all_silhouettes.append(silhouette)

    mean_silhouette_cost = -sum(all_silhouettes) / len(all_silhouettes)
    return mean_silhouette_cost


def get_best_grouping(returns: pd.DataFrame,
                      cost_function: Callable[[pd.DataFrame, Grouping], float],
                      ) -> Grouping:

    grouping = {i: [asset_class] for i, asset_class in enumerate(returns.columns)}
    all_distances = _corr_to_distance(returns.corr())
    returns.columns = list(range(returns.shape[1]))
    all_groupings = []

    while True:

        # get the cost of the current grouping
        cost_of_grouping = cost_function(all_distances, grouping)
        all_groupings.append((grouping.copy(), cost_of_grouping))

        if len(grouping) == 2:
            break

        # calculate 2 groups with smallest distance
        distances = _corr_to_distance(returns.corr())
        group_1, group_2 = _get_min_indices(distances)

        # merge groups and update the returns data
        grouping[group_1].extend(grouping[group_2])
        grouping.pop(group_2)
        returns[group_1] = 0.5 * (returns[group_1] + returns[group_2])
        returns = returns.drop(group_2, axis=1)

    best_grouping, min_cost = all_groupings[0][0], all_groupings[0][1]
    for grouping, cost in all_groupings:
        if cost < min_cost:
            best_grouping = grouping
            min_cost = cost

    return best_grouping
