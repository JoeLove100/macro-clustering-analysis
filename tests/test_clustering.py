import unittest
import pandas as pd
from clustering import _corr_to_distance, _get_min_indices, _get_avg_distance, _get_silhouette_cost


class TestClustering(unittest.TestCase):

    def test_corr_to_distance(self):
        # arrange
        test_corr = pd.DataFrame([[1, 0.5, -0.6], [0.5, 1, 0], [-0.6, 0, 1]])

        # act
        result = _corr_to_distance(test_corr)

        # assert
        expected_result = pd.DataFrame([[0, 1, 1.78885438],
                                        [1, 0, 1.41421356],
                                        [1.78885438, 1.41421356, 0]])
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_min_indices(self):
        # arrange
        test_distances = pd.DataFrame([[0, 0.3, 1.4, 0.1],
                                       [0.3, 0, 0.8, 0.7],
                                       [1.4, 0.8, 0, 1.7],
                                       [0.1, 0.7, 1.7, 0]])

        # assert
        result = _get_min_indices(test_distances)

        # assert
        self.assertEqual((0, 3), result)

    def test_get_avg_distance_in_cluster(self):
        # arrange
        assets = [f"asset_{i}" for i in range(4)]
        test_distances = pd.DataFrame([[0, 0.3, 1.4, 0.1],
                                       [0.3, 0, 0.8, 0.7],
                                       [1.4, 0.8, 0, 1.7],
                                       [0.1, 0.7, 1.7, 0]], index=assets, columns=assets)

        # act
        result = _get_avg_distance(test_distances, "asset_1", ["asset_1", "asset_2", "asset_3"])

        # assert
        self.assertAlmostEqual(0.75, result)

    def test_get_avg_distance_out_of_cluster(self):
        # arrange
        assets = [f"asset_{i}" for i in range(4)]
        test_distances = pd.DataFrame([[0, 0.3, 1.4, 0.1],
                                       [0.3, 0, 0.8, 0.7],
                                       [1.4, 0.8, 0, 1.7],
                                       [0.1, 0.7, 1.7, 0]], index=assets, columns=assets)

        # act
        result = _get_avg_distance(test_distances, "asset_1", ["asset_0", "asset_2", "asset_3"])

        # assert
        self.assertAlmostEqual(0.6, result)

    def test_get_silhouette_cost(self):
        # arrange
        assets = [f"asset_{i}" for i in range(4)]
        test_distances = pd.DataFrame([[0, 0.3, 1.4, 0.1],
                                       [0.3, 0, 0.8, 0.7],
                                       [1.4, 0.8, 0, 1.7],
                                       [0.1, 0.7, 1.7, 0]], index=assets, columns=assets)
        test_grouping = {0: ["asset_0", "asset_3"], 1: ["asset_1"], 2: ["asset_2"]}

        # act
        result = _get_silhouette_cost(test_distances, test_grouping)

        # assert
        self.assertAlmostEqual(-0.38095238, result)
