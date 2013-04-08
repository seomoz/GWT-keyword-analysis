
import unittest
import numpy as np

from gwt_ka import data_process

class TestGWTReferrals(unittest.TestCase):

    def test_load_data(self):
        gwt = data_process.GWTReferrals.load_data('gwt_csv_test.csv')

        data_actual = np.array([[50, 16,  32,  2.4],
                    [1000,    100, 10,  1.2],
                    [15,  3,   20,  3.4],
                    [3,   0.6,   20,  3.8],
                    [1555,    555, 36,  2.9],
                    [20,  10,  50,  8.1],
                    [3,   0.975,   32.5,  9.6],
                    [20,  3,   15,  7.7],
                    [25000,   5000,    20,  3.8]])
        data_actual[:, 2] = data_actual[:, 2] / 100.0

        self.assertTrue(
            np.allclose(data_actual, gwt._data.view(np.float).reshape(9, 4)))

    def setUp(self):
        self.data = [['a', 100, 10, 10, 1.1],
                    ['b', 50, 10, 20, 2.4],
                    ['c', 75, 55, 73, 4.8],
                    ['d', 55, 15, 27, 2.2]]

    def test_get_position(self):
        gwt = data_process.GWTReferrals(self.data)
        self.assertTrue(
            (np.array([1, 2, 5, 2], np.int) == gwt.get_position()).all())

    def test_ctr_curve(self):
        gwt = data_process.GWTReferrals(self.data)
        computed_positions, computed_ctr = gwt.ctr_curve()

        actual_ctr = np.array([10.0 / 100,
                        (10.0 + 15.0) / (50 + 55),
                        np.nan,
                        np.nan,
                        55.0 / 75])
        actual_positions = np.array([1, 2, 3, 4, 5], np.int)

        self.check_ctr(actual_positions, actual_ctr,
                    computed_positions, computed_ctr)

    def test_ctr_curve_mask(self):
        gwt = data_process.GWTReferrals(self.data)
        mask = [True, False, True, True]
        computed_positions, computed_ctr = gwt.ctr_curve(mask)

        actual_ctr = np.array([10.0 / 100,
                        15.0 / 55,
                        np.nan,
                        np.nan,
                        55.0 / 75])
        actual_positions = np.array([1, 2, 3, 4, 5], np.int)

        self.check_ctr(actual_positions, actual_ctr,
                    computed_positions, computed_ctr)

    def check_ctr(self, actual_positions, actual_ctr,
            computed_positions, computed_ctr):
        self.assertTrue(np.allclose(actual_positions, computed_positions))
        mask = np.isnan(actual_ctr)
        self.assertTrue(np.allclose(actual_ctr[~mask], computed_ctr[~mask]))
        self.assertTrue(np.isnan(computed_ctr[mask]).all())




if __name__ == "__main__":
    unittest.main()


