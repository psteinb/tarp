import unittest

import numpy as np
from tarp import get_tarp_coverage


def get_test_data(scalefugde=1.0):
    num_samples = 100
    num_sims = 100
    num_dims = 5
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))
    log_sigma = np.random.uniform(low=-5, high=-1, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(
        loc=theta, scale=scalefugde * sigma, size=(num_samples, num_sims, num_dims)
    )
    theta = np.random.normal(loc=theta, scale=sigma, size=(num_sims, num_dims))
    return samples, theta


class TarpTest(unittest.TestCase):
    def test_single(self):
        samples, theta = get_test_data()
        ecp, alpha = get_tarp_coverage(
            samples,
            theta,
            references="random",
            metric="euclidean",
            norm=False,
            bootstrap=False,
        )
        print("test_single", np.max(np.abs(ecp - alpha)))
        self.assertAlmostEqual(np.max(np.abs(ecp - alpha)), 0.0, delta=0.1)

    def test_norm(self):
        samples, theta = get_test_data()
        ecp, alpha = get_tarp_coverage(
            samples,
            theta,
            references="random",
            metric="euclidean",
            norm=True,
            bootstrap=False,
        )
        print("test_norm", np.max(np.abs(ecp - alpha)))
        self.assertAlmostEqual(np.max(np.abs(ecp - alpha)), 0.0, delta=0.1)

    def test_bootstrap(self):
        samples, theta = get_test_data()
        ecp, alpha = get_tarp_coverage(
            samples,
            theta,
            references="random",
            metric="euclidean",
            norm=False,
            bootstrap=True,
        )
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)
        # self.assertAlmostEqual(np.max(np.abs(ecp_mean - alpha)/ecp_std), 0., delta=10.)
        self.assertAlmostEqual(np.max(np.abs(ecp_mean - alpha)), 0.0, delta=0.12)


class ConfidenceTest(unittest.TestCase):
    def test_underdispersed(self):
        samples, theta = get_test_data(0.05)
        ecp, alpha = get_tarp_coverage(
            samples,
            theta,
            references="random",
            metric="euclidean",
            norm=False,
            bootstrap=False,
        )
        print("test_underdispersed", np.max(np.abs(ecp - alpha)))
        self.assertNotAlmostEqual(np.max(np.abs(ecp - alpha)), 0.0, delta=0.1)

    def test_overdispersed(self):
        samples, theta = get_test_data(20.0)
        ecp, alpha = get_tarp_coverage(
            samples,
            theta,
            references="random",
            metric="euclidean",
            norm=False,
            bootstrap=False,
        )
        print("test_overdispersed", np.max(np.abs(ecp - alpha)))
        self.assertNotAlmostEqual(np.max(np.abs(ecp - alpha)), 0.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
