import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats._multivariate import multivariate_normal_frozen


from mr_playground.sensor import SensorBase


class SimpleLocalizer:
    def __init__(self, sensing_range: float, fov: float):
        """
        Initialize the SimpleLocalizer sensor.

        :param sensing_range: Maximum distance the sensor can detect an object.
        :param fov: Field of View (FoV) in degrees.
        """
        self._sensing_range = sensing_range
        self._fov = np.radians(fov)  # Convert to radians

    def read(self,
             pos: np.ndarray,
             ground_truth: np.ndarray,
             noise_factor: float = 1.0) -> multivariate_normal_frozen:
        """
        Simulates sensor reading with Gaussian noise.

        :param pos: np.ndarray representing the sensor's position [x, y].
        :param ground_truth: np.ndarray representing the object's true position [x, y].
        :param noise_factor: Scaling factor for noise.
        :return: A multivariate_normal object representing the reading. If the object is out of range/FoV,
                 a uniform distribution covering the sensing range is returned.
        """
        # Compute the distance and direction of the object relative to the sensor
        delta_pos = ground_truth - pos
        distance = np.linalg.norm(delta_pos)

        # Compute the angle between the sensor's forward axis (x-direction) and the object
        angle = np.arctan2(delta_pos[1], delta_pos[0])  # Angle in radians

        # If the object is out of range or outside the field of view, return a uniform distribution
        if distance > self._sensing_range or np.abs(angle) > self._fov / 2:
            # Uniform distribution: Mean at center of sensor range, very high variance
            # Arbitrary central point in range
            uniform_mean = pos + np.array([self._sensing_range / 2, 0])
            # Large variance to simulate uncertainty
            high_variance = (self._sensing_range / 2) ** 2
            cov_matrix = np.array([[high_variance, 0], [0, high_variance]])
            return mvn(mean=uniform_mean, cov=cov_matrix)

        # Noise increases with distance
        std_dev = noise_factor * distance
        cov_matrix = np.array([[std_dev**2, 0],
                               [0, std_dev**2]])

        # Apply Gaussian noise to the ground truth measurement
        noisy_measurement = ground_truth + \
            np.random.multivariate_normal([0, 0], cov_matrix)

        # Return a Gaussian distribution representing the measurement
        return mvn(mean=noisy_measurement, cov=cov_matrix)
