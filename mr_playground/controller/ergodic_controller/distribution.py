import numpy as np


class Distribution(object):
    def __init__(self, map_size=(10, 10), num_points=20):
        """
        Initializes the probability distribution over a given map size.

        :param map_size: (tuple) (width, height) of the map.
        :param num_points: (int) Number of grid points per dimension.
        """
        self._num_points = num_points
        self._map_size = map_size  # Custom map size

        # Create a mesh grid for the given map size
        x_vals = np.linspace(0, map_size[0], num_points)
        y_vals = np.linspace(0, map_size[1], num_points)
        grid = np.meshgrid(x_vals, y_vals)
        self._grid = np.c_[grid[0].ravel(), grid[1].ravel()]  # Flatten grid

        # Default mean and variance (scaled to map size)
        self._means = [np.array(map_size) / 2]  # Center of the map
        # Variance scales with map size
        self._vars = [(np.array(map_size) * 5)**2]

        self._has_update = False
        self._grid_vals = self._compute_distribution(self._grid)

    def update(self, means: np.ndarray, vars: np.ndarray):
        self._means.append(means)
        self._vars.append(vars)
        self._grid_vals = self._compute_distribution(self._grid)

    def get_grid_spec(self):
        """
        Returns the grid structure and corresponding distribution values.

        :return: Tuple of (meshgrid coordinates, reshaped grid values).
        """
        xy = []
        for g in self._grid.T:
            xy.append(np.reshape(g, newshape=(
                self._num_points, self._num_points)))
        return xy, self._grid_vals.reshape(self._num_points, self._num_points)

    def _compute_distribution(self, x: np.ndarray):
        """
        Computes the probability density at given points.

        :param x: np.ndarray of shape (N, 2), containing points to evaluate.
        :return: np.ndarray of probability values.
        """
        assert len(x.shape) > 1, 'Input needs to be of size N x 2'
        assert x.shape[1] == 2, 'Does not have correct exploration dimension'

        val = np.zeros(x.shape[0])
        for m, v in zip(self._means, self._vars):
            innerds = np.sum((x - m)**2 / v, axis=1)
            val += np.exp(-innerds / 2.0)

        # Normalize the distribution to ensure it sums to 1
        val /= np.sum(val)
        return val
