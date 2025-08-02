__all__ = ['UniformSampler']

import numpy as np
from utils.solution_space import SolutionSpace
from sampler.base_sampler import BaseSampler

class UniformSampler(BaseSampler):
    """
    Implements uniform random sampling from a solution space.
    """
    def __init__(self, solution_space):
        """
        Initialize sampler with a solution space.
        
        Parameters:
        -----------
        solution_space : SolutionSpace
            The solution space to sample from
        """
        super().__init__(solution_space)
        
    def sample(self, n_samples=1):
        """
        Generate random samples from the solution space using uniform distribution.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate (default: 1)
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, dimensions) containing the samples
        """
        np.random.seed(42)
        samples = np.random.uniform(
            low=self.solution_space.bounds[0],
            high=self.solution_space.bounds[1],
            size=(n_samples, self.solution_space.dimension)
        )
        return samples
