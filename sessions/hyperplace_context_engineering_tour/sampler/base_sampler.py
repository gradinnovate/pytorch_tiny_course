__all__ = ['BaseSampler']

from abc import ABC, abstractmethod
from utils.solution_space import SolutionSpace

class BaseSampler(ABC):
    """
    Abstract base class for implementing different sampling strategies.
    """
    def __init__(self, solution_space):
        """
        Initialize sampler with a solution space.
        
        Parameters:
        -----------
        solution_space : SolutionSpace
            The solution space to sample from
        """
        if not isinstance(solution_space, SolutionSpace):
            raise TypeError("solution_space must be an instance of SolutionSpace")
        self.solution_space = solution_space

    @abstractmethod
    def sample(self, n_samples=1):
        """
        Generate samples from the solution space.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate (default: 1)
            
        Returns:
        --------
        Samples from the solution space
        """
        pass
