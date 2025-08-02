__all__ = ['SolutionSpace']

import numpy as np

class SolutionSpace:
    """
    Defines the solution space for 5-dimensional optimization problems.
    """
    def __init__(self, dimension=5, bounds=(-32.768, 32.768)):
        self.dimension = dimension  
        self.bounds = bounds
        
    def random_solution(self):
        """
        Generate a random solution within the bounds.
        
        Returns:
        --------
        numpy.ndarray
            A random 5-dimensional point within the solution space
        """
        return np.random.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=self.dimension
        )
    
    def is_within_bounds(self, solution):
        """
        Check if a solution is within the defined bounds.
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Point to check
            
        Returns:
        --------
        bool
            True if solution is within bounds, False otherwise
        """
        if len(solution) != self.dimension:
            return False
        return np.all((solution >= self.bounds[0]) & (solution <= self.bounds[1]))
