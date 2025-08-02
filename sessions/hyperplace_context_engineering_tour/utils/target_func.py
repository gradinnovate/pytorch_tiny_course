import numpy as np
__all__ = ['TargetFunction']
class TargetFunction:
    def __init__(self, name, dimension=5):
        self.name = name
        self.dimension = dimension
    def evaluate(self, x):
        if self.name == 'rosenbrock':
            return self.rosenbrock_function(x)
        elif self.name == 'griewank':
            return self.griewank_function(x)
        elif self.name == 'rastrigin':
            return self.rastrigin_function(x)
        elif self.name == 'michalewicz':
            return self.michalewicz_function(x)
        elif self.name == 'schwefel':
            return self.schwefel_function(x)
        elif self.name == 'sphere':
            return self.sphere_function(x)
        else:
            raise ValueError(f"Target function {self.name} not implemented")
        
    def rosenbrock_function(self, x):
        """
        Implements the {self.dimension}-dimensional Rosenbrock function.
        Input: x - numpy array of {self.dimension} elements
        Output: value of the Rosenbrock function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        result = 0
        for i in range(self.dimension-1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result

    def griewank_function(self, x):
        """
        Implements the {self.dimension}-dimensional Griewank function.
        Input: x - numpy array of {self.dimension} elements
        Output: value of the Griewank function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1))))
        return 1 + sum_term - prod_term

    def rastrigin_function(self, x):
        """
        Implements the {self.dimension}-dimensional Rastrigin function.
        Input: x - numpy array of {self.dimension} elements 
        Output: value of the Rastrigin function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        A = 10
        n = len(x)
        sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x))
        result = A * n + sum_term
        return result

    def michalewicz_function(self, x):
        """
        Implements the {self.dimension}-dimensional Michalewicz function.
        Input: x - numpy array of {self.dimension} elements
        Output: value of the Michalewicz function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        m = 10  # Steepness parameter
        i = np.arange(1, self.dimension + 1)
        return -np.sum(np.sin(x) * np.sin((i * x**2) / np.pi)**(2*m))

    def schwefel_function(self, x):
        """
        Implements the {self.dimension}-dimensional Schwefel function.
        Input: x - numpy array of {self.dimension} elements
        Output: value of the Schwefel function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        return 418.9829 * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def sphere_function(self, x):
        """
        Implements the {self.dimension}-dimensional Sphere function.
        Input: x - numpy array of {self.dimension} elements
        Output: value of the Sphere function at point x
        """
        if len(x) != self.dimension:
            raise ValueError(f"Input must be {self.dimension}-dimensional")
            
        result = np.sum(x**2)
        return result

    @staticmethod
    def get_supported_functions():
        """
        Returns a list of supported target function names.
        
        Returns:
        --------
        list
            List of strings containing the names of supported target functions
        """
        #return ['rosenbrock', 'griewank', 'rastrigin', 'schwefel', 'sphere']
        return ['rastrigin']
    
    @staticmethod
    def get_all_functions():
        return ['rosenbrock', 'michalewicz','griewank', 'rastrigin', 'schwefel', 'sphere']
