import numpy as np
from utils.target_func import TargetFunction
from utils.solution_space import SolutionSpace
from sampler.llm_sampler import LLMSampler
from sampler.uniform_sampler import UniformSampler
from tabulate import tabulate
import csv
import os
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    def __init__(self, dimension, save_dir_prefix: str=""):
        """Initialize experiment parameters.
        
        Args:
            dimension (int): dimension of input vector
            save_dir_prefix (str): prefix for the save directory
        """
        self.dimension = dimension
        self.save_dir_prefix = save_dir_prefix
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.saved_dir = os.path.join(self.base_dir, 'exp_results', self.save_dir_prefix)
        # Create output directories
        os.makedirs(self.saved_dir, exist_ok=True) 
       
    @abstractmethod
    def run(self, **kwargs):
        """Run the experiment across all configurations."""
        pass

    @abstractmethod
    def save_samples(self, **kwargs):
        """Save samples to CSV."""
        pass
    
    @abstractmethod
    def save_summary(self, **kwargs):
        """Save summary results to CSV."""
        pass

    @abstractmethod
    def print_summary(self):
        """Print summary table to console."""
        pass
