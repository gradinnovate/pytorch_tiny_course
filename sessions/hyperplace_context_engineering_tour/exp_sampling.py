import numpy as np
from utils.target_func import TargetFunction
from utils.solution_space import SolutionSpace
from sampler.llm_sampler import LLMSampler
from sampler.uniform_sampler import UniformSampler
from base_experiment import BaseExperiment
from tabulate import tabulate
import csv
import os
from tqdm import tqdm


class SamplingExperiment(BaseExperiment):
    def __init__(self, dimension, n_samples=100, use_prior=True):
        """Initialize sampling experiment.
        
        Args:
            dimension (int): Dimension for input vector
            n_samples (int): Number of samples per configuration
            use_prior (bool): Whether to use function-specific prior
        """
        save_dir = "sampler-with-prior" if use_prior else "sampler-no-prior"
        super().__init__(dimension, save_dir)
        self.n_samples = n_samples
        self.use_prior = use_prior
        self.target_functions = TargetFunction.get_supported_functions()
        self.samples = None
        self.summary = None

    def run_sampling(self, func_name):
        dim = self.dimension
        solution_space = SolutionSpace(dimension=dim)
        llm_sampler = LLMSampler(solution_space)
        uniform_sampler = UniformSampler(solution_space)
        target_func = TargetFunction(func_name, dimension=dim)
        
        # Generate samples
        target = func_name if self.use_prior else "unknown"
        print(f"Running LLM sampler for {func_name}...")
        llm_samples = llm_sampler.sample(self.n_samples, target_function=target)
        print(f"Running uniform sampler for {func_name}...")
        uniform_samples = uniform_sampler.sample(self.n_samples)
        
        # Evaluate samples
        eval_scores = [target_func.evaluate(sample) for sample in llm_samples]
        uniform_scores = [target_func.evaluate(sample) for sample in uniform_samples]
        
        summary = [ ["LLM",float(np.mean(eval_scores)), float(np.min(eval_scores))],
                    ["Uniform",float(np.mean(uniform_scores)), float(np.min(uniform_scores))]
                    ]
        samples = [[*x, y] for x,y in zip(llm_samples, eval_scores)]
        
        return summary, samples
        
        

        
    def run(self):
        func_name = self.target_functions[0]
        self.summary, self.samples = self.run_sampling(func_name)
        self.save_samples(func_name, self.samples)
        self.save_summary(func_name)

    def save_samples(self, func_name, sample_score_pairs):
        """Save samples and scores to CSV files."""
        dim = self.dimension
        filepath = os.path.join(
            self.saved_dir,
            f"{func_name}-dim={dim}.csv"
        )
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            row_header = [f"x{i+1}" for i in range(dim)] + ["score"]
            writer.writerow(row_header)
            for pair in sample_score_pairs:
                # pair is already a list, convert numpy types to float
                row = [float(x) for x in pair]
                writer.writerow(row)

    def save_summary(self, func_name:str):
        """Save summary results to CSV."""
        filepath = os.path.join(
            self.saved_dir,
            f'summary_{func_name}-n={self.n_samples}.csv'
        )
        headers = ['Sampler', 'Mean Score', 'Best Score']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.summary)

    def print_summary(self, summary=None):
        """Print summary table to console."""
        headers = [ 'Sampler', 'Mean Score', 'Best Score']
        if summary is None:
            summary = self.summary
        print("\nSampler Comparison Results:")
        print(tabulate(summary, headers=headers, tablefmt='grid'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sampling experiment')
    parser.add_argument('--use-prior', action='store_true', default=False,
                        help='Use function-specific prior (default: False)')
    parser.add_argument('--dimension', type=int, default=5,
                        help='Dimension for input vector (default: 5)')
    parser.add_argument('--n-samples', type=int, default=15,
                        help='Number of samples per configuration (default: 15)')
    
    args = parser.parse_args()
    
    exp = SamplingExperiment(dimension=args.dimension, 
                           n_samples=args.n_samples, 
                           use_prior=args.use_prior)
    exp.run()
    exp.print_summary()
    