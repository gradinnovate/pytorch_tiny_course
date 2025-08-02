import numpy as np
from utils.target_func import TargetFunction
from utils.solution_space import SolutionSpace
from llm_wrapper.llm_icl import LLMLearner
from sampler.uniform_sampler import UniformSampler
from base_experiment import BaseExperiment
from tabulate import tabulate
import csv
import os
from tqdm import tqdm
import random

    
class ICLExperiment(BaseExperiment):
    def __init__(self, dimension, context_length, k_samples=1, use_prior=True):
        """Initialize ICL experiment.
        
        Args:
            dimension (int): dimension of input vector
            context_length (int): Number of parameter-value pairs to use for in-context learning
            k_samples (int): Number of samples to generate per iteration (default: 1)
            use_prior (bool): Whether to use function-specific prior (default: True)
        """
        save_prefix = f"learner"
        
       
        if use_prior:
            save_prefix += "-with-prior"  
        else:
            save_prefix += "-no-prior"
        super().__init__(dimension, save_prefix)
        self.context_length = context_length
        self.k_samples = k_samples
        self.use_prior = use_prior
        self.target_functions = TargetFunction.get_supported_functions()
        self.num_runs = 20
        self.dimension = dimension
        self.summary = None
        self.samples = None
  
    
    def run_learning(self, func_name, ordering:str='ascending'):
        
        dim = self.dimension
        
        solution_space = SolutionSpace(dimension=dim)
        uniform_sampler = UniformSampler(solution_space)
        target_func = TargetFunction(func_name, dimension=dim)
        solutions = uniform_sampler.sample(self.context_length)
        scores = [target_func.evaluate(sol) for sol in solutions]
        total_examples = [(x,y) for x,y in zip(solutions, scores)] 
        
        
        llm_learner = LLMLearner(solution_space)
        hist_eval_scores = []
        results = []
        pbar = tqdm(range(self.num_runs), desc=f"Running {func_name} with {ordering} ordering")
     
        for _ in pbar:
            llm_learner.llm.clear_history()
            if ordering == 'ascending':
                ordered_examples = sorted(total_examples, key=lambda x: x[1])[:self.context_length]
            elif ordering == 'descending':
                ordered_examples = sorted(total_examples, key=lambda x: x[1], reverse=True)[-self.context_length:]
            else:
                ordered_examples = random.sample(total_examples, self.context_length)
            
            target = func_name if self.use_prior else "unknown"
            
            llm_samples = llm_learner.sample_with_examples(
                    self.k_samples,
                    target_function=target,
                    examples=ordered_examples
                )    
            eval_scores = [target_func.evaluate(sample) for sample in llm_samples]
            results.extend([[*x, y] for x,y in zip(llm_samples, eval_scores)])
            hist_eval_scores.extend(eval_scores)
            
            new_examples = [(x,y) for x,y in zip(llm_samples, eval_scores)]
            total_examples.extend(new_examples)
            pbar.update(1)
        pbar.close()
            
        baseline_samples = uniform_sampler.sample(self.context_length)
        baseline_scores = [target_func.evaluate(sample) for sample in baseline_samples]
        
        
        summary = [
            f"{ordering}",
            float(np.mean(hist_eval_scores)),
            float(np.min(hist_eval_scores)),
            float(np.mean(baseline_scores)),
            float(np.min(baseline_scores))
        ]
        return summary, results
        
    
   
    def run(self):
        func_name = self.target_functions[0]
        self.summary, self.samples = self.run_learning(func_name, ordering='ascending')
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
            f'summary_{func_name}-n={self.context_length}_k={self.k_samples}.csv'
        )
        headers = ['Ordering', 'LLM Mean Score', 'LLM Best Score', 'Baseline Mean Score', 'Baseline Best Score']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(self.summary)

    def print_summary(self, summary=None):
        """Print summary table to console."""
        headers = ['Ordering', 'LLM Mean Score', 'LLM Best Score', 'Baseline Mean Score', 'Baseline Best Score']
        print("\nICL Experiment Results:")
        if summary is None:
            summary = self.summary
        # summary is a single row, wrap it in a list for tabulate
        print(tabulate([summary], headers=headers, tablefmt='grid'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sampling experiment')
    parser.add_argument('--use-prior', action='store_true', default=False,
                        help='Use function-specific priors (default: False)')
    parser.add_argument('--dimension', type=int, default=5,
                        help='Dimensions to test (default: 5)')
    parser.add_argument('--k-samples', type=int, default=5,
                        help='Number of samples per configuration (default: 5)')
    
    args = parser.parse_args()
    np.random.seed(42)
    
    context_length = 30
    exp = ICLExperiment(dimension=args.dimension, context_length=context_length,
                        k_samples=args.k_samples, use_prior=args.use_prior)
    exp.run()
    exp.print_summary()