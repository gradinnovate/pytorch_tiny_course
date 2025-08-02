from sampler.base_sampler import BaseSampler
from llm_wrapper.llm_chat import LLMChat
from prompts.accqisition_system import ACQUISITION_SYSTEM_PROMPT
from prompts.accquisition_user import generate_acquisition_user_prompt
import numpy as np
import json

class LLMLearner(BaseSampler):
    """
    Uses LLM with in-context learning to generate candidate solutions.
    """
    def __init__(self, solution_space):
        """
        Initialize learner with a solution space.
        
        Parameters:
        -----------
        solution_space : SolutionSpace
            The solution space to sample from
        """
        super().__init__(solution_space)
        self.llm = LLMChat(system_prompt=ACQUISITION_SYSTEM_PROMPT)
    
    def sample(self, n_samples=1):
        pass  
    def sample_with_examples(self, k_samples=5, target_function=None, examples=None):
        """
        Generate samples using LLM based on example solutions and scores.
        
        Parameters:
        -----------
        k_samples : int
            Number of samples to generate (default: 5)
        target_function : str
            Name of the target function for characteristics lookup (default: None)
        examples : str
            Formatted string of example solution-score pairs
            
        Returns:
        --------
        list
            List of numpy array samples from the solution space
        """
        batch_size = 5
        all_samples = []
        
        # Process samples in batches
        for i in range(0, k_samples, batch_size):
            # Calculate batch samples needed
            batch_n = min(batch_size, k_samples - i)
            
            # Generate the user prompt for this batch
            prompt = generate_acquisition_user_prompt(
                self.solution_space,
                examples=examples,
                k_samples=batch_n,
                target_function=target_function
            )
            
            
            # Get response from LLM
            response = self.llm.send_message(prompt, temperature=0.1)
            
            for _ in range(3):
                try:
                    # Parse JSON response into numpy arrays
                    samples = json.loads(response)
                    batch_samples = [np.array(sample) for sample in samples]
                    all_samples.extend(batch_samples)
                    break
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, generate random samples as fallback
                    print(f"Parsing failed, response: {response}")
                    continue
        return all_samples[:k_samples]
