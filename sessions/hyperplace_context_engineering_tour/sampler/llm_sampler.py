__all__ = ['LLMSampler']

from sampler.base_sampler import BaseSampler
from llm_wrapper.llm_chat import LLMChat
from prompts.sample_system import SAMPLING_SYSTEM_PROMPT
from prompts.sample_user import generate_sampling_user_prompt
import numpy as np

class LLMSampler(BaseSampler):
    """
    Uses LLM to generate candidate solutions for optimization problems.
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
        self.llm = LLMChat(system_prompt=SAMPLING_SYSTEM_PROMPT)
        
    def sample(self, n_samples=1, target_function='ackley'):
        """
        Generate samples using LLM based on the target function.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate (default: 1)
        target_function : str
            Name of the target function ('ackley' or 'rastrigin') (default: 'ackley')
            
        Returns:
        --------
        list
            List of numpy array samples from the solution space
        """
        batch_size = 5
        all_samples = []
        
        # Process samples in batches
        for i in range(0, n_samples, batch_size):
            # Calculate batch samples needed
            batch_n = min(batch_size, n_samples - i)
            
            # Generate the user prompt for this batch
            prompt = generate_sampling_user_prompt(
                self.solution_space,
                n_samples=batch_n,
                target_function=target_function
            )
            
            # Get response from LLM
            response = self.llm.send_message(prompt, temperature=0.1)
            
            try:
                # Parse JSON response into list of samples
                import json
                import re
                
                # Remove JSON markdown wrapping robustly
                clean_response = response.strip()
                # Remove ```json or ``` at start and end using regex
                clean_response = re.sub(r'^```(?:json)?\s*', '', clean_response)
                clean_response = re.sub(r'\s*```\s*$', '', clean_response)
                clean_response = clean_response.strip()
                
                
                
                samples = json.loads(clean_response)
                
                # Validate samples and convert to numpy arrays
                if isinstance(samples[0], list):
                    # Multiple samples returned
                    valid_samples = [np.array(s) for s in samples if len(s) == self.solution_space.dimension]
                    all_samples.extend(valid_samples)
                else:
                    # Single sample returned
                    if len(samples) == self.solution_space.dimensions:
                        all_samples.append(np.array(samples))
                    
            except (json.JSONDecodeError, IndexError, TypeError):
                print(f"Error parsing response: {response}")
                continue
                
        return all_samples
