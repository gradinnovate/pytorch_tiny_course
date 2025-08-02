__all__ = ['ACQUISITION_USER_PROMPT', 'generate_acquisition_user_prompt']
from .function_prior import FUNCTION_CHARACTERISTICS_WEAK

ACQUISITION_USER_PROMPT = """Here are {n_examples} example pairs of solution→score:
{examples}

Current best score: {current_best_score}

Generate {k_samples} new candidate vectors within bounds [{lower_bound}, {upper_bound}] to minimize function value.

CRITICAL: Output ONLY a JSON array of {k_samples} vectors. NO explanations, NO markdown, NO code blocks.

Format: [[-1.23456789, 0.87654321, ...], [2.34567890, -1.11111111, ...], ...]"""

def generate_acquisition_user_prompt(solution_space, examples, k_samples=5, target_function=None):
    """
    Generate the user prompt for acquisition sampling.
    
    Parameters:
    -----------
    solution_space : SolutionSpace
        The solution space to sample from
    examples : list
        List of (solution, score) tuples to use as examples
    k_examples : int
        Number of examples to include in prompt (default: 5)
    target_function : str
        Name of target function for characteristics lookup (default: None)
        
    Returns:
    --------
    str
        Formatted prompt string
    """
    
    # Format examples as JSON strings
    n_examples = len(examples)
    example_str = ""
    for solution, score in examples:
        solution_json = [f"{x:.8f}" for x in solution]
        example_str += f"[{', '.join(solution_json)}] → {score}\n"
    
    current_best_score = min(score for _, score in examples)
    # Get function characteristics if target_function provided
    function_characteristics = "N/A"
    if target_function and target_function in FUNCTION_CHARACTERISTICS_WEAK:
        function_characteristics = FUNCTION_CHARACTERISTICS_WEAK[target_function]
    
    # Format the prompt with all parameters
    prompt = ACQUISITION_USER_PROMPT.format(
        lower_bound=solution_space.bounds[0],
        upper_bound=solution_space.bounds[1],
        function_characteristics=function_characteristics,
        n_examples=n_examples,
        examples=example_str,
        k_samples=k_samples,  # Always request 1 sample at a time
        current_best_score=current_best_score
    )

    return prompt