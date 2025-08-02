
__all__ = ['SAMPLING_USER_PROMPT', 'generate_sampling_user_prompt']

from .function_prior import FUNCTION_CHARACTERISTICS_WEAK

SAMPLING_USER_PROMPT = """Generate {n_samples} DIVERSE potential solution(s) for optimization.

Requirements:
- Each solution must be {dimension} numbers within bounds [{lower_bound}, {upper_bound}]
- Solutions should be formatted as a single JSON array containing {n_samples} sub-arrays
- Use up to 12 decimal places for precision
- Generate RANDOM and VARIED values - do NOT copy the example
- Each solution should be significantly different from others
- Consider these characteristics of the target function:
{function_characteristics}

Format: A single JSON array like [[val1, val2, ...], [val3, val4, ...], ...]

Do NOT output multiple separate arrays on different lines.
Do NOT copy example values - generate your own random numbers within bounds."""




def generate_sampling_user_prompt(solution_space, n_samples=1, target_function=None):
    """
    Generate a formatted user prompt for solution sampling.
    
    Parameters:
    -----------
    solution_space : SolutionSpace
        The solution space defining bounds and dimension
    n_samples : int
        Number of samples to request (default: 1)
    target_function : str
        Name of target function - 'ackley' or 'rastrigin' (default: 'ackley')
        
    Returns:
    --------
    str
        Formatted prompt string
    """
    # Remove example to avoid copying behavior
    example = ""
    
    if target_function is None or target_function not in FUNCTION_CHARACTERISTICS_WEAK:
        function_characteristics = "N/A"
        #print(f"Target function {target_function} not found in FUNCTION_CHARACTERISTICS_WEAK")
    else:
        function_characteristics = FUNCTION_CHARACTERISTICS_WEAK[target_function]
    
    return SAMPLING_USER_PROMPT.format(
        n_samples=n_samples,
        dimension=solution_space.dimension,
        lower_bound=solution_space.bounds[0],
        upper_bound=solution_space.bounds[1],
        function_characteristics=function_characteristics,
        example_solution=example
    )

