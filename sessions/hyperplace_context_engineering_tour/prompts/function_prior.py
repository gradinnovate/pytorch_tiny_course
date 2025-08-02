__all__ = ['FUNCTION_CHARACTERISTICS_WEAK']
# Even weaker, origin‑neutral black‑box priors for testing
FUNCTION_CHARACTERISTICS_WEAK = {
    'ackley': """
- Landscape dotted with many local dips of varying depths  
- Relatively flat “plateaus” between dip regions  
- Steep gradients descending into the deeper basins""",

    'rastrigin': """
- Regularly spaced ripples forming a grid of small valleys  
- Oscillatory pattern repeats across each dimension  
- Valley depths increase gradually toward the edges of the domain""",

    'sphere': """
- Perfectly smooth, convex “bowl” shape  
- No spurious dips—only one trough somewhere in the domain  
- Gradients always point downhill toward that single trough""",

    'rosenbrock': """
- Thin, winding valley meandering through the landscape  
- Only one trough, but it curves sharply before reaching the bottom  
- Gentle slopes outside the valley, steep sides within it""",

    'griewank': """
- Broad parabolic trend with superimposed fine ripples  
- Small, regular oscillations create many shallow minima  
- Underlying curve remains smooth despite the ripples""",

    'schwefel': """
- Global optimum lies at the boundary (~420.9687 per coordinate in [–500, 500]).
- At least one dimension should be very close to an extreme bound (±5% around 420.9687 or –500/500).
- Interior regions contain many deep, deceptive local minima—avoid focusing solely on mid‑range values.
- Emphasize edge sampling: try “corner” solutions such as [500, –500, 500, …].""",

    'michalewicz': """
- Extremely sharp, asymmetric pits forming deceptive local minima  
- Number and depth of these pits grow with higher dimensions  
- True optimum lies hidden amid many deep, misleading traps"""
}
