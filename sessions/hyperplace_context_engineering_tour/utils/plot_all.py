import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os,sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from utils.target_func import TargetFunction

# Create figs directory if it doesn't exist
figs_dir = os.path.join(base_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

# Number of points for plotting
n_points = 100

# Plot each target function
for func_name in TargetFunction.get_all_functions():
    # Create target function instance
    tf = TargetFunction(func_name, dimension=2)
    
    # Create meshgrid for 3D plot
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    Z = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            Z[i,j] = tf.evaluate(np.array([X[i,j], Y[i,j]]))
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                          linewidth=0, antialiased=True)
    fig.colorbar(surf, label='Function Value')
    ax.set_title(f'{func_name.capitalize()} Function')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('f(x₁,x₂)')
    
    # Save plot
    plt.savefig(f'figs/{func_name}_3d.png')
    plt.close()
