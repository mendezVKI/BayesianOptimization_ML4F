# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 20:47:08 2022

@author: mendez
"""

# This exercise is taken from 
# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
# This is a function to plot gaussian processes
from skopt.plots import plot_gaussian_process


#%% Cost function definition
noise_level = 0.1
def f(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
           + np.random.randn() * noise_level


#%% Plot f(x) + contours
fig, ax = plt.subplots(figsize=(5, 3)) 

x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.96 * noise_level for fx_i in fx],
                         [fx_i + 1.96 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()

plt.savefig('BO_function_tutorial.png',dpi=300)
plt.show()


#%% Step 2: Fun the BO Optimization
from skopt import gp_minimize

res = gp_minimize(f,                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

# Note that the object 'res' contains all the answers!
# This is the minimum
"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)

#%% Step 3: Analyze performances
from skopt.plots import plot_convergence
# Plot Convergence function
plot_convergence(res)
# You could also show the function evaluations

#plt.plot(res.func_vals,'ko')


#%% Plot the various steps of the optimization

plt.rcParams["figure.figsize"] = (8, 14)


def f_wo_noise(x):
    return f(x, noise_level=0)


n_I=10

# Temporary Folder
import os
import imageio
FOLDER='Temp'
if not os.path.exists(FOLDER):
 os.makedirs(FOLDER) 

for n_iter in range(n_I):
    # Plot true function.
    fig= plt.figure(figsize=(10, 4)) # This creates the figure
    plt.subplot(1, 2, 1)

    if n_iter == 0:
        show_legend = True
    else:
        show_legend = False

    ax = plot_gaussian_process(res, n_calls=n_iter,
                               objective=f_wo_noise,
                               noise_level=noise_level,
                               show_legend=show_legend, show_title=False,
                               show_next_point=False, show_acq_func=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Plot EI(x)
    plt.subplot(1, 2, 2)
    ax = plot_gaussian_process(res, n_calls=n_iter,
                               show_legend=show_legend, show_title=False,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True)
    ax.set_ylabel("")
    ax.set_xlabel("")

    Name=FOLDER+'/'+'step_'+str(n_iter)+'.png'
    plt.savefig(Name,dpi=300)
    plt.show()



  
# Make a Gif
GIFNAME='BO_Optimization.gif'
images=[]    
for k in range(n_iter):
  MEX= 'Preparing Im '+ str(k)+' of ' + str(n_iter-1)
  print(MEX)
  Name=FOLDER+'/'+'step_'+str(k)+'.png'
  images.append(imageio.imread(Name))

imageio.mimsave(GIFNAME, images,duration=0.8)
import shutil  # nice and powerfull tool to delete a folder and its content
# shutil.rmtree(FOLDER)


