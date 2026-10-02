# Burgers Equation environment

## General task overview

In this environment, the agent task is to control a wave. In particular, the Burgers equation reads as:  
$$
u_t + u u_x = \nu u_xx
$$
with Neumann homogenous boundary conditions: 

$ du/dn(x=0,t) = du/dn(x=L,t) $


This is a non linear case with a viscosity term included for stability purposes.
An external perturbation is applied as: \
$f(x,t) = A sin(2 \pi f t) * exp(-(x-x_f)^2/(2*sigma^2))$


![Burgers_render](img.png)


Hence, the task of the agent is to dump such perturbation. The final formulation of the environment is: 

$u_t + u u_x = f(x,t) + a(x,t)$

where **a(x,t)** is the action applied by the agent, normalized and modulated by a gaussian.

## The reward function

In this environment, we find that a 2-norm was not effective (too many steps were required to convergence).
Hence, a Gaussian negative reward is implemented as follows:\

$r_t = np.exp(-np.power(|u|) / (2 * sigma)) - 1$

evaluated after each control action. Finally note that this reward is 
changed of sign, yielding to a negative value: the agent will maximise 
such a signal. \
A bound is imposed on the displacement field. If the displacement goes beyond that range, then a negative reward
of -1e5 is assigned to the agent and the episode is ended (done==True, like terminal in tensorforce).

![Comparison between different rewards](img_1.png)

## The observation space

$ O = {36,)$ \
where the 36 values are 18 values of the displacement for the actual time step and the three
befor that: e.g.: $u(t), u(t-1), u(t-2)$. Now the observations are placed both before and after the control action.
The spatial location of the points are such that they're sufficiently after
the external perturbation to let it be fully developed. 

## Performances overview

With such a setting, the evaluated performances are the following: 

![img_2.png](img_2.png)

Training performances in Burger environment. The simulation is stopped automatically
(see `custom_callback.py`) when a plateau is reached. It took roughly 24M of timesteps for achieve good performance
and then the control seems to deteriorate until the simulation is stopped at 36M timesteps.
The exit condition on the collected reward was:
`var < self.treshold:`

with treshold = 0.15 in this specific case.

## Solver

In order to avoid unstabilities due to the time step and the CFL condition, 
an implicit method (Crank-Nicolson) is implemented and then solved with SciPy 
solve_banded (that has been proved to have a good performance between the methods tested).

