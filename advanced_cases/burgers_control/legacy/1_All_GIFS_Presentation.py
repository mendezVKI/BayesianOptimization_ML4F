# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:11:56 2023
@author: mendez
"""

import matplotlib.pyplot as plt
import numpy as np
from random import randint
import imageio
import time
import os
import shutil  # nice and powerfull tool to delete a folder and its content
# Optimization library
from scipy.optimize import minimize 
# The Burger environment by Fabio and Lorenzo
from Burgers.Burgers_implicit_env import Burgers_training

plt.ioff() # This is to avoid automatic plotting

#Customization of the plot 
plt.rc('text', usetex=False)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


#%% Import the environment
env = Burgers_training(name='try1',ic='fully_developed_deterministic',GAMMA=10)
# Reset the environment
obs = env.reset()
# Take any weight vector (within \pm 0.2)
w=np.array([0.2,-0.1,0.01])
# Get familiar with the environment without too much plotting!
for k in range(1500):
    action=np.dot(obs.T, w)
    obs, rew, done, _ = env.step(action)
    plt.plot(env.x,env.u)
    plt.show()


#%%% Make a Gif of the UNCONTROLLED SYSTEM

# Temporary Folder
FOLDER='Temp'
if not os.path.exists(FOLDER):
  os.makedirs(FOLDER)

# Take the initial observation
obs = env.reset()

x1=np.zeros(3) # This is now the set of weights (no action)--------
# Make a Gif 1
GIFNAME='uncontrolled'+str(x1)+'.gif'

Steps=5 # Save every 5 steps

for k in range(1500):  
  action=np.dot(obs.T, x1)
  #action = np.clip(action, -1, 1)
  obs, rew, done, _ = env.step(action)
  title_F='a(t)='+str(action) + '    time ='+str(np.round(k*env.dt,4))
  if k %Steps==0:
    fig= plt.figure(figsize=(10, 4)) # This creates the figure
    plt.plot(env.x,env.u)
    plt.plot(env.x[400],env.u[400],'ro')
    plt.plot(env.x[450],env.u[450],'ro')
    plt.plot(env.x[500],env.u[500],'ro')
    plt.plot(env.x[770],env.u[770],'go')
    plt.plot(env.x[820],env.u[820],'go')
    plt.title(title_F,size=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('u',fontsize=16)
    if np.max(np.abs(env.pert))==0:
     plt.fill(env.x,env.pert)
    else:
     plt.fill(env.x,env.pert/np.max(np.abs(env.pert))*np.abs(env.u[330]))
     
    Name=FOLDER+'\Rand_It_'+str(k//Steps)+'.png'
    print('Export '+str(k//Steps))
    plt.savefig(Name,dpi=100)
    plt.close('all')


# Prepare the video as usual
images=[]    
N_ITER=1500//Steps
for k in range(N_ITER):
  MEX= 'Preparing Im '+ str(k)+' of ' + str(N_ITER-1)
  print(MEX)
  Name=FOLDER+'\Rand_It_'+str(k)+'.png'
  images.append(imageio.imread(Name))

imageio.mimsave(GIFNAME, images,duration=0.1)
shutil.rmtree(FOLDER)


#%% RANDOM ACTION   -------------------------------------------------------------
# Temporary Folder
FOLDER='Temp'
if not os.path.exists(FOLDER):
  os.makedirs(FOLDER)

# Take the initial observation
obs = env.reset()

x1=np.array([0.5,-0.15,0.01])
# Make a Gif 1
GIFNAME='Random'+str(x1)+'.gif'

Steps=5

for k in range(1500):  
  action=np.dot(obs.T, x1)
  #action = np.clip(action, -1, 1)
  obs, rew, done, _ = env.step(action)
  title_F='a(t)='+str(action) + '    time ='+str(np.round(k*env.dt,4))
  if k %Steps==0:
    fig= plt.figure(figsize=(10, 4)) # This creates the figure
    plt.plot(env.x,env.u)
    plt.plot(env.x[400],env.u[400],'ro')
    plt.plot(env.x[450],env.u[450],'ro')
    plt.plot(env.x[500],env.u[500],'ro')
    plt.plot(env.x[770],env.u[770],'go')
    plt.plot(env.x[820],env.u[820],'go')
    plt.title(title_F,size=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('u',fontsize=16)
    # Plot the perturbation
    if np.max(np.abs(env.pert))==0:
     plt.fill(env.x,env.pert)
    else:
     plt.fill(env.x,env.pert/np.max(np.abs(env.pert))*np.abs(env.u[330]))
    # Plot the action 
    if np.max(np.abs(env.action_vec))==0:
     plt.fill(env.x,env.action_vec)
    else:
     plt.fill(env.x,env.action_vec/np.max(np.abs(env.action_vec))*np.abs(env.u[660]))
     
    Name=FOLDER+'\Rand_It_'+str(k//Steps)+'.png'
    print('Export '+str(k//Steps))
    plt.savefig(Name,dpi=100)
    plt.close('all')



images=[]    
N_ITER=1500//Steps
for k in range(N_ITER):
  MEX= 'Preparing Im '+ str(k)+' of ' + str(N_ITER-1)
  print(MEX)
  Name=FOLDER+'\Rand_It_'+str(k)+'.png'
  images.append(imageio.imread(Name))

imageio.mimsave(GIFNAME, images,duration=0.2)
import shutil  # nice and powerfull tool to delete a folder and its content
shutil.rmtree(FOLDER)

    
#%% Time frequency map without action
# This is the contour map in slide 28
x_g=env.x
t_k=np.linspace(0,15,1500)
H_XT=np.zeros((len(x_g),len(t_k)))
XG,TG=np.meshgrid(x_g,t_k)


x1=np.zeros(3)
for k in range(1500):   
  action=np.dot(obs.T, x1)
  #action = np.clip(action, -1, 1)
  obs, rew, done, _ = env.step(action)
  H_XT[:,k]=env.u


fig, ax = plt.subplots(figsize=(6, 3)) 
plt.contourf(x_g,t_k,H_XT.T,alpha=0.7)
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('t',fontsize=16)
plt.plot(env.x[400]*np.ones(len(t_k)),t_k,'k--')
plt.plot(env.x[450]*np.ones(len(t_k)),t_k,'k--')
plt.plot(env.x[500]*np.ones(len(t_k)),t_k,'k--')

# Region of evaluation
plt.plot(env.x[770]*np.ones(len(t_k)),t_k,'w-.')
plt.plot(env.x[820]*np.ones(len(t_k)),t_k,'w-.')

# Perturbation and control
plt.plot(6.6*np.ones(len(t_k)),t_k,'r-')
plt.plot(13.2*np.ones(len(t_k)),t_k,'r:')
plt.colorbar()
Name='Burger_Ch_MAPS.png'
plt.tight_layout()
plt.savefig(Name, dpi=200) 
plt.show()


# fig, ax = plt.subplots(figsize=(6, 3)) 
# plt.plot(env.x,H_XT[:,400])
# plt.plot(env.x[400],H_XT[400,400],'ro')
# plt.plot(env.x[450],H_XT[450,400],'ro')
# plt.plot(env.x[500],H_XT[500,400],'ro')
# plt.plot(env.x[770],H_XT[770,400],'go')
# plt.plot(env.x[820],H_XT[820,400],'go')


#%% Implement the BFGS optimizer ----------------------------

# Here is the cost function (to minimize in Scipy!)
def R_function(w):  ########## Cumulative reward over an episode
    # Initialize the environment
    obs = env.reset()
    # Set the learning parameters
    R_cumulative=0
    n_t=int(env.T/env.dt)
    for k in range(n_t):
        # Pick an action
        action=np.dot(w.T,obs) 
        # Run a step
        obs, rew, done, _ = env.step(action)
        # Sum up the collected rewards ( no terminal term at this stage)
        R_cumulative += rew/n_t    
    return R_cumulative



# Initial Value of the cost function
w0=np.array([0,0,0]) # Initial weights
print(R_function(w0))



# Select your method (uncomment your choice!)
#METHOD='Nelder-Mead'
#METHOD='SLSQP' 
METHOD='L-BFGS-B'

#%% Scipy optimizer
# Define boundaries
b0 = (-0.1,0.1)  # These are extreme bounds for w0
b1 = (-0.1,0.1)  # These are extreme bounds for w1
b2 = (-0.1,0.1)  # These are extreme bounds for w2
bnds = (b0, b1,b2)
# Reset the environment
obs = env.reset()
w0=np.array([0,0,0]) # Initial weights
start_time = time.time()
solution = minimize(R_function,w0,method='L-BFGS-B',\
                    bounds=bnds,\
                    options={'disp': True, 'ftol':0.000001, 'maxiter':3000})
w= solution.x # Optimal solution
print("Finished in %s s" % (time.time() - start_time))   



# Initial Value of the cost function
print(R_function(w))





# Make a Gif 1
GIFNAME='Results_'+METHOD+str(w)+'.gif'
Steps=5
# Temporary Folder
FOLDER='Temp'
if not os.path.exists(FOLDER):
  os.makedirs(FOLDER)

# Take the initial observation
obs = env.reset()

Steps=5

for k in range(1500):  
  action=np.dot(obs.T, w)
  #action = np.clip(action, -1, 1)
  obs, rew, done, _ = env.step(action)
  title_F='a(t)='+str(action) + '    time ='+str(np.round(k*env.dt,4))
  if k %Steps==0:
    fig= plt.figure(figsize=(10, 4)) # This creates the figure
    plt.plot(env.x,env.u)
    plt.plot(env.x[400],env.u[400],'ro')
    plt.plot(env.x[450],env.u[450],'ro')
    plt.plot(env.x[500],env.u[500],'ro')
    plt.plot(env.x[770],env.u[770],'go')
    plt.plot(env.x[820],env.u[820],'go')
    plt.title(title_F,size=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('u',fontsize=16)
    # Plot the perturbation
    if np.max(np.abs(env.pert))==0:
     plt.fill(env.x,env.pert)
    else:
     plt.fill(env.x,env.pert/np.max(np.abs(env.pert))*np.abs(env.u[330]))
    # Plot the action 
    if np.max(np.abs(env.action_vec))==0:
     plt.fill(env.x,env.action_vec)
    else:
     plt.fill(env.x,env.action_vec/np.max(np.abs(env.action_vec))*np.abs(env.u[660]))
     
    Name=FOLDER+'\Rand_It_'+str(k//Steps)+'.png'
    print('Export '+str(k//Steps))
    plt.savefig(Name,dpi=100)
    plt.close('all')



images=[]    
N_ITER=1500//Steps
for k in range(N_ITER):
  MEX= 'Preparing Im '+ str(k)+' of ' + str(N_ITER-1)
  print(MEX)
  Name=FOLDER+'\Rand_It_'+str(k)+'.png'
  images.append(imageio.imread(Name))

imageio.mimsave(GIFNAME, images,duration=0.2)
import shutil  # nice and powerfull tool to delete a folder and its content
shutil.rmtree(FOLDER)


#%% 2 Testing with BO

# This is an exploratory part; not essential. 
# Uncomment if we have time to play.

# #% BO STEP 1: We get 

# # Population
# from scipy.stats.qmc import LatinHypercube

# n_train=1000
# engine = LatinHypercube(d=3)
# sample_w = engine.random(n=n_train)*0.2-0.1
# R_cums=np.zeros(n_train)

# # Now we sample the reward function
# for j in range(n_train):
#     W=sample_w[j,:]
#     R_cums[j]=R_function(W)
#     print('Training Episode '+str(j))    


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(sample_w[:,0], sample_w[:,1], sample_w[:,2], marker='o',c=R_cums)

# # Use custom kernel and estimator to match previous example

# def Cost_nu_l(X):
#   L=X[0]; NU=X[1];   
#   m52 = ConstantKernel(1.0) * Matern(length_scale=L, nu=NU) #
#   gpr = GaussianProcessRegressor(kernel=m52)
#   # Reuse training data from previous 1D example
#   gpr.fit(sample_w, R_cums)
#   # Compute posterior mean and covariance
#   mu_s, cov_s = gpr.predict(sample_w, return_cov=True)
#   # Then then the error
#   E_2=np.linalg.norm(mu_s-R_cums)*1e10
#   return E_2


# X0=[0.1,2.5]
# bnds=[(0,0.5),(0.5,3)]
# start_time = time.time()
# solution = minimize(Cost_nu_l,X0,method=METHOD,\
#                     bounds=bnds,options={'disp': True, 'ftol':0.001, 'maxiter':3000})
# X= solution.x
# print("Finished in %s s" % (time.time() - start_time))   


#%% Set up the optimization in BO

# Redefine for GPyOpt 
def R_function(w):
    # Initialize the environment
    obs = env.reset()
    # Set the learning parameters
    R_cumulative=0
    n_t=int(env.T/env.dt)
    for k in range(n_t):
       # Pick an action
       action=w[0]*obs[0]+w[1]*obs[1]+w[2]*obs[2]
       # Run a step
       obs, rew, done, _ = env.step(action)
       # Sum up the collected rewards
       R_cumulative += rew    
    return R_cumulative

# Redefine function to be compatible with gp_minimize
#%% BO Control implementation
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
# Define the kernel for the Gaussian Process Regressor
m52 = ConstantKernel(1.0) * Matern(length_scale=0.1, nu=2.5) #
gpr = GaussianProcessRegressor(kernel=m52)
NN = 0.1 # Boundary of the search space
r = gp_minimize(R_function,
                [(-NN, NN),(-NN, NN),(-NN, NN)],
                base_estimator=gpr,
                acq_func='EI',      # expected improvement
                xi=0.01,            # exploitation-exploration trade-off 
                n_calls=20,
                verbose=True,
                random_state = randint(0, 1000)) #n_restarts_optimizer=100

w_opt=r.x
print(w_opt)
print(R_function(w_opt))


from skopt.plots import plot_convergence

plot_convergence(r)


#plot_convergence(np.array(r.x_iters), -r.func_vals)

#%%%%%%%%%%%%%%%% TESTING WITH GPy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import GPy
# import GPyOpt

# from GPyOpt.methods import BayesianOptimization

# NN=1
# bounds=(-NN, NN),(-NN, NN),(-NN, NN)

# #kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
# bounds =[{'name': 'w0', 'type': 'continuous', 'domain': (-NN,NN)},
#          {'name': 'w1', 'type': 'continuous', 'domain': (-NN,NN)},
#          {'name': 'w2', 'type': 'continuous', 'domain': (-NN,NN)}]

# # Redefine for GPyOpt 
# def R_function(w):
#     # Initialize the environment
#     obs = env.reset()
#     # Set the learning parameters
#     R_cumulative=0
#     n_t=int(env.T/env.dt)
#     for k in range(n_t):
#        # Pick an action
#        action=w[0]*obs[0]+w[1]*obs[1]+w[2]*obs[2]
#        # Run a step
#        obs, rew, done, _ = env.step(action)
#        # Sum up the collected rewards
#        R_cumulative += rew    
#     return R_cumulative

# optimizer = BayesianOptimization(f=R_function, 
#                                  domain=bounds,
#                                  model_type='GP',# kernel=kernel,
#                                  acquisition_type ='EI',
#                                  normalize_X=True)

# optimizer.run_optimization(max_iter=100)
# optimizer.plot_convergence()

# # The best solution via BO is 

# w=optimizer.X[np.argmin(optimizer.Y)]

# # Test via BO

METHOD='BO'
# Make a Gif 1
GIFNAME='Results_'+METHOD+'.gif'
Steps=5

# Temporary Folder
FOLDER='Temp'
if not os.path.exists(FOLDER):
  os.makedirs(FOLDER)

# Take the initial observation
obs = env.reset()

Steps=5

for k in range(1500):  
  action=np.dot(obs.T, w_opt)
  #action = np.clip(action, -1, 1)
  obs, rew, done, _ = env.step(action)
  title_F='a(t)='+str(action) + '    time ='+str(np.round(k*env.dt,4))
  if k %Steps==0:
    fig= plt.figure(figsize=(10, 4)) # This creates the figure
    plt.plot(env.x,env.u)
    plt.plot(env.x[400],env.u[400],'ro')
    plt.plot(env.x[450],env.u[450],'ro')
    plt.plot(env.x[500],env.u[500],'ro')
    plt.plot(env.x[770],env.u[770],'go')
    plt.plot(env.x[820],env.u[820],'go')
    plt.title(title_F,size=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('u',fontsize=16)
    # Plot the perturbation
    if np.max(np.abs(env.pert))==0:
     plt.fill(env.x,env.pert)
    else:
     plt.fill(env.x,env.pert/np.max(np.abs(env.pert))*np.abs(env.u[330]))
    # Plot the action 
    if np.max(np.abs(env.action_vec))==0:
     plt.fill(env.x,env.action_vec)
    else:
     plt.fill(env.x,env.action_vec/np.max(np.abs(env.action_vec))*np.abs(env.u[660]))
     
    Name=FOLDER+'\Rand_It_'+str(k//Steps)+'.png'
    print('Export '+str(k//Steps))
    plt.savefig(Name,dpi=100)
    plt.close('all')



images=[]    
N_ITER=1500//Steps
for k in range(N_ITER):
  MEX= 'Preparing Im '+ str(k)+' of ' + str(N_ITER-1)
  print(MEX)
  Name=FOLDER+'\Rand_It_'+str(k)+'.png'
  images.append(imageio.imread(Name))

imageio.mimsave(GIFNAME, images,duration=0.2)
import shutil  # nice and powerfull tool to delete a folder and its content
shutil.rmtree(FOLDER)

