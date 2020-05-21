#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:34:55 2020

Goal: Debug (if necessary) LLG equation currently being used

@author: evanraj
"""

import os
import sys
import time

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['backend'] = "Qt4Agg"
plt.rcParams['figure.max_open_warning'] = 100

l_path = '..'
m_path = os.path.abspath(l_path)
if not os.path.exists(m_path):
    print('Error importing modules. Need to specify new path')
    raise Exception
else:
    sys.path.append(m_path)

#import scipy as sp
#from scipy.sparse import csr_matrix

from Research import ferro_system1
Ferro_sys = ferro_system1.Ferro_sys

from Research import field_class
from Research.Ferro_sys_functions import sizing, Ricker_pulse, Gaussian_source, cardanos_method
from Research.running_ferro_sys_ADI_v2 import set_up_system
from scipy.linalg import norm

################ Setting up system ##################

##### System Parameters
dx = 0.4 # step size in x-direction
dy = 0.4 # step size in y-direction
dz = 0.4 # step size in z-direction
disc = np.array([dx, dy, dz])

max_x = 16.4
max_y = 16.4
max_z = dx ## 2D

gnx = np.round(max_x/dx)
gny = np.round(max_y/dy)
gnz = np.round(max_z/dz)

mu0 = 1.25667e-6
eps = 8.5422e-12
c = 1/(mu0*eps)**(1/2) ## Speed of light
gamma = 2.2e5 
K = 0
alpha = 0.2
gamma = 2.2E5
init_mag = 100.0 ## Initial magnetization in z-direction
H_s_val = 1E5 ## Static magnetic field value (uniform in x,y,z assumption)

CFL = 1/(2**(1/2))
dt = CFL*dx/c

#### Initializing system
test_sys = set_up_system(gnx, gny, gnz, disc)
test_sys.initialize_set_up_ADI()
test_sys.cubic_solver = cardanos_method
ts = test_sys

### Left Ricker Pulse forcing functions
def f_x(x,y,z,t):
    return 0
    
def f_y(x,y,z,t):
    f = 4E7
    beta0 = 4E4
    if abs(x) < disc[0]/4: #approx 0
        d = beta0*Ricker_pulse(f*t)
        return d
    else:
        return 0
       
def f_z(x,y,z,t):
    return 0

ts.fx = f_x
ts.fy = f_y
ts.fz = f_z

def reset_for_next_run():
    ts.E_old2.values = ts.E_new.values
    ts.H_old2.values = ts.H_new.values
    ts.B_old2.values = ts.B_new.values    
    
    
def LLG_run(t):
    ## Half-step parameters
    tol = ts.tol
    ticker1 = 0
    
    ################ Solving for Mn+1/2 (1) ################
    ts.B_old.values = ts.B_old2.values ### Setting up for only forward scheme LLG
    M_values_1 = ts.LLG_ADI(t, 'old')
    
    ts.M_old.values = M_values_1 
            
    ############# Updating E,B,H n+1/2 (2-3) ################
    ts.ADI_first_half(t)
    # ts.plot_line('E','y')
    
    ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (4) #######
    M_values_4 = ts.LLG_ADI(t, 'old')
    
    while ticker1 < ts.maxiter_half_step and norm(M_values_1 - M_values_4) > tol:
        ticker1 += 1
        M_values_1 = M_values_4
        ts.M_old.values = M_values_1
        ts.ADI_first_half(t)
        M_values_4 = ts.LLG_ADI(t, 'old')
        print('Current iteration for half-step fixed point:', ticker1)
        print('Current residual for half-step:', norm(M_values_1 - M_values_4))
    
    if ticker1 == ts.maxiter_half_step:
        print('**Warning. Convergence in fixed-point not reached in first half step**')
        
    ts.M_old.values = M_values_4
    
    #####################################################
    ################ Second Half step ###################
    ##################################################### 
    
    ## Full-step parameters
    ticker2 = 0
    
    ################# Solving for M_n+1 (5) #################
    ts.B_new.values = ts.B_old.values ### Setting for forward scheme LLG
    M_new_values_5 = ts.LLG_ADI(t, 'new')
    
    ts.M_new.values = M_new_values_5
    
    ############# Updating E,B,H n+1  (6-7) ################
    ts.ADI_second_half(t)
    # ts.E_old.values = ts.E_new.values
    # ts.plot_line('E','y')
    
    ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (8) #######
    M_new_values_8 = ts.LLG_ADI(t, 'new')
    
    while ticker2 < ts.maxiter_whole_step and norm(M_new_values_5 - M_new_values_8) > tol:
        ticker2+=1
        M_new_values_5 = M_new_values_8
        ts.M_new.values = M_new_values_5
        ts.ADI_first_half(t)
        M_new_values_8 = ts.LLG_ADI(t, 'new')
        print('Current iteration for whole-step fixed point:', ticker2)
        print('Current residual for whole-step:', norm(M_values_1 - M_values_4))
        
    ts.M_new.values = M_new_values_8
        
    if ticker2 == ts.maxiter_whole_step:
        print('**Warning. Convergence in fixed-point not reached in second half step**')
    
for k in range(0,50):
    t = k*dt
    ts.single_run_ADI_v2(t)
    if k%10 == 0:
        ts.plot_slice('M','z')
    reset_for_next_run()

    
# def LLG_ADI( t, X = 'old'):
#     '''
#     Uses Joly FDTD trick to solve LLG explicitly. 
    
#     If X = n, uses B_(n-1/2), B_n
    
#     Old refers to half-step, i.e. n to n+1/2 
#     New refers to whole-step, i.e. n+1/2 to n
    
#     Updates M_X
#     '''     
# ## Parameters
# dt = ts.dt/2 ### Half-step for ADI
# bdp = ts.better_dot_pdt
# gamma = ts.gamma
# K = ts.K

# M_old = ts.M_old
    
# f = 2*ts.M_old2.values
# B_on = (ts.B_old2.values + ts.B_old.values)/2

# a = -( (abs(gamma)*dt/2)*(B_on/mu0 + ts.H_s.values) + alpha*ts.M_old2.values)
# lam = -K*abs(gamma)*dt/4
    
    
# a_dot_f =  bdp(a.T,f.T).T

# ## Projection of 'easy' axis
# p_x = np.zeros(shape = (ts.M_old2.values.shape[1],1))
# p_y = np.copy(p_x)
# p_z = np.ones(shape = (ts.M_old2.values.shape[1],1))
# p = np.concatenate((p_x, p_y, p_z), axis = 1).T


# x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
# x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T

# x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    