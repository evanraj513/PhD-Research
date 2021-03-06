#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:03:23 2020

Used to run ferro_system1.single_run_v2, i.e. the Yee scheme

@author: evanraj
"""

import os
import sys
from datetime import date
import time


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['backend'] = "Qt4Agg"
from collections import OrderedDict
import pandas as pd

l_path = '..'
m_path = os.path.abspath(l_path)
if not os.path.exists(m_path):
    print('Error importing modules. Need to specify new path')
    raise Exception
else:
    sys.path.append(m_path)

from Research import ferro_system1
Ferro_sys = ferro_system1.Ferro_sys

from Research.Ferro_sys_functions import sizing, Ricker_pulse, Gaussian_source, mkdir_p

t0 = time.time()

########################################################################
######################## Run settings ########################
########################################################################

cont = True ## Continue with time-run. Else, just set up system.

disp = False ## Display a run
ho_disp = 5 ## How often to display the run

def disp_what():
    '''
    What to display if disp == true
    '''
    # fig_Mz = R_sys.plot_slice('M','z',0)
    # fig_Ez = R_sys.plot_slice('E','x',0)
    # fig_Ey = R_sys.plot_slice('E','y',0)
    
    # fig_Ey0 = R_sys.plot_line('E','y',0,0)
    # fig_Mz0 = R_sys.plot_line('M','z',0,0)
    # fig_Ey1,ax_Ey1 = R_sys.plot_slice('E','y',0)
    # fig_Ey2 = R_sys.plot_line('E','y',100,0)
    fig_Ey3,ax_Ey3 = R_sys.plot_line('E','y',3,0)
    # ax_Ey1.view_init(elev = 90,azim = 90)
    
    # fig = plt.figure()
    # t_val = np.arange(0,T,dt)
    # Ricker_vec = np.vectorize(f_y)
    # ax = fig.add_subplot(111)
    # ax.plot(t_val,Ricker_vec(0,0,0,t_val))
    # ax.axvline(x=t)
    # Title = 'Current Boundary value: '+str(round(f_y(0,0,0,t),2))
    # ax.set_title(Title)
    
    # print(Ricker_pulse(t))

hold_on = 0 ## Pause the run or not. BE SURE THIS IS OFF IF DOING REMOTE
ho_hold = 1 ## How often to hold

save_time_steps = True ## Turn on to ho time steps from run
save_final_time = False ## Turn on to save final time step 
ho_save = 50 #How often to save

today1 = date.today()
name_date = today1.strftime("%d_%m_%y")
mkdir_p(name_date)
name_data ='Ricker_Yee_free_space'

######################## Parameters (global) ########################
mu0 = 1.25667e-6
eps = 8.5422e-12
c = 1/(mu0*eps)**(1/2) ## Speed of light
gamma = 2.2e5 
K = 0
alpha = 0.2
H_s_val = 10**5 ## H_s value
init_mag = 0 ## Magnetization initialization constant for M_z, 0 => free-space (non-LLG)

################## Parameters (system) ########################
dx = 0.04
dy = dx
dz = dx
disc = np.array([dx, dy, dz]) ### (dx, dy, dz)
max_x = 15

CFL = 1/(2**(1/2)) ### Testing. Soon this will be increased                  
dt = CFL*disc[0]/c
T = 501*dt ## Final time
# T = np.round(T,np.int(abs(np.log(dt)/np.log(10))))

## Making sure we have an odd-number for global nodes, otherwise
## the grid will not be uniform for Yee scheme
if np.round(max_x/disc[0])%2 == 0:
    print('Warning. Even grid. Extending spatial domain to','\n',\
          'have an odd-number of grid points in all direction')
    gnx = round((max_x+disc[0])/disc[0])
    
elif np.round(max_x/disc[0])%2 == 1:
    gnx = round(max_x/disc[0])
    
# else:
#     print('That domain and disc not available',
#           '\n','Attempting closest available disc.')
#     max_x = max_x - (max_x/disc[0])%2
#     gnx = round(max_x/disc[0])

########################### System parameters (contd) ########################   
gny = gnx
# gnz = gnx
gnz = 1 ### 2D Implementation

###################################################
### Boundary conditions, and Forcing terms ########
###################################################

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

### Huang conditions
# def f_x(x,y,z,t):
#     return 0
    
# def f_y(x,y,z,t):
    
#     return 0
       
# def f_z(x,y,z,t):
#     return 0

### Centered Gaussian source
# def f_x(x,y,z,t):
#      return 0
   
# def f_y(x,y,z,t):
    
#      if t < 42*dt:
#          # if max_x/2 - 2*dx < x < max_x/2 and max_x/2 - dy < y < max_x/2 + dy:
#              # print(x,y)
#          if max_x/2 - 2*dx < x < max_x/2 and max_x/2 - dy < y < max_x/2 + dy: #approx in the middle
#              val = Gaussian_source(dt,t)
#              # print(val)
#              return val
#          else:
#              return 0
#      else:
#          return 0
       
# def f_z(x,y,z,t):
#      return 0

########################### Initializing system ########################

def set_up_system(gnx,gny,gnz,disc):
    '''
    Sets up Ferrosystem to be run. 
    '''
    
    ### Initial conditions
    node_count = np.array([gnx, gny, gnz])
    a = np.round(np.array(sizing(node_count[0], node_count[1], node_count[2])).prod(axis=1))
    
    ####################################################
    ########### Initial conditions ################
    ####################################################
    E0_x = np.zeros(shape = (int(a[0]),1))
    E0_y = np.zeros(shape = (int(a[1]),1))
    E0_z = np.zeros(shape = (int(a[2]),1))
    E0 = np.concatenate((E0_x, E0_y, E0_z),axis=1).T
    
    M0_x = np.zeros(shape = (int(a[3]),1))
    M0_y = np.zeros(shape = (int(a[4]),1))
    M0_z = init_mag*np.ones(shape = (int(a[5]),1))
    M0 = np.concatenate((M0_x, M0_y, M0_z),axis=1).T
    
    H0_x = np.zeros(shape = (int(a[3]),1))
    H0_y = np.zeros(shape = (int(a[4]),1))
    H0_z = np.zeros(shape = (int(a[5]),1))
    H0 = np.concatenate((H0_x, H0_y, H0_z),axis=1).T
    
    H_s_x = H_s_val*np.ones(shape = (int(a[3]),1))
    H_s_y = H_s_val*np.ones(shape = (int(a[4]),1))
    H_s_z = H_s_val*np.ones(shape = (int(a[5]),1))
    H_s = np.concatenate((H_s_x, H_s_y, H_s_z),axis=1).T
    
    R_sys = Ferro_sys(node_count,disc,E0,H0,M0,H_s)
    
    ####################################################
    ################# Run parameters ##################
    ####################################################
    R_sys.dt = dt
    R_sys.mu0 = mu0
    R_sys.eps = eps
    R_sys.gamma = gamma
    R_sys.K = K
    R_sys.alpha = alpha
    R_sys.H_s_val = H_s_val
    
    return R_sys

#################################################
########### Actually running system #############

R_sys = set_up_system(gnx,gny,gnz,disc)

R_sys.fx = f_x
R_sys.fy = f_y
R_sys.fz = f_z

ticker = 0

R_sys.set_up_der_matrices() ### Sets up matrices and operators for ADI method

for t in np.arange(dt,T,dt):
    ### re-initializing the system
    R_sys.set_up()
    ticker += 1
    if cont == True:
        pass
    else:
        break
    ## For output
    print('Current Time: ',np.round(t,np.int(abs(np.log(dt)/np.log(10)))+2),' and Ticker: ',ticker)

    ### Running the system
    R_sys.T = t
    R_sys.single_run_v2(t)
    
    ## Plotting stuff to demo
    if disp == True:
        # print('Display?')
        if ticker%ho_disp < 1E-12:
            # print('yes!')
            disp_what()
            
    if hold_on == True:
        if ticker%ho_hold < 1e-12:
            wait = input('Press Enter to continue')
        
    if save_time_steps == True:
        if ticker%ho_save < 1E-12:
            name_time = 'ticker:'+str(ticker)
            print('Saving current conditions at time:',name_time)
            mkdir_p(name_date+'/'+name_time) #Creates sub-directory for this specific time
            R_sys.save_data(name_date+'/'+name_time+'/'+name_data) #Saves Field data
            ### Note: parameters are now saved outside the loop
            
    ## Updating X_n-1 = X_n, X_n = X_n+1 for next time-step
    R_sys.E_old2.values = R_sys.E_old.values
    R_sys.B_old2.values = R_sys.B_old.values
    R_sys.M_old2.values = R_sys.M_old.values
    R_sys.H_old2.values = R_sys.H_old.values
    
    R_sys.E_old.values = R_sys.E_new.values
    R_sys.B_old.values = R_sys.B_new.values
    R_sys.M_old.values = R_sys.M_new.values
    R_sys.H_old.values = R_sys.H_new.values

t1 = time.time()
print('Time taken:', t1-t0)

## Saving data
if save_final_time == True:
    mkdir_p(name_date+'/'+'time:'+str(round(T-dt,12)))
    R_sys.save_data(name_date+'/'+'Final_time'+'/'+name_data) ## Names defined above
    
# Run parameters
    # Global Parameters
if save_time_steps == True:
    data = OrderedDict()
    data['mu0'] = mu0
    data['eps'] = eps 
    data['gamma'] = gamma
    data['K'] = K
    data['alpha'] = alpha
    data['H_s'] = H_s_val
    
            # System Parameters
    data['max_x'] = max_x
    data['disc_x'] = disc[0]
    data['disc_y'] = disc[1]
    data['disc_z'] = disc[2]
    data['gnx'] = gnx
    data['gny'] = gny
    data['gnz'] = gnz
    data['T'] = T
    data['dt'] = dt
    data['CFL'] = c*dt/disc[0]
    data['Time tiken'] = round(t1-t0,2)
    
    df = pd.DataFrame([data])
    df.to_csv(name_date+'/'+name_data+'_param.csv')

