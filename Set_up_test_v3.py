#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:25:19 2020

@author: evanraj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:41:06 2020

Script to test new sparse matrix structure and ADI structure 
in ferro_system. Code was implemented Spring 2020 Week 3-5

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
from Research.Ferro_sys_functions import sizing, Ricker_pulse, Gaussian_source
from scipy.linalg import norm

################ Setting up system ##################

##### System Parameters
dx = 0.04 # step size in x-direction
dy = 0.04 # step size in y-direction
dz = 0.04 # step size in z-direction
disc = np.array([dx, dy, dz])

max_x = 71*dx
max_y = 71*dx
max_z = 1*dx ## 3D
# max_z = 1*dz ## 2D

gnx = np.round(max_x/dx)
gny = np.round(max_y/dy)
gnz = np.round(max_z/dz)

init_mag = 0.0 ## Initial magnetization 
H_s_val = 0.0 ## Static magnetic field value (uniform in x,y,z assumption)

c = 3e8
CFL = 1/(2**(1/2))
dt = CFL*dx/c

#### Initializing system
def set_up_system(gnx,gny,gnz,disc):
    '''
    Sets up Ferrosystem to be run. 
    '''
    def sizing(gnx,gny,gnz):
        '''
        Parameters
        ----------
        gn_ : int
            global node count in _ direction
            
        
        Returns
        -------
        list of np.arrays, length 6
            Each array gives the local node count [nx,ny,nz]
            for the outer (first 3) and inner (last 3) fields
           
        '''
        if gnz != 1:
            size_outer_x = np.array([(gnx-1)/2, (gny+1)/2, (gnz+1)/2])
            size_outer_y = np.array([(gnx+1)/2, (gny-1)/2, (gnz+1)/2])
            size_outer_z = np.array([(gnx+1)/2, (gny+1)/2, (gnz-1)/2])
            
            size_inner_x = np.array([(gnx+1)/2, (gny-1)/2, (gnz-1)/2])
            size_inner_y = np.array([(gnx-1)/2, (gny+1)/2, (gnz-1)/2])
            size_inner_z = np.array([(gnx-1)/2, (gny-1)/2, (gnz+1)/2])
            
        else:
            '''
            Note that for this case, the not-used will have the amount of 
            the field with which they are associated, simply to make concatentation
            work properly
            '''
            size_outer_x = np.array([(gnx-1)/2, (gny+1)/2, 1])
            size_outer_y = np.array([(gnx+1)/2, (gny-1)/2, 1])
            size_outer_z = np.array([(gnx+1)/2, (gny-1)/2, 1]) # This will not be included in calculations
        
            
            size_inner_x = np.array([(gnx-1)/2, (gny-1)/2, 1]) # This will not be included in calculations
            size_inner_y = np.array([(gnx-1)/2, (gny-1)/2, 1]) # This will not be included in calculations
            size_inner_z = np.array([(gnx-1)/2, (gny-1)/2, 1])
            
        return [size_outer_x, size_outer_y, size_outer_z,\
                size_inner_x, size_inner_y, size_inner_z]
    
    ### Initial conditions
    a = np.round(np.array(sizing(gnx, gny, gnz)).prod(axis=1))
    
    ####################################################
    ########### Initial conditions ################
    ####################################################
    E0_x = np.zeros(shape = (int(a[0]),1))
    E0_y = np.zeros(shape = (int(a[1]),1))
    E0_z = np.zeros(shape = (int(a[2]),1))
    E0 = np.concatenate((E0_x, E0_y, E0_z),axis=1).T
    
    # B0_x = np.zeros(shape = (int(a[3]),1))
    # B0_y = np.zeros(shape = (int(a[4]),1))
    # B0_z = np.zeros(shape = (int(a[5]),1))
    # B0 = np.concatenate((B0_x, B0_y, B0_z),axis=1).T
    
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
    node_count = np.array([gnx,gny,gnz])
    # print(node_count)
    
    R_sys = Ferro_sys(node_count,disc,E0,H0,M0,H_s)
    
    ####################################################
    ################# Run parameters ##################
    ####################################################
    R_sys.dt = dt
    R_sys.H_s_val = H_s_val
    
    ## Keeping parameters set in system, so below is not necessary
    # R_sys.mu0 = mu0
    # R_sys.eps = eps
    # R_sys.gamma = gamma
    # R_sys.K = K
    # R_sys.alpha = alpha
    
    return R_sys

test_sys = set_up_system(gnx, gny, gnz, disc)
test_sys.initialize_set_up_ADI()
ts = test_sys

tsr = set_up_system(gnx,gny,gnz,disc)
tsr.set_up_der_matrices()

tsr2 = set_up_system(gnx,gny,gnz,disc)
tsr2.set_up_der_matrices()

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

tsr.fx = f_x
tsr.fy = f_y
tsr.fz = f_z

tsr2.fx = f_x
tsr2.fy = f_y
tsr2.fz = f_z

def do_a_run(k):
    ts.ADI_first_half(dt*k)
    ts.ADI_second_half(dt*k)
    
    tsr.single_run_v2(dt*k)

def reset_for_next_run():
    ts.E_old2.values = ts.E_new.values
    ts.H_old2.values = ts.H_new.values
    ts.B_old2.values = ts.B_new.values
    
    tsr.E_old2.values = tsr.E_old.values
    tsr.H_old2.values = tsr.H_old.values
    tsr.B_old2.values = tsr.B_old.values
    
    tsr.E_old.values = tsr.E_new.values
    tsr.H_old.values = tsr.H_new.values
    tsr.B_old.values = tsr.B_new.values
    
def reset_tsr2():
    tsr2.E_old2.values = tsr.E_old.values
    tsr2.H_old2.values = tsr.H_old.values
    tsr2.B_old2.values = tsr.B_old.values
    
    tsr2.E_old.values = tsr.E_new.values
    tsr2.H_old.values = tsr.H_new.values
    tsr2.B_old.values = tsr.B_new.values
    
    
def do_ADI_run(k):
    '''
    New ADI run based on assumptions of Ricker pulse, as described by
    week 7 of spring term algorithm. 
    '''
    F_old2 = ts.Fy((k-1)*dt)
    b_ind = ts.bound_ind
    
    for l in b_ind[1]:
        ts.E_old2.y.value[l] = F_old2[l]
    
    ######### E_n+1/2 = E_n + dt/(2eps)*(-curl_R(H_n)) + F_n
    ts.E_old.values = ts.E_old2.values + dt/(2*ts.eps)*(-ts.curl_R(ts.H_old2.values, 'i'))
    F_old = ts.Fy(k*dt - dt/2)
    
    b_ind = ts.bound_ind
    
    for l in b_ind[1]:
        ts.E_old.y.value[l] = F_old[l]
    
    # ts.E_old.values += F_old
    
    ts.B_old.values = ts.B_old2.values + dt/2*(-ts.curl_L(ts.E_old2.values, 'o'))
    ts.H_old.values = ts.B_old.values/ts.mu0
    
    ## Step 2
    E_new_RHS = ts.E_old.values + dt/(2*ts.eps)*(-1/ts.mu0*ts.curl_R(ts.B_old.values, 'i'))
    ts.E_new.values = ts.step_2a_inv(E_new_RHS)
    F_new = ts.Fy(k*dt)
    
    for l in b_ind[1]:
        ts.E_new.y.value[l] = F_new[l]
    
    ts.B_new.values = ts.B_old.values + dt/2*(-ts.curl_L(ts.E_new.values,'o'))
    ts.H_new.values = ts.B_new.values/ts.mu0
    
def do_Yee_run(k):
    ### Enforcing boundary conditions at first
    b_ind = tsr.bound_ind
    F_old2 = tsr.Fy((k-1)*dt)
    for l in b_ind[1]:
        tsr.E_old2.y.value[l] = F_old2[l]
    
    ################## First half-step ###################
    tsr.E_old.values = tsr.E_old2.values + (dt/2)/tsr.eps*(-ts.curl_R(tsr.H_old2.values,'i'))
    ## Boundary conditions
    F_old = tsr.Fy(k*dt-dt/2)
    for l in b_ind[1]:
        tsr.E_old.y.value[l] = F_old[l]
    tsr.B_old.values = tsr.B_old2.values + (dt/2)*(-ts.curl_L(tsr.E_old.values, 'o'))
    tsr.H_old.values = tsr.B_old.values/tsr.mu0
    
    ################## Second half-step ###################
    tsr.E_new_values = tsr.E_old.values + dt/tsr.eps*(-ts.curl_R(tsr.H_old.values,'i'))
    ## Boundary conditions
    F_new = tsr.Fy(k*dt)
    for l in b_ind[1]:
        tsr.E_new.y.value[l] = F_new[l]
    tsr.B_new.values = tsr.B_old.values + dt*(-ts.curl_L(tsr.E_new.values,'o'))
    tsr.H_new.values = tsr.B_new.values/tsr.mu0
    
tsr.dt = dt/2
tsr2.dt = dt/2
    
for k in range(1,2):
    '''
    Here's the breakdown of what is happeneing here:
        1. ts (test_system) will be used to run the simplified ADI code above
            X_old2: time-step n
            X_old: time-step n+1/2 (referenced for plotting, so may change values 
                                    between runs)
            X_new: time-step n+1
        
        2. tsr (test_system regular) will be used to run the simplified Yee
                    scheme above, written with ADI functionality
            E_old2: time-step n
            E_old: time-step n+1/2
            E_new: time-step n+1
            
            H_old2: time-step n+1/4
            H_old: time-step n+3/4
            H_new: time-step n+5/4
            
        3. tsr2 will be used to run Yee scheme that is 'working', but at half-steps
            E_old2: time-step n
            E_old: time-step n+1/2
            E_new: time-step n+1
            
            H_old2: time-step n+1/4
            H_old: time-step n+3/4
            H_new: time-step n+5/4
    '''
    
    print('k:',k)
    do_ADI_run(k)
    do_Yee_run(k)
    tsr2.single_run_v2(dt*k-dt/2) ## Half-step
    reset_tsr2()
    tsr2.single_run_v2(dt*k) ## Whole-step
    
    ts.T = k*dt
    tsr.T = k*dt
    tsr2.T = k*dt
    
    if k%1 == 0:
          ts.E_old.values = ts.E_new.values
         # tsr.E_old.values = tsr.E_old2.values
         
          fig1,ax1 = ts.plot_line('E','y')
          fig2,ax2 = tsr.plot_line('E','y')
         
          fig12, (ax121,ax122) = plt.subplots(1,2)
          line1 = ax1.get_lines()[0]
          line2 = ax2.get_lines()[0]
         
          ax121.plot(line1.get_data()[0], line1.get_data()[1],label = 'ADI')
          ax121.plot(line2.get_data()[0], line2.get_data()[1], '--', label = 'Yee')
         
          ax122.plot(line1.get_data()[0], (line1.get_data()[1] - line2.get_data()[1])/norm(ts.E_old.y.value))
         
          ax121.set_title('Plots of ADI vs. Yee')
          ax121.legend()
          ax122.set_title('Plots of difference')
         
          fig3,ax3 = ts.plot_line('H','z')
          fig4,ax4 = tsr.plot_line('H','z')
         
          fig34, (ax341,ax342) = plt.subplots(1,2)
          line3 = ax3.get_lines()[0]
          line4 = ax4.get_lines()[0]
         
          ax341.plot(line3.get_data()[0], line3.get_data()[1], label = 'ADI')
          ax341.plot(line4.get_data()[0], line4.get_data()[1], '--', label = 'Yee')
         
          ax342.plot(line3.get_data()[0], (line3.get_data()[1] - line4.get_data()[1])/norm(ts.E_old.y.value))
         
          ax341.set_title('Plots of ADI vs. Yee')
          ax341.legend()
          ax342.set_title('Plots of difference')
         
    # reset_for_next_run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    