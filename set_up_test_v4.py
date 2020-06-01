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

#from Research import field_class
#from Research.Ferro_sys_functions import sizing, Ricker_pulse, Gaussian_source, cardanos_method, round_to_3
#from Research.running_ferro_sys_ADI_v2 import set_up_system

from Research import Ferro_sys_functions as fsf


from scipy.linalg import norm

################ Setting up system ##################

##### System Parameters
dx = 0.2 # step size in x-direction
dy = 0.2 # step size in y-direction
dz = 0.2 # step size in z-direction
disc = np.array([dx, dy, dz])

max_x = 10.2
max_y = max_x
max_z = dz ## 2D
# max_z = max_x ## 3D

gnx = np.round(max_x/dx)
gny = np.round(max_y/dy)
gnz = np.round(max_z/dz)

#### Normal parameters
mu0 = 1.25667e-6
eps = 8.5422e-12
c = 1/(mu0*eps)**(1/2) ## Speed of light
K = 0
alpha = 0.2
gamma = 2.2E5
init_mag = 100.0 ## Initial magnetization in z-direction
H_s_val = 1E5 ## Static magnetic field value (uniform in x,y,z assumption)

CFL = 1/(2**(1/2))*5
dt = CFL*dx/c

#### Initializing system
test_sys = fsf.set_up_system(gnx, gny, gnz, disc,init_mag = 100)
# test_sys.initialize_set_up_ADI()
test_sys.cubic_solver = fsf.cardanos_method
ts = test_sys ### System running code to debug
ts.dt = dt
# ts.gamma = 0.0
# ts.alpha = 0.0

# ts2 = set_up_system(gnx, gny, gnz, disc) ### Running with 1 step Joly trick
# ts2.initialize_set_up_ADI()
# ts2.cubic_solver = cardanos_method

tsr = fsf.set_up_system(gnx,gny,gnz,disc,init_mag = 100) ### test_system_regular: runs Yee to compare to new algorithms
tsr.dt = dt/2
# tsr.gamma = 0.0
# tsr.alpha = 0.0
tsr.set_up_der_matrices()
# tsr.initialize_set_up_ADI()

### Left Ricker Pulse forcing functions
def f_x(x,y,z,t):
    return 0
    
def f_y(x,y,z,t):
    f = 4E7
    beta0 = 4E4
    if abs(x) < disc[0]/4: #approx 0
        d = beta0*fsf.Ricker_pulse(f*t)
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

def reset_ts_for_next_run():
    ts.E_old2.values = ts.E_new.values
    ts.H_old2.values = ts.H_new.values
    ts.B_old2.values = ts.B_new.values  
    ts.M_old2.values = ts.M_new.values
    
def reset_tsr_for_next_run():
    # tsr.E_old2.values = tsr.E_old.values
    # tsr.H_old2.values = tsr.H_old.values
    # tsr.B_old2.values = tsr.B_old.values
    # tsr.M_old2.values = tsr.M_old.values
    
    # tsr.E_old.values = tsr.E_new.values
    # tsr.H_old.values = tsr.H_new.values
    # tsr.B_old.values = tsr.B_new.values
    # tsr.M_old.values = tsr.M_new.values
    
    tsr.E_old2.values = tsr.E_new.values
    tsr.H_old2.values = tsr.H_new.values
    tsr.B_old2.values = tsr.B_new.values
    tsr.M_old2.values = tsr.M_new.values
    
def LLG_portion(system, X = 'Old'):
    '''
    Performs the Joly trick on system. 
    
    If X == 'Old', performs trick on old2 to update old
    If X == 'New', performs trick on old to update new
    
    These two set-ups are necessary for the ADI, but Yee scheme is written
        to use it as well now. 
    '''
    
    ## Parameters
    dt = system.dt/2 ### Half-step for ADI
    bdp = system.better_dot_pdt
    gamma = system.gamma
    alpha = system.alpha
    K = system.K
    
    M_old = system.M_old
            
    if X == 'old' or X == 'Old': ## Set-up for half-step
    ## old2 = n-1/2, old = n+1/2 for Joly legibility
        
        f = 2*system.M_old2.values
        B_on = (system.B_old2.values + system.B_old.values)/2
        
        a = -( (abs(gamma)*dt/2)*(B_on/mu0 + system.H_s.values) + alpha*system.M_old2.values)
        lam = -K*abs(gamma)*dt/4
        
    elif X == 'new' or X == 'New':
        f = 2*system.M_old.values
        B_on = (system.B_old.values + system.B_new.values)/2
        
        a = -( (abs(gamma)*dt/2)*(B_on/mu0 + system.H_s.values) + alpha*system.M_old.values)
        lam = -K*abs(gamma)*dt/4
        
    a_dot_f =  bdp(a.T,f.T).T
    
    ## Projection of 'easy' axis
    p_x = np.zeros(shape = (system.M_old2.values.shape[1],1))
    p_y = np.copy(p_x)
    p_z = np.ones(shape = (system.M_old2.values.shape[1],1))
    p = np.concatenate((p_x, p_y, p_z), axis = 1).T
    
    if K == 0 or t == dt:
        x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
        x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
        
        x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
            
    else:
        
        cubic_solver = system.cubic_solver
        
        a1 = lam**2
        b1 = 2*lam*(bdp(a.T, p.T) + lam*(bdp(p.T, f.T)))
        c1 = 1+np.linalg.norm(a)**2 - lam*(bdp(a.T, f.T)) + 3*lam*\
        (bdp(a.T, p.T)) * (bdp(p.T, f.T)) + \
        lam**2*(bdp(p.T, f.T))
        d1 = -lam*(bdp(a.T, f.T)*(bdp(p.T,f.T))) - (bdp(a.T, p.T)*(bdp(p.T,f.T))**2)\
        + lam*((bdp(a.T, p.T)*(bdp(p.T,f.T))**2))\
        +np.linalg.norm(a)**2*(bdp(p.T, f.T))
        -bdp(np.cross(a.T, p.T),f.T)
        Z = np.zeros(shape = b1.shape)
        X = np.copy(Z)
        Y = np.copy(Z)
        x_new_values = np.copy(Z)
        for k in np.arange(0,x_new_values.shape[1]):
            if k%100 == 1:
                Z[k] = cubic_solver(a1,b1[k],c1[k],d1[k],M_old.x.value[k],disp = 'Yes')
            else:
                Z[k] = cubic_solver(a1,b1[k],c1[k],d1[k],M_old.x.value[k],disp = 'no')
        
        X = (bdp(a.T,f.T)) - lam*Z*(Z+bdp(p.T,f.T))
        Y = Z+bdp(p.T,f.T)
        
        x_new_values = 1/np.linalg.norm(np.cross(a.T,p.T).T)**2*\
        ((X - (bdp(a.T,p.T))*Y).T*a\
          + (((np.linalg.norm(a))**2*Y) - (bdp(a.T,p.T))).T*X\
          + (Z*np.cross(a.T, p.T)).T)
        
    if X == 'old' or X == 'Old':
        return x_new_values.T - system.M_old2.values
    elif X == 'new' or X == 'New':
        return x_new_values.T - system.M_old.values
    else:
        print('Something is wrong in LLG_portion')
    
def Yee_run(t):
    '''
    A test Yee run to (compare with ADI) run below. 
    
    Will be implemented to do two time-steps, to match counters and whatnot
    with ADI method
    
    n = old2 = t-tsr.dt*2
    n+1/2 = old = t-tsr.dt
    n+1 = new = t
    
    Note that tsr.dt = ts.dt/2. 
    '''
    ### First Half-step
    E_old2 = tsr.E_old2
    # H_old = tsr.H_old
    # M_old2 = tsr.M_old2
    B_old2 = tsr.B_old2
    
    dt = tsr.dt
    b_ind = tsr.bound_ind
    
    ## Get curl values
    tsr.H_old.values = tsr.H_old2.values ## necessary to use the next line
    tsr.set_up_H_curl()
    
    ## Boundary conditions being satisfied
    # F_old = np.concatenate((tsr.Fx(t), tsr.Fy(t), tsr.Fz(t)),axis=1) ## F_n-1/2
    # E_old.values += F_old.T
    
    '''
    Note, the above is not necessary, as we solve for E first. 
    '''
    
    ## Update E_old  
    F = np.concatenate((tsr.Fx(t-dt), tsr.Fy(t-dt), tsr.Fz(t-dt)),axis=1) ## F_n+1/2
    E_old_values = E_old2.values + dt/eps*tsr.H_old_curl ## +dt*sigma*E_old.values + dt*J
    
    #Setting all E boundaries
    for j in b_ind[0]:
        E_old_values[0][j] = F.T[0][j] #x_bound(j)
    for k in b_ind[1]:
        E_old_values[1][k] = F.T[1][k]
    for l in b_ind[2]:
        E_old_values[2][l] = F.T[2][l]
    
    tsr.E_old.values = E_old_values
    
    tsr.E_new.values = tsr.E_old.values ## Similar to H above, necessary for next line
    tsr.set_up_E_curl()
    
    ## Update B_old
    tsr.B_old.values = B_old2.values - dt*tsr.E_new_curl
    
    ## Solving for M_old
    tsr.M_old.values = LLG_portion(tsr, X = 'Old')
    
    ## Update H_new
    tsr.H_old.values = tsr.B_old.values/mu0 - tsr.M_old.values   
    
    # tsr.plot_slice('E','y')
    # tsr.plot_slice('M','z')
    # tsr.plot_slice('E','z')
    
    ### Setting TM mode to 0
    tsr.E_old.values[2] = tsr.E_old.z.value*0
    
    tsr.M_old.values[0] = tsr.M_old.x.value*0
    tsr.M_old.values[1] = tsr.M_old.y.value*0
    
    tsr.B_old.values[0] = tsr.B_old.x.value*0
    tsr.B_old.values[1] = tsr.B_old.y.value*0
    
    tsr.M_old.values[0] = tsr.M_old.x.value*0
    tsr.M_old.values[1] = tsr.M_old.y.value*0
    
    ##### Second Half-step
    E_old = tsr.E_old
    # H_old = tsr.H_old
    # M_old = tsr.M_old
    B_old = tsr.B_old
    
    ## Get curl values
    tsr.set_up_H_curl()
    
    ## Boundary conditions being satisfied
    # F_old = np.concatenate((tsr.Fx(t), tsr.Fy(t), tsr.Fz(t)),axis=1) ## F_n-1/2
    # E_old.values += F_old.T
    
    ## Update E_new
    F = np.concatenate((tsr.Fx(t), tsr.Fy(t), tsr.Fz(t)),axis=1) ## F_n+1/2
    E_new_values = E_old.values + dt/eps*tsr.H_old_curl ## +dt*sigma*E_old.values + dt*J
    
    #Setting all E boundaries
    for j in b_ind[0]:
        E_new_values[0][j] = F.T[0][j] #x_bound(j)
    for k in b_ind[1]:
        E_new_values[1][k] = F.T[1][k]
    for l in b_ind[2]:
        E_new_values[2][l] = F.T[2][l]
    
    tsr.E_new.values = E_new_values
    
    tsr.set_up_E_curl()
    
    ## Update B_new
    B_new_values = B_old.values - dt*tsr.E_new_curl
    tsr.B_new.values = B_new_values
    
    ## Solving for M_new
    tsr.M_new.values = LLG_portion(tsr,X = 'New')
    
    ## Update H_new
    tsr.H_new.values = B_new_values/mu0 - tsr.M_new.values   
    
    
    ### Setting TM mode to 0
    tsr.E_new.values[2] = tsr.E_old.z.value*0
    
    tsr.M_new.values[0] = tsr.M_old.x.value*0
    tsr.M_new.values[1] = tsr.M_old.y.value*0
    
    tsr.B_new.values[0] = tsr.B_old.x.value*0
    tsr.B_new.values[1] = tsr.B_old.y.value*0
    
    tsr.H_new.values[0] = tsr.M_old.x.value*0
    tsr.H_new.values[1] = tsr.M_old.y.value*0
    
def ADI_first_half_test(system,t):
    '''
    Runs the first ADI half-step. Uses:
        
        En = E_old2
        Bn = B_old2
        Hn = H_old2
        Mn+1/2 = M_old
    
    to compute:
        En+1/2 = E_old
        Bn+1/2 = B_old
        Hn+1/2 = H_old
    
    '''
    
    E_old2 = system.E_old2
    H_old2 = system.H_old2
    # M_old2 = system.M_old2
    B_old2 = system.B_old2
    
    E_old = system.E_old
    B_old = system.B_old
    M_old = system.M_old
    H_old = system.H_old
    
    dt = system.dt
    # bdp = system.better_dot_pdt
    
    ## Parameter choices given in system
    mu0 = system.mu0
    eps = system.eps
    sigma = system.sigma 
    
    ### Applying boundary conditions
    F_old2 = np.concatenate((system.Fx(t-dt), system.Fy(t-dt), system.Fz(t-dt)),axis=1)
    E_old2.values += F_old2.T
    
    ##### Solving for E_n+1/2
    s_a = 1/mu0*system.curl_L(B_old2.values,'Inner')
    s_b = dt/(2*mu0)*system.curl_LL(E_old2.values)
    s_c = system.curl_L(M_old.values,'Inner')
    s_d = system.curl_R(H_old2.values,'Inner')
    
    s_main = (s_a - s_b - s_c - s_d)
    J_old = np.concatenate((system.Jx(t-3*dt/4), system.Jy(t-3*dt/4), system.Jz(t-3*dt/4)),axis=1)
    
    E_old_RHS = E_old2.values + dt/(2*eps)*s_main - (dt/4)*sigma*E_old2.values + dt/2*J_old.T
    
    ### Using back-solve for new values
    E_old_values = system.step_1a_inv(E_old_RHS)
    
    # Setting all E boundaries to 0
    b_ind = system.bound_ind
    
    for j in b_ind[0]:
        E_old_values[0][j] = 0 #x_bound(j)
    for k in b_ind[1]:
        E_old_values[1][k] = 0
    for l in b_ind[2]: 
        E_old_values[2][l] = 0
        
    ### Add in forcing terms at the half-step
    F_old = np.concatenate((system.Fx(t-dt/2), system.Fy(t-dt/2), system.Fz(t-dt/2)),axis=1)
    E_old_values += F_old.T
    
    E_old.values = E_old_values
    
    # system.plot_slice()
    
    ###### Solving for B_n+1/2
    B_old_values = B_old2.values + dt/2*(\
        system.curl_R(E_old_values, 'o') - \
            system.curl_L(E_old2.values, 'o'))
        
    B_old.values = B_old_values
    
    ###### Solving for H_n+1/2
    H_old_values = 1/mu0*B_old_values - M_old.values
    H_old.values = H_old_values
    
    # system.plot_line('H','z')
    
def ADI_second_half_test(system,t):
    '''
    Runs the second half of the ADI scheme. Uses:
        Mn+1 = M_new
        Hn+1/2 = H_old
        En+1/2 = E_old
        Bn+1/2 = B_old
        
    to compute
        En+1 = E_new
        Bn+1 = B_new
        Hn+1 = H_new
        
    '''
    
    E_old = system.E_old
    B_old = system.B_old
    # M_old = system.M_old
    H_old = system.H_old
    
    E_new = system.E_new
    B_new = system.B_new
    M_new = system.M_new
    H_new = system.H_new
    
    dt = system.dt
    b_ind = system.bound_ind
    # bdp = system.better_dot_pdt
    
    ## Parameter choices given in system
    mu0 = system.mu0
    eps = system.eps
    sigma = system.sigma 
    
    s2_a = system.curl_L(H_old.values, 'i')
    s2_b = 1/mu0*system.curl_R(B_old.values, 'i')
    s2_c = dt/(2*mu0)*system.curl_RR(E_old.values)
    s2_d = system.curl_R(M_new.values,'i')
    
    s2_main = s2_a - s2_b - s2_c + s2_d
    J_new = np.concatenate((system.Jx(t-dt/4), system.Jy(t-dt/4), system.Jz(t-dt/4)),axis=1)
    
    E_new_RHS = E_old.values + dt/(2*eps)*s2_main - dt/4*sigma*E_old.values +dt/2*J_new.T
    
    # Backsolve
    
    E_new_values = system.step_2a_inv(E_new_RHS)
    
    #Setting all E boundaries to 0
    for j in b_ind[0]:
        E_new_values[0][j] = 0 #x_bound(j)
    for k in b_ind[1]:
        E_new_values[1][k] = 0
    for l in b_ind[2]:
        E_new_values[2][l] = 0
    
    # ### Forcing terms at the half-step
    
    F_new = np.concatenate((system.Fx(t), system.Fy(t), system.Fz(t)),axis=1)
    E_new_values += F_new.T
    
    E_new.values = E_new_values
    
    ### Solving for B_n+1
    B_new.values = B_old.values + dt/2*(\
                    system.curl_R(E_old.values,'o') - 
                    system.curl_L(E_new_values, 'o'))
    
    ### Solving H_n+1
    H_new.values = 1/mu0*B_new.values - M_new.values

def ADI_run(t):
    ## Half-step parameters
    tol = ts.tol
    ticker1 = 0
    
    ################ Solving for Mn+1/2 (1) ################
    ts.B_old.values = ts.B_old2.values ### Setting up for only forward scheme LLG
    M_values_1 = LLG_portion(ts, 'old')
    
    ts.M_old.values = M_values_1 
            
    ############# Updating E,B,H n+1/2 (2-3) ################
    ADI_first_half_test(ts,t)
    # ts.plot_line('E','y')
    
    ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (4) #######
    M_values_4 = LLG_portion(ts, 'old')
    
    while ticker1 < 3 and norm(M_values_1 - M_values_4) > tol:
        ticker1 += 1
        M_values_1 = M_values_4
        # print('res here:', norm(M_values_1 - M_values_4))
        ts.M_old.values = M_values_1
        ts.ADI_first_half(t)
        M_values_4 = ts.LLG_ADI(t, 'old')
        
        # ts.M_old.values = M_values_4
        # ts.plot_slice('M','z')
        
        # ts.M_old.values = M_values_1 - M_values_4
        # ts.plot_slice('M','z')
        
        # ts.M_old.values = M_values_4
        print(' ~ Current iteration for half-step fixed point:', ticker1)
        print(' ~ Current residual for half-step:', norm(M_values_1 - M_values_4))
    
    if ticker1 == ts.maxiter_half_step:
        print('** Warning. Convergence in fixed-point not reached in first half step **')
        
    ts.M_old.values = M_values_4
    
    #####################################################
    ################ Second Half step ###################
    ##################################################### 
    
    ## Full-step parameters
    ticker2 = 0
    
    ################# Solving for M_n+1 (5) #################
    ts.B_new.values = ts.B_old.values ### Setting for forward scheme LLG
    M_new_values_5 = LLG_portion(ts, 'new')
    
    ts.M_new.values = M_new_values_5
    
    ############# Updating E,B,H n+1  (6-7) ################
    ADI_second_half_test(ts,t)
    # ts.E_old.values = ts.E_new.values
    # ts.plot_line('E','y')
    
    ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (8) #######
    M_new_values_8 = LLG_portion(ts, 'new')
    
    # while ticker2 < ts.maxiter_whole_step and norm(M_new_values_5 - M_new_values_8) > tol:
    #     ticker2+=1
    #     M_new_values_5 = M_new_values_8
    #     ts.M_new.values = M_new_values_5
    #     ts.ADI_first_half(t)
    #     M_new_values_8 = ts.LLG_ADI(t, 'new')
    #     print(' + Current iteration for whole-step fixed point:', ticker2)
    #     print(' + Current residual for whole-step:', norm(M_new_values_5 - M_new_values_8))
        
    ts.M_new.values = M_new_values_8
        
    # if ticker2 == ts.maxiter_whole_step:
    #     print('**Warning. Convergence in fixed-point not reached in second half step**')


    
def plot_y_cs(F = 'E', comp = 'y', cs = 0, s = 0, direc = 'y'):
    '''
    Test function to plot cross-sections of ferro_sys in y-direction as opposed
    to x-direction. Should help with finding if and where uniformity in y is lost
    
    Transforms happen as:
              |(x_1, y_1)  |          (x_1, y_1) ... (x_nx, y_1)    
        val = |(x_1, y_2)  |              :      .        :        
              |    :       |   -->        :         .     :        
              |(x_1, y_ny) |         (x_1, y_ny) ... (x_nx, y_ny)      
              |(x_2, y_1)  | 
              |    :       |                      
              |(x_nx, y_ny)|
              
        and then from there, you can actually plot either x or y, whatever you want. 
        
    Note: The input has to be flipped with how the reshape function works. So now:
        val[i,j,k] = val(z_i, y_j, x_k). 
        
    '''
    if F == 'E':
        if comp == 'x':
            pc = ts.E_old.x
        elif comp == 'y':
            pc = ts.E_old.y
        elif comp == 'z':
            pc = ts.E_old.z
    elif F == 'M':
        if comp == 'x':
            pc = ts.M_old.x
        elif comp == 'y':
            pc = ts.M_old.y
        elif comp == 'z':
            pc = ts.M_old.z
    
    nx = pc.nx
    ny = pc.ny
    nz = pc.nz
    
    if s >= nz:
        print('Error, that slice (s) is not in the domain')
        raise Exception
    if cs >= nx:
        print('Error, that (cs) is not in the domain')
        raise Exception
    
    ### Step size in direction direc
    if direc == 'z':
        d = pc.dz
    elif direc == 'y':
        d = pc.dy
    elif direc == 'x':
        d == pc.dx
    else:
        print('Error in assigning step-size. Direction off. Perhaps its a capital letter.')
        raise Exception
    
    val = pc.value
           ## slices, cols, rows, i.e. z, y, x
    val = val.reshape((nz, ny, nx)) ## reshaping. Now a 2D plot as above. 
    
    # if direc == 'x':
        # plot_val = np.arange(1,nx+1)*dy
    
    if F == 'E':
        y_val = np.arange(1,ny+1)*dy
    elif F == 'M':
        y_val = np.arange(1, ny+1)*dy 
    fig,ax = plt.subplots(1,1)
    ax.plot(y_val, val[s,:,cs])
    '''
    The above is not finished. The difficult is selecting the values to plot
    against. not val[x,x,x] but rather x_val. Each field needs to start either
    at 1 or 0, depending on if it's inner, outer, and x,y,z. That's a lot of 
    if statements. Fow now, I'm going to leave it to only plot y-direction
    for Ey, Mz. 
    '''
    return fig, ax
 

ts.initialize_set_up_ADI()   
# ts.tol = 1e-8
for k in range(1,20):
    print(k)
    t = k*dt
    # LLG_run(t)
    # ts.T = round_to_3(t)
    
    ADI_run(t)
    ts.T = fsf.round_to_3(tsr.dt*k)
    # Yee_run(t)
    # tsr.T = fsf.round_to_3(tsr.dt*k)
    
    ## Double half-step forces tsr_old = ts_old time-wise
    # tsr.single_run_v2(t-dt/2)
    # reset_tsr_for_next_run()
    # tsr.T = round_to_3(tsr.dt*k)
    # Yee_run(t)
    
    if k%1 == 0:
        # print(ts.M_old.values[2].std())
        
        # ts.M_old.values = abs(tsr.M_old.values - ts.M_new.values)
        
        fig2,ax2 = ts.plot_line('E','y')
        ax2.set_title(r'ADI plot: $E_y$'+
                      '\n Ticker: '+str(k))
        
        fig3,ax3 = ts.plot_line('M','z')
        ax3.set_title(r'ADI plot: $M_z$'+
                      '\n Ticker: '+str(k))
        # fig2,ax2 = tsr.plot_slice('E','y')
        # ax2.set_title(r'Yee plot: $E_y$'+
        #               '\n Ticker: '+str(k))
        
        # fig3,ax3 = tsr.plot_slice('M','z')
        # ax3.set_title(r'Yee plot: $M_z$'+
        #               '\n Ticker: '+str(k))
        # ax.set_ylim(99.999,100.001)
        # fig,ax = tsr.plot_slice('M','z')
        # ax.set_zlim(99.999,100.001)
        # ax.set_zlim(99.999,100.001)
        # fig1, ax1 = plot_y_cs(F = 'M', comp = 'z', s = 0, cs = 4, direc = 'y')
        # fig2, ax2 = plot_y_cs(F = 'M', comp = 'z', s = 0, cs = 6, direc = 'y')
        
    reset_ts_for_next_run()
    # reset_tsr_for_next_run()
    
    ## For plotting only
    # tsr.E_old.values = tsr.E_new.values
    # tsr.M_old.values = tsr.M_new.values
    
    # tsr.plot_slice('E','y')
    # tsr.plot_slice('M','z')
    # tsr.plot_slice('E','z')
      
    
#############3 Old Code ################
# def LLG_ADI( t, X = 'old'):
#     '''
#     Uses Joly FDTD trick to solve LLG explicitly. 
    
#     If X = n, uses B_(n-1/2), B_n
    
#     Old refers to half-step, i.e. n to n+1/2 
#     New refers to whole-step, i.e. n+1/2 to n
    
#     Updates M_X
#     '''     
#     ## Parameters
#     dt = ts.dt/2 ### Half-step for ADI
#     bdp = ts.better_dot_pdt
#     gamma = ts.gamma
#     K = ts.K
    
#     # M_old = ts.M_old
        
#     f = 2*ts.M_old2.values
#     B_on = (ts.B_old2.values + ts.B_old.values)/2
    
#     a = -( (abs(gamma)*dt/2)*(B_on/mu0 + ts.H_s.values) + alpha*ts.M_old2.values)
#     lam = -K*abs(gamma)*dt/4
        
#     a_dot_f =  bdp(a.T,f.T).T
    
#     # ## Projection of 'easy' axis
#     # p_x = np.zeros(shape = (ts.M_old2.values.shape[1],1))
#     # p_y = np.copy(p_x)
#     # p_z = np.ones(shape = p_x.shape)
#     # p = np.concatenate((p_x, p_y, p_z), axis = 1).T
    
#     x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
#     x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
    
#     x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
        
#     return x_new_values - ts.M_old2.values

    
    
    
    
    
    
    
    
    
    