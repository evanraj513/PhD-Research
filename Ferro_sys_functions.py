#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:52:49 2020

Will store the functions to implement in ferro_system
Descriptions will be on each function. 

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


########################## General functions ##########################
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

        
def sizing(nx,ny,nz):
    '''
    Gives the number of **local** nodes for the outer and inner fields. 
    
    Recall that each component of each field lies on different nodes for
    the Yee scheme. This function returns these local node counts, given
    a global discretization in the x,y,z direction. 
    
    For now, nx = ny, and nz is either 1 or nx = ny = nz
    
    '''
    
    if nx != ny:
        print('Error. Discretization not currently available. Break')
        raise Exception
    
    if nz != 1:
        size_outer_x = np.array([(nx-1)/2, (ny+1)/2, (nz+1)/2])
        size_outer_y = np.array([(nx+1)/2, (ny-1)/2, (nz+1)/2])
        size_outer_z = np.array([(nx+1)/2, (ny+1)/2, (nz-1)/2])
        
        size_inner_x = np.array([(nx+1)/2, (ny-1)/2, (nz-1)/2])
        size_inner_y = np.array([(nx-1)/2, (ny+1)/2, (nz-1)/2])
        size_inner_z = np.array([(nx-1)/2, (ny-1)/2, (nz+1)/2])
        
    else:
        '''
        Note that for this case, the not-used will have the amount of 
        the field with which they are associated, simply to make concatentation
        work properly
        '''
        size_outer_x = np.array([(nx-1)/2, (ny+1)/2, 1])
        size_outer_y = np.array([(nx+1)/2, (ny-1)/2, 1])
        size_outer_z = np.array([(nx+1)/2, (ny-1)/2, 1]) # This will not be included in calculations
    
        
        size_inner_x = np.array([(nx-1)/2, (ny-1)/2, 1]) # This will not be included in calculations
        size_inner_y = np.array([(nx-1)/2, (ny-1)/2, 1]) # This will not be included in calculations
        size_inner_z = np.array([(nx-1)/2, (ny-1)/2, 1])
        
    return [size_outer_x, size_outer_y, size_outer_z,\
            size_inner_x, size_inner_y, size_inner_z]
        
def cardanos_method(a,b,c,d,eps = 1E-12):
    '''
    Solves for the real root of a cubic polynomial. 
    Note that it will not solve for the symmetric
    real roots, even if they exist. 
    See here for more explanation:
        https://brilliant.org/wiki/cardano-method/
    '''
    if abs(a) < eps:
        print('*'*40,'\n','Error, not a cubic. Aborting','\n','*'*40)
        raise Exception
        
    def real_cubic_root(arg):
        '''
        Forces python to return the real root of a cubic. 
        Note: "< 0" case works as 
            -(-n)^(1/3) = ((-1)^3(-n))^(1/3) = ((-1)(-n))^(1/3) = (n)^1/3
        '''
        if type(arg) == complex:
            return (arg)**(1/3)
        else:
            if arg < 0:
                return -(-arg)**(1/3)
            else:
                return (arg)**(1/3)
    
    Q = (3*a*c - b**2)/(9*a**2)
    R = (9*a*b*c - 27*a**2*d - 2*b**3)/(54*a**3)
#    
#    if Q**3 > R**2-eps:
#        print('*'*40,'\n','Break. Error in cardanos method, cannot use',
#              'Dont know why, but Q**3 > R**2 is hard.','\n','*'*40)
#        raise Exception
    
    S = real_cubic_root(R + (Q**3 + R**2)**(1/2))
    T = real_cubic_root(R - (Q**3 + R**2)**(1/2))
    print('Q: ',Q,'\n'*2,
          'R: ',R,'\n'*2,
          'S: ',S,'\n'*2,
          'T: ',T,'\n'*2)
    
    root = S+T - (b)/(3*a)
    
    if abs(root.imag) > eps:
        print('Warning. Imaginary part of the root is too large.')
        wait = input('Press ENTER, or CTRL C to break')        
    
    return root.real
    
#############################################################################
################ Boundary conditions, and Forcing functions #################
#############################################################################

### Ricker Pulse
    
## Ricker Pulse parameters
a = 1.34
b = 3.4
k = (np.pi/a)**2
c = 3E8

def Ricker_pulse(x):
    '''
    Actual Ricker pulse function
    '''
    if x < b:
        pa = 2*k/11*(2*k*(x-a)**2-1)
        pb = (np.e**(-k*(x-a)**2))
        
        return pa*pb
    else:
        return 0
    
## Non-ricker pulse function
def Exp_pulse(x):
    return np.exp(-(x)**2)
            

def Gaussian_source(dt,t):
    '''
    Gaussian source
    '''
    return (10-15*np.cos(t/dt*np.pi/20)+6*np.cos(2*t/dt*np.pi/20)-np.cos(3*t/dt*np.pi/20))/32




######################### Old Code ##################################

# def make_1D_plot(Field = 'E', comp = 'y', cs = 0, s = 0):
#     '''
#     Function to plot a 'flat' (i.e. 2D) plot of the comp of the field,
#     at some specific cross-section. Not a 3D slice, but a 2D cross-section
#     of a specific slice
#     '''
#     F = Field
    

#     if F == 'E':
#         if comp == 'x':
#             pc = R_sys.E_old.x
#             ind = R_sys.ind_rev_x_out
#         elif comp == 'y':
#             pc = R_sys.E_old.y
#             ind = R_sys.ind_rev_y_out
#         elif comp == 'z':
#             pc = R_sys.E_old.z
#             ind = R_sys.ind_rev_z_out
#         else:
#             print('Error, not "x", "y", "z". No comprendo, start over')
#             raise Exception
            
#     elif F == 'B':
#         if comp == 'x':
#             pc = R_sys.B_old.x
#             ind = R_sys.ind_rev_x_inn
#         elif comp == 'y':
#             pc = R_sys.B_old.y
#             ind = R_sys.ind_rev_y_inn
#         elif comp == 'z':
#             pc = R_sys.B_old.z
#             ind = R_sys.ind_rev_z_inn
#         else:
#             print('Error, not "x", "y", "z". No comprendo, start over')
#             raise Exception
            
#     elif F == 'M':
#         if comp == 'x':
#             pc = R_sys.B_old.x
#             ind = R_sys.ind_rev_x_inn
#         elif comp == 'y':
#             pc = R_sys.B_old.y
#             ind = R_sys.ind_rev_y_inn
#         elif comp == 'z':
#             pc = R_sys.B_old.z
#             ind = R_sys.ind_rev_z_inn
#         else:
#             print('Error, not "x", "y", "z". No comprendo, start over')
#             raise Exception
            
#     elif F == 'H':
#         if comp == 'x':
#             pc = R_sys.B_old.x
#             ind = R_sys.ind_rev_x_inn
#         elif comp == 'y':
#             pc = R_sys.B_old.y
#             ind = R_sys.ind_rev_y_inn
#         elif comp == 'z':
#             pc = R_sys.B_old.z
#             ind = R_sys.ind_rev_z_inn
#         else:
#             print('Error, not "x", "y", "z". No comprendo, start over')
#             raise Exception
        
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
    
#     m_s = s*pc.nx*pc.ny
#     if s >= pc.nz:
#         print('Error, outside of domain. Cannot plot')
#         raise Exception
        
#     if comp == 'x':
#         m_cs = pc.nx*cs
        
#         x1 = np.zeros(shape = (pc.nx,1))
#         y1 = np.copy(x1)
        
#         for k in np.arange(m_s+m_cs,m_s+m_cs+x1.shape[0]):
#             x1[k-m_s] = ind(k)[0]
#             y1[k-m_s] = pc.value[k]
            
#                   #   (num rows, num cols)
#         x1 = x1.reshape(pc.ny, pc.nx)
#         y1 = y1.reshape(pc.ny, pc.nx)
#         ax.plot(x1,y1)
#         title = 'Plot of: '+F+'_'+comp+'\n'+ 'slice number: '+str(s)+\
#                 '\n'+'cross_section: '+str(cs)
#     #        print(title)
#         ax.set_title(title)
    
#     return fig

    

# def det_E_M():
#     '''
#     Determines the vector of error_moduli for a given system
#     as described in running notes 2/13
#     '''
#     M = R_sys.M_old.values
    
#     E_M = abs(np.linalg.norm(M,axis=0) - np.linalg.norm(M0,axis=0))\
#     /(np.linalg.norm(M0,axis=0))
    
#     return E_M

