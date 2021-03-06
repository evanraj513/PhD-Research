#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 2020

@author: evanraj

This will serve as my second attempt at coding the coupled Maxwell's equations with
the LLG equation. System given as a class. 

"""

import os
import sys

l_path = '..'
m_path = os.path.abspath(l_path)
if not os.path.exists(m_path):
    print('Error importing modules. Need to specify new path')
    raise Exception
else:
    sys.path.append(m_path)

import numpy as np
#import scipy as sp
from scipy.sparse import lil_matrix, csr_matrix, linalg
from mpl_toolkits import mplot3d # Needed even though 'unused' variable
import matplotlib.pyplot as plt
from matplotlib import cm ## For adding colorbars
from scipy.linalg import norm
#from matplotlib import cm
plt.rcParams['backend'] = "Qt4Agg"

## For saving data
import pandas as pd
from collections import OrderedDict

from Research import field_class
Field = field_class.Field_2
Vector = field_class.Vector

### Ferrosystem parameters
mu0 = 1.25667e-6 #Permeability of free space
eps = 8.85422e-12 #Permoittivity of free space
sigma = 0.0 ## Conductivity. 0 = lossless media
gamma = 2.2e5 #From Puttha's paper
K = 0
alpha = 0.2
H_s_guess = 10**5

''' TO DO 
    1. (done) Find values for mu0, eps that are realistic (check paper again)
    2. (done) Determine how we will find M_new. In paper, just need to read it again
    3. (done) Determine if np.cross is working
    4. (done) Determine path stuff
    5. (done) Add in check that given initial values are np.arrays of arrays (maybe)
    '''

class Ferro_sys(object):
    '''    
    Initialization Attributes
    -------------------------       
    global_node_count: np.array, length = 1 or 3       Note: In old code, this was "size"
        Will give the number of nodes in one 'direction'
            - Assuming same number of nodes in each row
                if only one number is passed
    disc: np_array of length 3
        Contains the step-size in the x,y,z direction, in that order
    X0: list
        3 columns of initial conditions for field X, each column corresponding to 
        the X,Y,Z components 
        
    System Attributes
    -----------------  
    dt: float
        Time-step value. 
    T: float
        Run current time (for plotting only)
    gn_: int
        gives the global number of nodes in any direction. 
    a: np.ndarray
        Gives the different sizing for each component of the inner/outer Fields
    E_n_: np.ndarray
        [nx, ny, nz] for E_(_)
    B_n_: np.ndarray
        [nx, ny, nz] for B_(_) = H_(_) = M_(_)
    
    bound_ind: np.ndarray 3 by x
        Gives list of nodes on boundary
        
    '''
    
    def __init__(self,global_node_count,disc,E0,H0,M0,H_s = H_s_guess):
        # Parameter Set-up
        self.dt = 0.1 
        self.T = -1.0
        ### Ferrosystem parameters
        self.mu0 = mu0 #Permeability of free space
        self.eps = eps #Permoittivity of free space
        self.sigma = sigma #Conductivity
        self.gamma = gamma #LLG parameter 1
        self.K = K #Joly projection method
        self.alpha = alpha ## LLG parameter 2
        self.H_s_guess = H_s_guess ## Static magnetic field

        self.disc = disc
        self.node_count = global_node_count
        if global_node_count.size == 1:
            self.gnx = global_node_count
            self.gny = global_node_count
            self.gnz = global_node_count
        elif global_node_count.size == 3:
            self.gnx = global_node_count[0]
            self.gny = global_node_count[1]
            self.gnz = global_node_count[2]
        else:
            print('*'*40,'\n','Error in given node_count. Abort','\n','*'*40)
            raise Exception
        
        a = self.sizing(self.gnx,self.gny,self.gnz)
        self.a = a
        
        self.E_nx = a[0].astype('int')
        self.E_ny = a[1].astype('int')
        self.E_nz = a[2].astype('int')
        
        self.B_nx = a[3].astype('int')
        self.B_ny = a[4].astype('int')
        self.B_nz = a[5].astype('int')
         
        ### Below included to deal with updated names
        
        self.size_Bx = self.B_nx
        self.size_By = self.B_ny
        self.size_Bz = self.B_nz
        
        # Initialize all fields
        self.E_old = E0
        self.H_old = H0
        self.M_old = M0
        self.B0 = self.mu0*H0 + M0 
        self.B_old = self.B0
        self.H_s = H_s
        
        self.E_new = E0
        self.H_new = H0
        self.M_new = M0
        self.B_new = self.B0
        
        self.H_old2 = H0
        self.M_old2 = M0
        self.B_old2 = self.B0
        self.E_old2 = E0
        
        ### Gives nodes that are on the boundary 
        self.bound_ind = self.bound_ind()
        
        ## ADI parameters
        self.tol = 1E-12
        self.maxiter_half_step = 10
        self.maxiter_whole_step = 10
      
    ### Setters to initialize field classes
    @property
    def E_old2(self):
        return self._E_old2
    @E_old2.setter
    def E_old2(self, values):
        self._E_old2 = Field(self.E_nx, self.E_ny,self.E_nz,self.disc,values)
#        self._E_old.E = True
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to E_old')
            raise Exception
   
    @property
    def H_old2(self):
        return self._H_old2
    @H_old2.setter
    def H_old2(self, values):
        self._H_old2 = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to H_old')
            raise Exception        
        
    @property
    def M_old2(self):
        return self._M_old2
    @M_old2.setter
    def M_old2(self, values):
        self._M_old2 = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to M_old')
            raise Exception        
         
    @property
    def B_old2(self):
        return self._B_old2
    @B_old2.setter
    def B_old2(self, values):
        self._B_old2 = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to B_old')
            raise Exception  
        
    @property
    def E_old(self):
        return self._E_old
    @E_old.setter
    def E_old(self, values):
#        print(self.size_Ex, self.size_Ey, self.size_Ez)
        self._E_old = Field(self.E_nx, self.E_ny,self.E_nz,self.disc,values)
#        self._E_old.E = True
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to E_old')
            raise Exception
   
    @property
    def H_old(self):
        return self._H_old
    @H_old.setter
    def H_old(self, values):
        self._H_old = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to H_old')
            raise Exception        
        
    @property
    def M_old(self):
        return self._M_old
    @M_old.setter
    def M_old(self, values):
        self._M_old = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to M_old')
            raise Exception  
            
    @property
    def B_old(self):
        return self._B_old
    @B_old.setter
    def B_old(self, values):
        self._B_old = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to M_old')
            raise Exception  
            
    @property
    def E_new(self):
        return self._E_new
    @E_new.setter
    def E_new(self, values):
        self._E_new = Field(self.E_nx, self.E_ny
                            ,self.E_nz,self.disc,values)
#        self._E_new.E = True
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to E_new')
            raise Exception
   
    @property
    def H_new(self):
        return self._H_new
    @H_new.setter
    def H_new(self, values):
        self._H_new = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to H_new')
            raise Exception        
        
    @property
    def M_new(self):
        return self._M_new
    @M_new.setter
    def M_new(self, values):
        self._M_new = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to M_new')
            raise Exception        
         
    @property
    def B_new(self):
        return self._B_new
    @B_new.setter
    def B_new(self, values):
        self._B_new = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to B_new')
            raise Exception 
        else:
            pass
        
    @property
    def H_s(self):
        return self._H_s
    @H_s.setter
    def H_s(self, val):
#        shape1 = np.int(np.round(np.prod(self.size_inner_x)))
        values = val*np.ones(shape = self.B_old.values.shape)
        self._H_s = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to H_s')
            raise Exception      
            
        else: 
            pass
        
    def sizing(self,nx,ny,nz):
        '''
        For a given global nx, ny, nz, gives the nx, ny, nz that is associated with
        the x, y, z of both the 'inner' fields and 'outer fields' of the Yee Scheme

        The inner field refers to which Field has x,y components on the odd slices
        and the outer field refers to which Field has x,y, comp on the even slices
        
        Note that slices refer to z-slices. 
        '''
        
        
        if nz != 1:
            size_outer_x = np.array([(nx-1)/2, (ny+1)/2, (nz+1)/2])
            size_outer_y = np.array([(nx+1)/2, (ny-1)/2, (nz+1)/2])
            size_outer_z = np.array([(nx+1)/2, (ny+1)/2, (nz-1)/2])
        
        #    size_Bx = np.array([(nx-3)/2, (nx-1)/2, (nx-1)/2])
        #    size_By = np.array([(nx-1)/2, (nx-3)/2, (nx-1)/2])
        #    size_Bz = np.array([(nx-1)/2, (nx-1)/2, (nx-3)/2])
            
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
            size_outer_z = np.array([(nx-1)/2, (ny+1)/2, 1]) # Not included in calculations
        
        #    size_Bx = np.array([(nx-3)/2, (nx-1)/2, (nx-1)/2])
        #    size_By = np.array([(nx-1)/2, (nx-3)/2, (nx-1)/2])
        #    size_Bz = np.array([(nx-1)/2, (nx-1)/2, (nx-3)/2])
            
            size_inner_x = np.array([(nx-1)/2, (ny-1)/2, 1]) # Not included in calculations
            size_inner_y = np.array([(nx-1)/2, (ny-1)/2, 1]) # Not included in calculations
            size_inner_z = np.array([(nx-1)/2, (ny-1)/2, 1])
            
        return [size_outer_x, size_outer_y, size_outer_z,\
                size_inner_x, size_inner_y, size_inner_z]
        
    def ind_rev_x_out(self,m):
        '''
        This will return the global (x,y,z) value for the 
        mth node of the x component of the outer field
        '''
        
        x = self.E_old.x
        dx = x.dx
        dy = x.dy
        dz = x.dz
        
        [j,k,l] = x.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = dx + j*2*dx
        y = k*2*dy
        z = l*2*dz
        
        return np.array([x,y,z])
    
    def ind_rev_y_out(self,m):
        '''
        This will return the global (x,y,z) value for the 
        mth node of the y component of the outer field
        '''
        
        y = self.E_old.y
        dx = y.dx
        dy = y.dy
        dz = y.dz
        
        [j,k,l] = y.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = j*2*dx
        y = dy + k*2*dy
        z = l*2*dz
        
        return np.array([x,y,z])
    
    def ind_rev_z_out(self,m):
        '''
        This will return the global (x,y,z) value for the 
        mth node of the z component of the outer field
        '''
        
        z = self.E_old.z
        dx = z.dx
        dy = z.dy
        dz = z.dz
        
        [j,k,l] = z.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = j*2*dx
        y = k*2*dy
        z = dz + l*2*dz
        
        return np.array([x,y,z])
    
    def ind_rev_x_inn(self,m):
        '''
        This will return the global (x,y,z) value for the 
        m^th node of the x component of the inner field
        '''
        
        x = self.B_old.x
        dx = x.dx
        dy = x.dy
        dz = x.dz
        
        [j,k,l] = x.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = j*2*dx
        y = dy + k*2*dy
        z = dz + l*2*dz
        
        return np.array([x,y,z])
    
    def ind_rev_y_inn(self,m):
        '''
        This will return the global (x,y,z) value for the 
        m^th node of the y component of the inner field
        '''
        
        z = self.E_old.z
        dx = z.dx
        dy = z.dy
        dz = z.dz
        
        [j,k,l] = z.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = dx + j*2*dx
        y = k*2*dy
        z = dz + l*2*dz
        
        return np.array([x,y,z])
    
    def ind_rev_z_inn(self,m):
        '''
        This will return the global (x,y,z) value for the 
        mth node of the z component of the inner field
        '''
        
        z = self.B_old.z
        dx = z.dx
        dy = z.dy
        dz = z.dz
        
        [j,k,l] = z.index_rev(m)
        
        j = j/dx
        k = k/dy
        l = l/dz
        
        x = dx + j*2*dx
        y = dy + k*2*dy
        z = l*2*dz
        
        return np.array([x,y,z])
    
    def plot_slice(self,F = 'E',comp = 'y',s = 0):
        '''
        Plots the 'comp' component of the 'F' field in the 's' slice down the z-axis
        
        Note that this plotted component is denoted pc
        '''
        
        if F == 'E':
            if comp == 'x':
                pc = self.E_old.x
                ind = self.ind_rev_x_out
            elif comp == 'y':
                pc = self.E_old.y
                ind = self.ind_rev_y_out
            elif comp == 'z':
                pc = self.E_old.z
                ind = self.ind_rev_z_out
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'B':
            if comp == 'x':
                pc = self.B_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.B_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.B_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'M':
            if comp == 'x':
                pc = self.M_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.M_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.M_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'H':
            if comp == 'x':
                pc = self.H_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.H_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.H_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
            
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
        
        m_s = s*pc.nx*pc.ny
        if s >= pc.nz:
            print('Error, outside of domain. Cannot plot')
            raise Exception
        
        x1 = np.zeros(shape = (pc.nx*pc.ny,1))
        y1 = np.copy(x1)
        z1 = np.copy(x1)
        
        for k in np.arange(m_s,m_s+x1.shape[0]):
            x1[k-m_s] = ind(k)[0]
            y1[k-m_s] = ind(k)[1]
            z1[k-m_s] = pc.value[k]
            
                  #   (num rows, num cols)
        x1 = x1.reshape(pc.ny, pc.nx)
        y1 = y1.reshape(pc.ny, pc.nx)
        z1 = z1.reshape(pc.ny, pc.nx)
        ax.plot_surface(x1,y1,z1,cmap=cm.coolwarm)
        title = 'Plot of: '+F+'_'+comp+' at time: '+str(self.T)+'\n'+ 'slice: '+str(s)
#        print(title)
        ax.set_title(title)
        
        return fig,ax
    
    def plot_line(self,Field = 'E', comp = 'y', cs = 0, s = 0):
        '''
        Function to plot a 'flat' (i.e. 2D) plot of the comp of the field,
        at some specific cross-section. Not a 3D slice, but a 2D cross-section
        of a specific slice
        
        cs is the cross_section you want plotted
        s is the slice number you will take the cross-section from.
        
        pc stands for plot component
        ind stands for index
        
        m_s is the number of nodes to add to get to the correct slice
        m_cs is the number of nodes to add to get to the correct cross_section
        
        Will always plot across the x_axis for now.
        
        '''
        F = Field

    
        if F == 'E':
            if comp == 'x':
                pc = self.E_old.x
                ind = self.ind_rev_x_out
            elif comp == 'y':
                pc = self.E_old.y
                ind = self.ind_rev_y_out
            elif comp == 'z':
                pc = self.E_old.z
                ind = self.ind_rev_z_out
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'B':
            if comp == 'x':
                pc = self.B_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.B_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.B_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'M':
            if comp == 'x':
                pc = self.M_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.M_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.M_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
                
        elif F == 'H':
            if comp == 'x':
                pc = self.H_old.x
                ind = self.ind_rev_x_inn
            elif comp == 'y':
                pc = self.H_old.y
                ind = self.ind_rev_y_inn
            elif comp == 'z':
                pc = self.H_old.z
                ind = self.ind_rev_z_inn
            else:
                print('Error, not "x", "y", "z". No comprendo, start over')
                raise Exception
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        m_s = s*pc.nx*pc.ny
        if s >= pc.nz:
            print('Error, outside of domain. Cannot plot')
            raise Exception
            
        if comp == 'x':
            m_cs = pc.nx*cs
            
            x1 = np.zeros(shape = (pc.nx,1))
            y1 = np.copy(x1)
            
            for k in np.arange(m_s+m_cs,m_s+m_cs+x1.shape[0]-1):
                x1[k-m_cs] = ind(k)[0]
                y1[k-m_cs] = pc.value[k]
                
            ax.plot(x1,y1)
#            title = 'Plot of: '+F+'_'+comp+'\n'+ 'slice number: '+str(s)+\
#                    '\n'+'cross_section: '+str(cs)
            title = 'Plot of: '+F+'_'+comp+'\n'+\
                    '\n'+'cross_section: '+str(cs)+'\n'+\
                    'Time: '+str(self.T)
            ax.set_xlabel('x')
            ax.set_ylabel(F+comp)
        #        print(title)
            ax.set_title(title)
            
        elif comp == 'y':
            '''
            Why this works
            --------------
            m_cs is defined as is because the discretization is along the x-axis
            so pc.nx represents how many nodes to add, regardless of which comp
            you are plotting
            
            Similarly, this is why x1 is the shape of pc.nx
            
            If one were plotting a different axis, you would have to go through
            and select the proper indices, probably building some sort of set of
            indices. 
            '''
            
            
            m_cs = pc.nx*cs
            
            x1 = np.zeros(shape = (pc.nx,1))
            y1 = np.copy(x1)
            
            for k in np.arange(m_s+m_cs,m_s+m_cs+x1.shape[0]):
                x1[k-m_cs] = ind(k)[0]
                y1[k-m_cs] = pc.value[k]
                
                      #   (num rows, num cols)
            ax.plot(x1,y1)
            title = 'Plot of: '+F+'_'+comp+'\n'+ 'slice number: '+str(s)+\
                    '\n'+'cross_section: '+str(cs)+'\n'+\
                    'Time: '+str(self.T)
        #        print(title)
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel(F+comp)
            
        elif comp == 'z':
            '''
            Why this works? See above
            '''
            
            m_cs = pc.nx*cs
            
            x1 = np.zeros(shape = (pc.nx,1))
            y1 = np.copy(x1)
            
            for k in np.arange(m_s+m_cs,m_s+m_cs+x1.shape[0]):
                x1[k-m_cs] = ind(k)[0]
                y1[k-m_cs] = pc.value[k]
                
                      #   (num rows, num cols)
            ax.plot(x1,y1)
            title = 'Plot of: '+F+'_'+comp+' at time: '+str(self.T)+'\n'\
                    +'slice: ' + str(s)+ 'cross_section: ' + str(cs)
        #        print(title)
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel(F+comp)
            
        return fig, ax    
    
    def fx(self,x1,y1,z1,t1):
        a = x1*y1*z1
        if t1 < 1:
            return 0
        else:
            return a
        
    def fy(self,x1,y1,z1,t1):
        if t1 < 1:
            return 0
        else:
            return y1**2
        
    def fz(self,x1,y1,z1,t1):
        if t1 < 1:
            return 0
        else:
            return y1**2
    
    def Fx(self,t):
        F_x = np.zeros(shape=(self.E_old.x.value.shape[0],1))
        dx = self.disc[0]
        dy = self.disc[1]
        dz = self.disc[2]
        
        nx = self.E_old.x.nx
        ny = self.E_old.x.ny
        nz = self.E_old.x.nz
        for ll in np.arange(0,nz):
            for kk in np.arange(0,ny):
                for jj in np.arange(0,nx):
                    x = (jj+1/2)*2*dx
                    y = kk*2*dy
                    z = ll*2*dz
                    # print('jj',jj,'\n',
                    #       'kk',kk,'\n',
                    #       'll',ll,'\n',
                    #       'x: ',x,'\n',
                    #       'y: ',y,'\n',
                    #       'z: ',z,'\n')
                    # if jj == nx-2:
                    #     wait = input('Press ENTER to continue')
                        
                    F_x[jj + nx*kk + nx*ny*ll] = self.fx(x,y,z,t)
                    
        return F_x

    def Fy(self, t):
        F_y = np.zeros(shape=(self.E_old.y.value.shape[0],1))
        dx = self.disc[0]
        dy = self.disc[1]
        dz = self.disc[2]
    
        nx = self.E_old.y.nx
        ny = self.E_old.y.ny
        nz = self.E_old.y.nz
        
        for ll in np.arange(0,nz):
            for kk in np.arange(0,ny):
                for jj in np.arange(0,nx):
                    x = jj*2*dx
                    y = (kk+1/2)*2*dy
                    z = ll*2*dz
                    # print('jj',jj,'\n',
                    #       'kk',kk,'\n',
                    #       'll',ll,'\n',
                    #       'x: ',x,'\n',
                    #       'y: ',y,'\n',
                    #       'z: ',z,'\n')
#                    if jj == nx-2:
#                        wait = input('Press ENTER to continue')
                    F_y[jj + nx*kk + nx*ny*ll] = self.fy(x,y,z,t)

    
        return F_y
    
    def Fz(self,t):
        F_z = np.zeros(shape=(self.E_old.z.value.shape[0],1))
        dx = self.disc[0]
        dy = self.disc[1]
        dz = self.disc[2]
        
        nx = self.E_old.z.nx
        ny = self.E_old.z.ny
        nz = self.E_old.z.nz
        for ll in np.arange(0,nz):
            for kk in np.arange(0,ny):
                for jj in np.arange(0,nx):
                    x = jj*2*dx
                    y = kk*2*dy
                    z = ll*2*dz
                    
                    # print(x/dx,y/dy,z/dz)
                    
                    F_z[jj + nx*kk + nx*ny*ll] = self.fz(x,y,z,t)

        return F_z
    
    def bound_ind(self):
        '''
        This function will compute the indices of the boundary nodes for 
        both E and B. This will depend on if it is an even grid or an odd grid
        '''
        Ex = self.E_old.x
        Ey = self.E_old.y
        Ez = self.E_old.z
        
        Bx = self.B_old.x
        By = self.B_old.y
        # Bz = self.B_old.z
        
        ### Indexing for Odd node count
        if (self.node_count %2 == 1).all() and self.gnz != 1:
            ind_Ex = np.array([0])
            for ll in np.arange(0,Ex.nz):
                for kk in np.arange(0,Ex.ny):
                    for jj in np.arange(0,Ex.nx):
                        ind = [jj+kk*Ex.nx + ll*Ex.nx*Ex.ny]
                        if ll == 0 or ll == Ex.nz-1:
                            ind_Ex = np.concatenate((ind_Ex, ind))
                        elif kk == 0 or kk == Ex.ny-1:
                            ind_Ex = np.concatenate((ind_Ex, ind))
                            
            ind_By = np.array([0])
            for ll in np.arange(0,By.nz):
                for kk in np.arange(0,By.ny):
                    for jj in np.arange(0,By.nx):
                        ind = [jj+kk*By.nx + ll*By.nx*By.ny]
                        if kk == 0 or kk == By.ny-1:
                            ind_By = np.concatenate((ind_By, ind))
                            
                            
            ind_Ey = np.array([0])
            for ll in np.arange(0,Ey.nz):
                for kk in np.arange(0,Ey.ny):
                    for jj in np.arange(0,Ey.nx):
                        ind = [jj+kk*Ey.nx + ll*Ey.nx*Ey.ny]
                        if ll == 0 or ll == Ey.nz-1:
                            ind_Ey = np.concatenate((ind_Ey, ind))
                        elif jj == 0 or jj == Ey.nx-1:
                            ind_Ey = np.concatenate((ind_Ey, ind))
                            
            ind_Bx = np.array([0])
            for ll in np.arange(0,Bx.nz):
                for kk in np.arange(0,Bx.ny):
                    for jj in np.arange(0,Bx.nx):
                        ind = [jj+kk*Bx.nx + ll*Bx.nx*Bx.ny]
                        if jj == 0 or jj == Bx.nx-1:
                            ind_Bx = np.concatenate((ind_Bx, ind))
                            
            ind_Ez = np.array([0])
            for ll in np.arange(0,Ez.nz):
                for kk in np.arange(0,Ez.ny):
                    for jj in np.arange(0,Ez.nx):
                        ind = [jj+kk*Ez.nx + ll*Ez.nx*Ez.ny]
                        if kk == 0 or kk == Ez.ny-1:
                            ind_Ez = np.concatenate((ind_Ez, ind))
                        elif jj == 0 or jj == Ez.nx-1:
                            ind_Ez = np.concatenate((ind_Ez, ind))
                            
            ind_Bz = np.array([])
            
        ## Indexing for 2D case: TE mode ##
        elif (self.node_count %2 == 1).all() and self.gnz == 1:
            # print('we made it here')
            ind_Ex = np.array([0])
            for ll in np.arange(0,Ex.nz):
                for kk in np.arange(0,Ex.ny):
                    for jj in np.arange(0,Ex.nx):
                        ind = [jj+kk*Ex.nx + ll*Ex.nx*Ex.ny]
#                        if ll == 0 or ll == Ex.nz-1:
#                            ind_Ex = np.concatenate((ind_Ex, ind))
                        if kk == 0 or kk == Ex.ny-1:
                            ind_Ex = np.concatenate((ind_Ex, ind))
                            
            ind_By = np.array([0])
            for ll in np.arange(0,By.nz):
                for kk in np.arange(0,By.ny):
                    for jj in np.arange(0,By.nx):
                        ind = [jj+kk*By.nx + ll*By.nx*By.ny]
                        if kk == 0 or kk == By.ny-1:
                            ind_By = np.concatenate((ind_By, ind))
                            
                            
            ind_Ey = np.array([0])
            for ll in np.arange(0,Ey.nz):
                for kk in np.arange(0,Ey.ny):
                    for jj in np.arange(0,Ey.nx):
                        ind = [jj+kk*Ey.nx + ll*Ey.nx*Ey.ny]
#                        if ll == 0 or ll == Ey.nz-1:
#                            ind_Ey = np.concatenate((ind_Ey, ind))
                        if jj == 0 or jj == Ey.nx-1:
                            ind_Ey = np.concatenate((ind_Ey, ind))
                            
            ind_Bx = np.array([0])
            for ll in np.arange(0,Bx.nz):
                for kk in np.arange(0,Bx.ny):
                    for jj in np.arange(0,Bx.nx):
                        ind = [jj+kk*Bx.nx + ll*Bx.nx*Bx.ny]
                        if jj == 0 or jj == Bx.nx-1:
                            ind_Bx = np.concatenate((ind_Bx, ind))
                            
            ind_Ez = np.array([0])
            for ll in np.arange(0,Ez.nz):
                for kk in np.arange(0,Ez.ny):
                    for jj in np.arange(0,Ez.nx):
                        ind = [jj+kk*Ez.nx + ll*Ez.nx*Ez.ny]
                        if kk == 0 or kk == Ez.ny-1:
                            ind_Ez = np.concatenate((ind_Ez, ind))
                        elif jj == 0 or jj == Ez.nx-1:
                            ind_Ez = np.concatenate((ind_Ez, ind))
                            
            ind_Bz = np.array([])
            
        else:
            print('Did not enter into loop')
            print(self.node_count%2 == 1, self.gnz == 1)
            print(type(self.node_count))
            print('The error is probably a data-type in node_count, need to round!')
                        
                        
        ind_Ex = np.unique(ind_Ex)
        ind_Ey = np.unique(ind_Ey)
        ind_Ez = np.unique(ind_Ez)
        
        ind_Bx = np.unique(ind_Bx)
        ind_By = np.unique(ind_By)
        ind_Bz = np.unique(ind_Bz)
        
        return np.array([ind_Ex, ind_Ey, ind_Ez, ind_Bx, ind_By, ind_Bz])
        
    def single_run(self,t):
        '''
        A single time-step implemented
        Note that it assumes this is the staggered grid:
            
            |     |     |     |
          E_old B_old E_new B_new
          
        with associated variables M,H matching the B spots
        
        t: at what time step is this?
            
        '''
        ## Set-up field parameters
#        size = self.sizes
#        disc = self.disc
        
        E_old = self.E_old
        H_old = self.H_old
        M_old = self.M_old
        B_old = self.B_old
        
        dt = self.dt
        b_ind = self.bound_ind
        bdp = self.better_dot_pdt
        
        ## Parameters
        mu0 = self.mu0
        eps = self.eps
        gamma = self.gamma
        K = self.K
        alpha = self.alpha
        
        ## Actual computation of time stepping
        F = np.concatenate((self.Fx(t), self.Fy(t), self.Fz(t)),axis=1)
        E_new_values = E_old.values + dt/eps*H_old.curl()
        
        #Setting all E boundaries to 0
        for j in b_ind[0]:
            E_new_values[0][j] = 0 #x_bound(j)
        for k in b_ind[1]:
            E_new_values[1][k] = 0
        for l in b_ind[2]:
            E_new_values[2][l] = 0
        
        #Forcing term and boundary conditions inside F
        E_new_values = E_new_values+F.T
        self.E_new = E_new_values
        
        self.E_new_setup()
        
        B_new_values = B_old.values - dt*self.E_new.curl()
        self.B_new = B_new_values
        
        B_on = (B_old.values + B_new_values)/2
        
        f = 2*M_old.values
        a = -(abs(gamma)*dt/2)*(B_on/mu0 + self.H_s.values) - alpha*M_old.values
        lam = -K*abs(gamma)*self.dt/4
        
        a_dot_f =  bdp(a.T,f.T).T
        
        p_x = np.zeros(shape = (M_old.values.shape[1],1))
        p_y = np.copy(p_x)
        p_z = np.ones(shape = (M_old.values.shape[1],1))
        p = np.concatenate((p_x, p_y, p_z), axis = 1).T
        
        if K == 0 or abs(t-dt) < 1e-12:
            x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
            x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
            
            x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
                
        else:
            
            cubic_solver = self.cubic_solver
            
            a1 = lam**2
            b1 = 2*lam*(bdp(a.T, p.T) + lam*(bdp(p.T, f.T)))
            c1 = 1+np.linalg.norm(a) - lam*(bdp(a.T, f.T)) + 3*lam*\
            (bdp(a.T, p.T)) * (bdp(p.T, f.T)) + \
            lam**2*(bdp(p.T, f.T))
            d1 = -lam*(bdp(a.T, p.T)*(bdp(p.T,f.T))) - (bdp(a.T, p.T)*(bdp(p.T,f.T)))\
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
              + (((np.linalg.norm(a))**2*Y) - (bdp(a.T,p.T))).T*p\
              + (Z*np.cross(a.T, p.T)).T)
            
            
        self.M_new = x_new_values.T - M_old.values
        
        self.H_new = B_new_values/mu0 - self.M_new.values
        
    ### Derivative matrices ###
    def set_up_der_matrices(self):
        '''
        Saves the derivative matrices to used in single_run_v2
        
        The B,M matrices are identical to the H matrices
        and thus won't be stored. 
        '''
        
        self.Ex_Dy_mat = self.E_old.x.Dy_E_mat()
        self.Ex_Dz_mat = self.E_old.x.Dz_E_mat()
        
        self.Ey_Dx_mat = self.E_old.y.Dx_E_mat()
        self.Ey_Dz_mat = self.E_old.y.Dz_E_mat()
        
        self.Ez_Dx_mat = self.E_old.z.Dx_E_mat()
        self.Ez_Dy_mat = self.E_old.z.Dy_E_mat()
        
        self.Hx_Dy_mat = self.H_old.x.Dy_B_mat()
        self.Hx_Dz_mat = self.H_old.x.Dz_B_mat()
        
        self.Hy_Dx_mat = self.H_old.y.Dx_B_mat()
        self.Hy_Dz_mat = self.H_old.y.Dz_B_mat()
        
        self.Hz_Dx_mat = self.H_old.z.Dx_B_mat()
        self.Hz_Dy_mat = self.H_old.z.Dy_B_mat()
        
    def set_up_H_curl(self):
        '''
        Sets up the curl of H as an array stored in H_old_curl
        '''
        
        Hx = self.H_old.x.value
        Hy = self.H_old.y.value
        Hz = self.H_old.z.value
        
        Hc1 = self.Hz_Dy_mat*Hz - self.Hy_Dz_mat*Hy
        Hc2 = self.Hx_Dz_mat*Hx - self.Hz_Dx_mat*Hz
        Hc3 = self.Hy_Dx_mat*Hy - self.Hx_Dy_mat*Hx
        
        self.H_old_curl = np.array([Hc1,Hc2,Hc3])
        
        
    def set_up_E_curl(self):
        '''
        Sets up the curl of E as an array stored in H_old_curl
        '''
        Ex = self.E_new.x.value
        Ey = self.E_new.y.value
        Ez = self.E_new.z.value
        
        Ec1 = self.Ez_Dy_mat*Ez - self.Ey_Dz_mat*Ey
        Ec2 = self.Ex_Dz_mat*Ex - self.Ez_Dx_mat*Ez
        Ec3 = self.Ey_Dx_mat*Ey - self.Ex_Dy_mat*Ex
        
        self.E_new_curl = np.array([Ec1,Ec2,Ec3])
        
    def curl_L(self,values,Field = ''):
        '''
        Sets up the left side of the curl for ADI method
        
        values is the (3 x n) np.array that is the Fields values
        of which we are taking the curl_L of
        
        Field states whether this is in inner or outer field
        
        i signifies inner (standard for Yee scheme M,H,B)
        o signifies outer (standard for Yee scheme E)
        
        '''
        
        F = Field
        
        if F == 'Outer' or F == 'o':
            Ec1 = self.Ez_Dy_mat*values[2]
            Ec2 = self.Ex_Dz_mat*values[0]
            Ec3 = self.Ey_Dx_mat*values[1]
            
        elif F == 'Inner' or F == 'i':
            Ec1 = self.Hz_Dy_mat*values[2]
            Ec2 = self.Hx_Dz_mat*values[0]
            Ec3 = self.Hy_Dx_mat*values[1]
            
        else:
            print('Warning. Input not valid. Break and reconfigure')
            raise Exception
        
        return np.array([Ec1,Ec2,Ec3])
        
    def curl_R(self, values, Field = ''):
        '''
        Sets up the right side of the curl for ADI method
        
        values is the (3 x n) np.array that is the Fields values
        of which we are taking the curl_L of
        
        Field states whether this is in inner or outer field
        
        i signifies inner (standard for Yee scheme M,H,B)
        o signifies outer (standard for Yee scheme E)
        '''

        F = Field
        
        if F == 'Outer' or F == 'o':
            Ec1 = self.Ey_Dz_mat*values[1]
            Ec2 = self.Ez_Dx_mat*values[2]
            Ec3 = self.Ex_Dy_mat*values[0]
            
        elif F == 'Inner' or F == 'i':
            Ec1 = self.Hy_Dz_mat*values[1]
            Ec2 = self.Hz_Dx_mat*values[2]
            Ec3 = self.Hx_Dy_mat*values[0]
            
        else:
            print('Warning. nput not valid. Break and reconfigure')
            raise Exception
            
        return np.array([Ec1,Ec2,Ec3])
    
    def curl_LL(self, values):
        '''
        Returns the two-time left-side curl operator applied to the values. 
        '''
        
        Ec1 = self.Hy_Dz_mat.dot(self.Ez_Dx_mat.dot(values[2]))
        Ec2 = self.Hz_Dx_mat.dot(self.Ex_Dy_mat.dot(values[0]))
        Ec3 = self.Hx_Dy_mat.dot(self.Ey_Dz_mat.dot(values[1]))
        
        return np.array([Ec1, Ec2, Ec3])
    
    def curl_RR(self, values):
        '''
        Returns the two-time right-side curl operator applied to values
        '''
        
        Ec1 = self.Hz_Dy_mat.dot(self.Ey_Dx_mat.dot(values[1]))
        Ec2 = self.Hx_Dz_mat.dot(self.Ez_Dy_mat.dot(values[2]))
        Ec3 = self.Hy_Dx_mat.dot(self.Ex_Dz_mat.dot(values[0]))
        
        return np.array([Ec1, Ec2, Ec3])
    
    def curl_LR(self, values):
        
        Ec1 = self.Hz_Dy_mat.dot(self.Ex_Dy_mat.dot(values[0]))
        Ec2 = self.Hx_Dz_mat.dot(self.Ey_Dz_mat.dot(values[1]))
        Ec3 = self.Hy_Dx_mat.dot(self.Ez_Dx_mat.dot(values[2]))
        
        return np.array([Ec1, Ec2, Ec3])
    
    def curl_RL(self, values):
        
        Ec1 = self.Hy_Dz_mat.dot(self.Ex_Dz_mat.dot(values[0]))
        Ec2 = self.Hz_Dx_mat.dot(self.Ey_Dx_mat.dot(values[1]))
        Ec3 = self.Hx_Dy_mat.dot(self.Ez_Dy_mat.dot(values[2]))
        
        return np.array([Ec1, Ec2, Ec3])
    
    def set_up_step_1a_inv(self):
        '''
        Sets up step_1a (as in huang 2018) to compute E^n+1/2
        '''
        ## Parameters
        dt = self.dt
        eps = self.eps
        mu0 = self.mu0
        
        ## Double operator set-up (curl_L)*(curl_R)
        Ec1 = self.Hz_Dy_mat.dot(self.Ex_Dy_mat)
        Ec2 = self.Hx_Dz_mat.dot(self.Ey_Dz_mat)
        Ec3 = self.Hy_Dx_mat.dot(self.Ez_Dx_mat)
        
                            ## 1 - dt^2/(4*eps*mu0)*(curl_L)*(curl_R)
        self.a1_1 = csr_matrix(np.identity(Ec1.shape[0])) - (dt**2)/(4*eps*mu0)*Ec1
        self.a1_2 = csr_matrix(np.identity(Ec2.shape[0])) - (dt**2)/(4*eps*mu0)*Ec2
        self.a1_3 = csr_matrix(np.identity(Ec3.shape[0])) - (dt**2)/(4*eps*mu0)*Ec3
        
        self.a1_1lu = linalg.splu(self.a1_1)
        self.a1_2lu = linalg.splu(self.a1_2)
        self.a1_3lu = linalg.splu(self.a1_3)
        
    def step_1a_inv(self,values):
        '''
        Completes step_1a (as in huang 2018) to compute E^n+1/2
        '''
        
        Ret_1 = self.a1_1lu.solve(values[0])
        Ret_2 = self.a1_2lu.solve(values[1])
        Ret_3 = self.a1_3lu.solve(values[2])
        
        return np.array([Ret_1, Ret_2, Ret_3])
    
    def set_up_step_2a_inv(self):
        '''
        Sets up step_1a (as in huang 2018) to compute E^n+1/2
        '''
        ## Parameters
        dt = self.dt
        eps = self.eps
        mu0 = self.mu0
        
        ## Double operator set-up (curl_R)*(curl_L)
        Ec1 = self.Hy_Dz_mat.dot(self.Ex_Dz_mat)
        Ec2 = self.Hz_Dx_mat.dot(self.Ey_Dx_mat)
        Ec3 = self.Hx_Dy_mat.dot(self.Ez_Dy_mat)
        
                            ## 1 - dt^2/(4*eps*mu0)*(curl_R)*(curl_L)
        self.a2_1 = csr_matrix(np.identity(Ec1.shape[0])) - (dt**2)/(4*eps*mu0)*Ec1
        self.a2_2 = csr_matrix(np.identity(Ec2.shape[0])) - (dt**2)/(4*eps*mu0)*Ec2
        self.a2_3 = csr_matrix(np.identity(Ec3.shape[0])) - (dt**2)/(4*eps*mu0)*Ec3
        
        ## Setting up lu factorization
        self.a2_1lu = linalg.splu(self.a1_1)
        self.a2_2lu = linalg.splu(self.a1_2)
        self.a2_3lu = linalg.splu(self.a1_3)
        
    def step_2a_inv(self,values):
        '''
        Completes up step_1a (as in huang 2018) to compute E^n+1/2
        '''
        
        Ret_1 = self.a2_1lu.solve(values[0])
        Ret_2 = self.a2_2lu.solve(values[1])
        Ret_3 = self.a2_3lu.solve(values[2])
        
        return np.array([Ret_1, Ret_2, Ret_3])
        
    def initialize_set_up_ADI(self):
        '''
        Runs all the set-ups above for the ADI method
        '''
        self.set_up_der_matrices()
        self.set_up_step_1a_inv()
        self.set_up_step_2a_inv()
        
        
    def single_run_v2(self,t):
        '''
        An update to the above run. Main difference is that derivative matrices 
        are now stored, and reused. Also, Vectors and Fields aren't regenerated
        but rather their values are updated
        '''
        ## Set-up field parameters
#        size = self.sizes
#        disc = self.disc
        
        E_old = self.E_old
        H_old = self.H_old
        M_old = self.M_old
        B_old = self.B_old
        
        dt = self.dt
        b_ind = self.bound_ind
        bdp = self.better_dot_pdt
        
        ## Parameters
        mu0 = self.mu0
        eps = self.eps
        gamma = self.gamma
        K = self.K
        alpha = self.alpha
        
        ## Get curl values
        self.set_up_H_curl()
        
        ## Actual computation of time stepping
        F = np.concatenate((self.Fx(t), self.Fy(t), self.Fz(t)),axis=1)
        E_new_values = E_old.values + dt/eps*self.H_old_curl##*(self.curl_L(H_old.values,'i') - self.curl_R(H_old.values,'i'))
        
        #Setting all E boundaries to 0
        for j in b_ind[0]:
            E_new_values[0][j] = 0 #x_bound(j)
        for k in b_ind[1]:
            E_new_values[1][k] = 0
        for l in b_ind[2]:
            E_new_values[2][l] = 0
        
        #Forcing term and boundary conditions inside F
        E_new_values = E_new_values+F.T
        
        self.E_new.values = E_new_values
        
        # self.E_new_setup()
        self.set_up_E_curl()
        
        B_new_values = B_old.values - dt*self.E_new_curl
        self.B_new.values = B_new_values
        
        B_on = (B_old.values + B_new_values)/2
        
        f = 2*M_old.values
        a = -(abs(gamma)*dt/2)*(B_on/mu0 + self.H_s.values) - alpha*M_old.values
        lam = -K*abs(gamma)*self.dt/4
        
        a_dot_f =  bdp(a.T,f.T).T
        
        p_x = np.zeros(shape = (M_old.values.shape[1],1))
        p_y = np.copy(p_x)
        p_z = np.ones(shape = (M_old.values.shape[1],1))
        p = np.concatenate((p_x, p_y, p_z), axis = 1).T
        
        if K == 0 or abs(t-dt) < 1e-12:
            x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
            x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
            
            x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
                
        else:
            
            cubic_solver = self.cubic_solver
            
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
            
            
        self.M_new.values = x_new_values.T - M_old.values
        
        self.H_new.values = B_new_values/mu0 - self.M_new.values        
        
    def single_run_ADI(self,t):
        '''
        A single time-step for ADI scheme implemented. 
        
        Main idea and algebraic steps outlined in Huang_2018 and Yao 2017
        
        M will be computed at each half step using Joly
        Then ADI method for full maxwells equations with non-zero M
            will be implemented to update remaining fields
        
        '''
        
        ### Parameters
        E_old2 = self.E_old2
        H_old2 = self.H_old2
        M_old2 = self.M_old2
        B_old2 = self.B_old2
        
        E_old = self.E_old
        B_old = self.B_old
        M_old = self.M_old
        H_old = self.H_old
        
        E_new = self.E_new
        B_new = self.B_new
        M_new = self.M_new
        H_new = self.H_new
        
        dt = self.dt
        b_ind = self.bound_ind
        bdp = self.better_dot_pdt
        
        ## Parameter choices given in system
        mu0 = self.mu0
        eps = self.eps
        gamma = self.gamma
        K = self.K
        alpha = self.alpha
        sigma = self.sigma 

        ################ Solving for Mn+1 ##################
        f = 2*M_old2.values
        a = -(abs(gamma)*dt/2)*(B_old2.values/mu0 + self.H_s.values) - alpha*M_old2.values
        lam = -K*abs(gamma)*dt/4
        
        a_dot_f =  bdp(a.T,f.T).T
        
        ## Projection of 'easy' axis
        p_x = np.zeros(shape = (M_old2.values.shape[1],1))
        p_y = np.copy(p_x)
        p_z = np.ones(shape = (M_old2.values.shape[1],1))
        p = np.concatenate((p_x, p_y, p_z), axis = 1).T
        
        if K == 0 or t == dt:
            x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
            x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
            
            x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
                
        else:
            
            cubic_solver = self.cubic_solver
            
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
            
            
        self.M_new.values = x_new_values.T - M_old2.values
        
        ### Setting M_n+1/2 values to be numerical average between old2 and new
        
        self.M_old.values = (self.M_old2.values + self.M_new.values)/2
        
        ######################################################
        #################### ADI Scheme ######################
        ######################################################
        
        ################ First Half step ###################
        
        ##### Solving for E_n+1/2
        s_a = 1/mu0*self.curl_L(B_old2.values,'Inner')
        s_b = dt/(2*mu0)*self.curl_LL(E_old2.values)
        s_c = self.curl_L(M_old.values,'Inner')
        s_d = self.curl_R(H_old2.values,'Inner')
        
        s_main = (s_a - s_b - s_c - s_d)
        E_old_RHS = E_old2.values + dt/(2*eps)*s_main - (dt/2)*sigma*E_old2.values
        
        ### Add in forcing terms at the half-step
        # F_old = np.concatenate((self.Fx(t-dt/2), self.Fy(t-dt/2), self.Fz(t-dt/2)),axis=1)
        
        # E_old_RHS += dt/2*F_old.T
        
        ### Using back-solve for new values
        E_old_values = self.step_1a_inv(E_old_RHS)
        
        # #Setting all E boundaries to 0
        # b_ind = self.bound_ind

        # for j in b_ind[0]:
        #     E_old_values[0][j] = 0 #x_bound(j)
        # for k in b_ind[1]:
        #     E_old_values[1][k] = 0
        # for l in b_ind[2]:
        #     E_old_values[2][l] = 0
            
        # E_old_values += F_old.T
        
        self.E_old.values = E_old_values
        
        ###### Solving for B_n+1/2
        B_old_values = B_old2.values + dt/2*(\
            self.curl_R(E_old_values, 'o') - \
                self.curl_L(E_old2.values, 'o'))
            
        self.B_old.values = B_old_values
        
        ###### Solving for H_n+1/2
        H_old_values = 1/mu0*B_old_values - M_old.values
        self.H_old.values = H_old_values
        
        ################ Second Half step ###################
        ### Solving for E_n+1
        
        s2_a = self.curl_L(H_old.values, 'i')
        s2_b = 1/mu0*self.curl_R(B_old.values, 'i')
        s2_c = dt/(2*mu0)*self.curl_RR(E_old.values)
        s2_d = self.curl_R(M_new.values,'i')
        
        s2_main = s2_a - s2_b - s2_c + s2_d
        
        F_new = np.concatenate((self.Fx(t), self.Fy(t), self.Fz(t)),axis=1)
        
        E_new_RHS = E_old.values + dt/(2*eps)*s2_main - dt/2*sigma*E_old.values
        
        # E_new_RHS += F_new.T
        
        # Backsolve
        
        E_new_values = self.step_2a_inv(E_new_RHS)
        
        #Setting all E boundaries to 0
        for j in b_ind[0]:
            E_new_values[0][j] = 0 #x_bound(j)
        for k in b_ind[1]:
            E_new_values[1][k] = 0
        for l in b_ind[2]:
            E_new_values[2][l] = 0
        
        E_new_values += F_new.T
        
        self.E_new.values = E_new_values
        
        ### Solving for B_n+1
        self.B_new.values = B_old.values + dt/2*(\
                        self.curl_R(E_old.values,'o') - 
                        self.curl_L(E_new_values, 'o'))
        
        ### Solving H_n+1
        self.H_new_values = 1/mu0*B_new.values - M_new.values
        
    def LLG_ADI(self, t, X = 'old'):
        '''
        Uses Joly FDTD trick to solve LLG explicitly. 
        
        If X = n, uses B_(n-1/2), B_n
        
        Old refers to half-step, i.e. n to n+1/2 
        New refers to whole-step, i.e. n+1/2 to n
        
        Updates M_X
        '''     
        ## Parameters
        dt = self.dt/2 ### Half-step for ADI
        bdp = self.better_dot_pdt
        gamma = self.gamma
        K = self.K
        
        M_old = self.M_old
                
        if X == 'old' or X == 'Old': ## Set-up for half-step
        ## old2 = n-1/2, old = n+1/2 for Joly legibility
            
            f = 2*self.M_old2.values
            B_on = (self.B_old2.values + self.B_old.values)/2
            
            a = -( (abs(gamma)*dt/2)*(B_on/mu0 + self.H_s.values) + alpha*self.M_old2.values)
            lam = -K*abs(gamma)*dt/4
            
        elif X == 'new' or X == 'New':
            f = 2*self.M_old.values
            B_on = (self.B_old.values + self.B_new.values)/2
            
            a = -( (abs(gamma)*dt/2)*(B_on/mu0 + self.H_s.values) + alpha*self.M_old.values)
            lam = -K*abs(gamma)*dt/4
            
        a_dot_f =  bdp(a.T,f.T).T
        
        ## Projection of 'easy' axis
        p_x = np.zeros(shape = (self.M_old2.values.shape[1],1))
        p_y = np.copy(p_x)
        p_z = np.ones(shape = (self.M_old2.values.shape[1],1))
        p = np.concatenate((p_x, p_y, p_z), axis = 1).T
        
        if K == 0 or t == dt:
            x_new_num = f + (a_dot_f)*a - np.cross(a.T,f.T).T
            x_new_den = np.array(1+np.linalg.norm(a,axis=0)**2).T
            
            x_new_values = np.divide(x_new_num.T, np.array([x_new_den]).T)
                
        else:
            
            cubic_solver = self.cubic_solver
            
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
            return x_new_values.T - self.M_old2.values
        elif X == 'new' or X == 'New':
            return x_new_values.T - self.M_old.values
        else:
            print('Something is wrong in LLG_ADI')
            raise Exception
            
    def ADI_first_half(self,t):
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
        
        E_old2 = self.E_old2
        H_old2 = self.H_old2
        # M_old2 = self.M_old2
        B_old2 = self.B_old2
        
        E_old = self.E_old
        B_old = self.B_old
        M_old = self.M_old
        H_old = self.H_old
        
        dt = self.dt
        # bdp = self.better_dot_pdt
        
        ## Parameter choices given in system
        mu0 = self.mu0
        eps = self.eps
        sigma = self.sigma 
        
        ##### Solving for E_n+1/2
        s_a = 1/mu0*self.curl_L(B_old2.values,'Inner')
        s_b = dt/(2*mu0)*self.curl_LL(E_old2.values)
        s_c = self.curl_L(M_old.values,'Inner')
        s_d = self.curl_R(H_old2.values,'Inner')
        
        s_main = (s_a - s_b - s_c - s_d)
        E_old_RHS = E_old2.values + dt/(2*eps)*s_main - (dt/2)*sigma*E_old2.values ## +dt/2*J
        
        ### Using back-solve for new values
        E_old_values = self.step_1a_inv(E_old_RHS)
        
        # Setting all E boundaries to 0
        b_ind = self.bound_ind
        
        for j in b_ind[0]:
            E_old_values[0][j] = 0 #x_bound(j)
        for k in b_ind[1]:
            E_old_values[1][k] = 0
        for l in b_ind[2]:
            E_old_values[2][l] = 0
            
        ### Add in forcing terms at the half-step
        F_old = np.concatenate((self.Fx(t-dt/2), self.Fy(t-dt/2), self.Fz(t-dt/2)),axis=1)
        E_old_values += F_old.T
        
        E_old.values = E_old_values
        
        # self.plot_slice()
        
        ###### Solving for B_n+1/2
        B_old_values = B_old2.values + dt/2*(\
            self.curl_R(E_old_values, 'o') - \
                self.curl_L(E_old2.values, 'o'))
            
        B_old.values = B_old_values
        
        ###### Solving for H_n+1/2
        H_old_values = 1/mu0*B_old_values - M_old.values
        H_old.values = H_old_values
        
        # self.plot_line('H','z')
        
    def ADI_second_half(self,t):
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
        
        E_old = self.E_old
        B_old = self.B_old
        # M_old = self.M_old
        H_old = self.H_old
        
        E_new = self.E_new
        B_new = self.B_new
        M_new = self.M_new
        H_new = self.H_new
        
        dt = self.dt
        b_ind = self.bound_ind
        # bdp = self.better_dot_pdt
        
        ## Parameter choices given in system
        mu0 = self.mu0
        eps = self.eps
        sigma = self.sigma 
        
        s2_a = self.curl_L(H_old.values, 'i')
        s2_b = 1/mu0*self.curl_R(B_old.values, 'i')
        s2_c = dt/(2*mu0)*self.curl_RR(E_old.values)
        s2_d = self.curl_R(M_new.values,'i')
        
        s2_main = s2_a - s2_b - s2_c + s2_d
        
        E_new_RHS = E_old.values + dt/(2*eps)*s2_main - dt/2*sigma*E_old.values ## +dt/2*J
        
        # Backsolve
        
        E_new_values = self.step_2a_inv(E_new_RHS)
        
        #Setting all E boundaries to 0
        for j in b_ind[0]:
            E_new_values[0][j] = 0 #x_bound(j)
        for k in b_ind[1]:
            E_new_values[1][k] = 0
        for l in b_ind[2]:
            E_new_values[2][l] = 0
        
        # ### Forcing terms at the half-step
        
        F_new = np.concatenate((self.Fx(t), self.Fy(t), self.Fz(t)),axis=1)
        E_new_values += F_new.T
        
        E_new.values = E_new_values
        
        ### Solving for B_n+1
        B_new.values = B_old.values + dt/2*(\
                        self.curl_R(E_old.values,'o') - 
                        self.curl_L(E_new_values, 'o'))
        
        ### Solving H_n+1
        H_new.values = 1/mu0*B_new.values - M_new.values
        
    def single_run_ADI_v2(self,t):
        '''
        A single time-step for ADI scheme implemented. 
        
        Main idea and algebraic steps outlined in Huang_2018 and Yao 2017
        
        Let n be the current time. This is the algorithm that will be implemented
        
        1. Mn+1/2(1) computed using **only** Bn and Mn
        2. En+1/2 computed using ADI half-step described in daily write up Week 5/6
        3. Bn+1/2 updated with ADI half-step
        4. Mn+1/2 **recomputed** with Bn, Bn+1/2, Mn
            while res = Mn+1/2(1) - Mn+1/2(4) < tol and ticker < max_iter
            4.a. Mn+1/2(1) computed using Bn, Bn+1/2, Mn
            Steps 2-3 redone, Mn+1/2(4) recomputed
            
        5. Repeat steps 1-4 for the next half-step. 
        '''
        ####################################################
        ################ First Half step ###################
        ####################################################
        
        # print('Running the new version')
        
        ## Half-step parameters
        tol = self.tol
        ticker1 = 0
        
        ################ Solving for Mn+1/2 (1) ################
        self.B_old.values = self.B_old2.values ### Setting up for only forward scheme LLG
        M_values_1 = self.LLG_ADI('old')
        
        self.M_old.values = M_values_1 
                
        ############# Updating E,B,H n+1/2 (2-3) ################
        self.ADI_first_half(t)
        # self.plot_line('E','y')
        
        ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (4) #######
        M_values_4 = self.LLG_ADI(t, 'old')
        
        while ticker1 < self.maxiter_half_step and norm(M_values_1 - M_values_4) > tol:
            ticker1 += 1
            M_values_1 = M_values_4
            self.M_old.values = M_values_1
            self.ADI_first_half(t)
            M_values_4 = self.LLG_ADI('old')
            print('Current iteration for half-step fixed point:', ticker1)
            print('Current residual for half-step:', norm(M_values_1 - M_values_4))
        
        if ticker1 == self.maxiter_half_step:
            print('**Warning. Convergence in fixed-point not reached in first half step**')
            
        self.M_old.values = M_values_4
        
        #####################################################
        ################ Second Half step ###################
        ##################################################### 
        
        ## Full-step parameters
        ticker2 = 0
        
        ################# Solving for M_n+1 (5) #################
        self.B_new.values = self.B_old.values ### Setting for forward scheme LLG
        M_new_values_1 = self.LLG_ADI(t, 'new')
        
        self.M_new.values = M_new_values_1
        
        ############# Updating E,B,H n+1  (6-7) ################
        self.ADI_second_half(t)
        # self.E_old.values = self.E_new.values
        # self.plot_line('E','y')
        
        ########## Computing Mn+1/2(4) using Bn, Bn+1/2 (8) #######
        M_new_values_4 = self.LLG_ADI('new')
        
        while ticker2 < self.maxiter_whole_step and norm(M_values_1 - M_new_values_4) > tol:
            ticker2+=1
            M_new_values_1 = M_new_values_4
            self.M_new.values = M_new_values_1
            self.ADI_first_half(t)
            M_new_values_4 = self.LLG_ADI('new')
            print('Current iteration for whole-step fixed point:', ticker2)
            print('Current residual for whole-step:', norm(M_values_1 - M_values_4))
            
        self.M_new.values = M_new_values_4
            
        if ticker2 == self.maxiter_whole_step:
            print('**Warning. Convergence in fixed-point not reached in second half step**')
        
        
        
    def cubic_solver(self,a,b,c,d,x0,disp = 'no'):
        '''
        Solves the cubic function f(x) = ax^3 + bx^2 + cx + d = 0
        for the real root, assuming f(x) has only one real root
        '''
        
        step_size = 1/np.linalg.norm(self.M_old.values)
        ss = step_size
        
        def res_cubic(x):
            '''
            cubic function
            '''
            return a*x**3 + b*x**2 + c*x + d
        
        root = self.secant_method(res_cubic,x0,x0 + ss,maxit=100)
        
        if disp == 'yes' or disp == 'Yes':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.arange(root-50000*ss,root+50000*ss,100*ss)
            ax.plot(x,res_cubic(x))
            plt.show()
        else:
            pass
        
        return root
        
        
    def better_dot_pdt(self,a,b):
        '''
        This computes the dot-product applied to two arrays of 
        function values. Returns array of dot-product values i.e. 
        
        a.shape = [m,n]
        b.shape = [m,n]
        
        Let k be either a or b
        
        k = (k_x, k_y, k_z)
        
        k_ij = k(i)_j = k[i][j]
        
        bdp(a,b)[l] = a[1][l] * b[1][l] + ... + a[n-1][l] * b[n-1][l]
        
                            This sum will be done with np.dot
                            
                            
        '''
        
        if a.shape != b.shape:
            print('Error in dot pdt. Abort')
            raise Exception
            
        val = np.zeros(shape = (a.shape[0], 1))
            
        for k in np.arange(0,a.shape[0]):
            try:
                val[k] = np.dot(a[k], b[k])
            except:
                print('Error at k = ',k,' assignment')
                raise Exception
#            else:
#                print('Something went wrong, I dont know what'
#                      '\n','Aborting')
#                raise Exception
            
        return val
        
        
        
    def secant_method(self,func, x0, x1, alpha1=1.0, tol=1E-9, maxit=200):
        """
        Uses the secant method to find f(x)=0.  
        
        INPUTS
        
            * f     : function f(x)
            * x0    : initial guess for root
            * x1    : second guess for root
            * alpha1 : relaxation coefficient: modifies Secant step size
            * tol   : convergence tolerance
            * maxit : maximum number of iteration, default=200        
        """
    
        x, xprev = x1, x0
        f, fprev = func(x), func(xprev)
        
        rel_step = 2.0 *tol
        k = 0
        
        while (abs(f) > tol) and (rel_step) > tol and (k<maxit):        
            rel_step = abs(x-xprev)/abs(x)
            
            # Full secant step
            dx = -f/(f - fprev)*(x - xprev)
            
            # Update `xprev` and `x` simultaneously
            xprev, x = x, x + alpha1*dx
            
            # Update `fprev` and `f`:
            fprev, f = f, func(x)
            
            k += 1
        if k == maxit-1:
            print('Warning: convergence reached to: ', func(x),
                  'Method terminated as max iterations reached',
                  '\n Proceed? \n')
            wait = input('Press ENTER, or CTRL C to break')        
                    
        return x
    
    def save_data(self,name = 'blank.csv'):
        ''' 
        Converts data to pandas data-frame, and saves as two csv
        under 'name_outer', 'name_inner'. Requires two csv's as different size
        arrays
        '''
        
        E_old = self.E_old
        B_old = self.B_old
        H_old = self.H_old
        M_old = self.M_old
        
        data_outer = OrderedDict()
        data_inner = OrderedDict()
        
        data_outer['E.x'] = E_old.x.value
        data_outer['E.y'] = E_old.y.value
        data_outer['E.z'] = E_old.z.value
        
        data_inner['B.x'] = B_old.x.value
        data_inner['B.y'] = B_old.y.value
        data_inner['B.z'] = B_old.z.value
        
        data_inner['H.x'] = H_old.x.value
        data_inner['H.y'] = H_old.y.value
        data_inner['H.z'] = H_old.z.value
    
        data_inner['M.x'] = M_old.x.value
        data_inner['M.y'] = M_old.y.value
        data_inner['M.z'] = M_old.z.value
        
        ## Converting to DataFrame and Saving
        df_outer = pd.DataFrame(data_outer)
        df_inner = pd.DataFrame(data_inner)

        df_outer.to_csv(name+'_outer.csv')
        df_inner.to_csv(name+'_inner.csv')



