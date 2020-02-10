#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:06:04 2019

@author: evanraj

This will serve as my first attempt at coding the coupled Maxwell's equations with
the LLG equation. System given as a class

"""

import os
import sys

import numpy as np

l_path = '..'
m_path = os.path.abspath(l_path)
if not os.path.exists(m_path):
    print('Error importing modules. Need to specify new path')
    raise Exception
else:
    sys.path.append(m_path)

import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix

from Research import field_class
Field = field_class.Field_2
Vector = field_class.Vector
mu0 = 10.0 
eps = 1.0
gamma = 1.0
K = 0
alpha = 1.0

''' TO DO 
    1. Find values for mu0, eps that are realistic (check paper again)
    2. (done) Determine how we will find M_new. In paper, just need to read it again
    3. Determine if np.cross is working
        - It should be. Based on the 
    4. (done) Determine path stuff
    5. Add in check that given initial values are np.arrays of arrays (maybe)
    '''

class Ferro_sys(object):
    '''
    First attempt at coding system from (INSERT NAME). 
    
    Attributes
    ----------
    X0: list
        3 columns of initial conditions for field X, each column corresponding to 
        the X,Y,Z components        
    size: integer
        Will give the number of nodes in one 'direction'
            - Assuming same number of nodes in each row
    disc: np_array of length 3
        Contains the step-size in the x,y,z direction, in that order
    dt: float
        Time-step value. 
    T: float
        Final time
        
    '''
    
    
    
    def __init__(self,size,disc,E0,H0,M0,H_s):
        # Parameter Set-up
        self.dt = 0.1 
        self.T = 1.0

        self.disc = disc
        self.size = size
        nx = size
        
        # Sizing #
        ### Sizing for even number of nodes v.1.1
#        self.size_Ex = np.array([nx/2, nx/2, nx/2]).astype('int')
#        self.size_Ey = np.array([nx/2, nx/2, nx/2]).astype('int')
#        self.size_Ez = np.array([nx/2, nx/2, nx/2]).astype('int')
#        
#        self.size_Bx = np.array([nx/2, nx/2, nx/2]).astype('int')
#        self.size_By = np.array([nx/2, nx/2, nx/2]).astype('int')
#        self.size_Bz = np.array([nx/2, nx/2, nx/2]).astype('int')
        
        ### Sizing for odd number of nodes below 
        self.size_Ex = np.array([(nx-1)/2, (nx+1)/2, (nx+1)/2]).astype('int')
        self.size_Ey = np.array([(nx+1)/2, (nx-1)/2, (nx+1)/2]).astype('int')
        self.size_Ez = np.array([(nx+1)/2, (nx+1)/2, (nx-1)/2]).astype('int')
        
        self.size_Bx = np.array([(nx+1)/2, (nx-1)/2, (nx-1)/2]).astype('int')
        self.size_By = np.array([(nx-1)/2, (nx+1)/2, (nx-1)/2]).astype('int')
        self.size_Bz = np.array([(nx-1)/2, (nx-1)/2, (nx+1)/2]).astype('int')
#       ### Old sizing v.1
#        self.size_Bx = np.array([(nx-3)/2, (nx-1)/2, (nx-1)/2]).astype('int')
#        self.size_By = np.array([(nx-1)/2, (nx-3)/2, (nx-1)/2]).astype('int')
#        self.size_Bz = np.array([(nx-1)/2, (nx-1)/2, (nx-3)/2]).astype('int')
        
        ## See Research Notes 1/27/20 for more explanation on above ##
        
        # Field set-up #
        self.E_old = E0

        self.H_s = H_s
        self.H_old = H0
        self.M_old = M0
        B0 = mu0*H0 + M0 #Curious is this will work. Looks like it should. 
        self.B_old = B0
        
        self.E_new = E0
        self.H_new = H0
        self.M_new = M0
        self.B_new = B0
        
        self.bound_ind = self.bound_ind()
        
    @property
    def E_old(self):
        return self._E_old
    @E_old.setter
    def E_old(self, values):
        self._E_old = Field(self.size_Ex, self.size_Ey,self.size_Ez,self.disc,values)
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
            print('Something is wrong. Assigned incorrect array to B_old')
            raise Exception  
            
    @property
    def E_new(self):
        return self._E_new
    @E_new.setter
    def E_new(self, values):
        self._E_new = Field(self.size_Ex, self.size_Ey
                            ,self.size_Ez,self.disc,values)
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
    def H_s(self, values):
        self._H_s = Field(self.size_Bx, self.size_By
                            ,self.size_Bz,self.disc,values)
        
        if type(values) != np.ndarray or values.shape[0] != 3:
            print('Something is wrong. Assigned incorrect array to H_new')
            raise Exception        
        
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
        
        x = dx + j*2*dx
        y = k*2*dy
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
        
        x = j*2*dx
        y = dy + k*2*dy
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
    
    def set_up(self):
        '''
        This will set-up the system to be run. 
        -Should this just be in the initialize? Maybe, who knows?
            Ans: Probably not, as we will re-initialize each field as an update to
                the single run. Thus, will need to keep reassigning derivatives and whatnot
        '''
#        self.H_old.x.Dx = self.H_old.x.Dx_B
        self.H_old.x.Dy = self.H_old.x.Dy_B
        self.H_old.x.Dz = self.H_old.x.Dz_B
        
        self.H_old.y.Dx = self.H_old.y.Dx_B
#        self.H_old.y.Dy = self.H_old.y.Dy_B
        self.H_old.y.Dz = self.H_old.y.Dz_B
        
        self.H_old.z.Dx = self.H_old.z.Dx_B
        self.H_old.z.Dy = self.H_old.z.Dy_B
#        self.H_old.z.Dz = self.H_old.z.Dz_B
        
#        self.M_old.x.Dx = self.H_old.x.Dx_B
        self.M_old.x.Dy = self.M_old.x.Dy_B
        self.M_old.x.Dz = self.M_old.x.Dz_B
        
        self.M_old.y.Dx = self.M_old.y.Dx_B
#        self.M_old.y.Dy = self.M_old.y.Dy_B
        self.M_old.y.Dz = self.M_old.y.Dz_B
        
        self.M_old.z.Dx = self.M_old.z.Dx_B
        self.M_old.z.Dy = self.M_old.z.Dy_B
#        self.M_old.z.Dz = self.M_old.z.Dz_B

#        self.B_old.x.Dx = self.B_old.x.Dx_B
        self.B_old.x.Dy = self.B_old.x.Dy_B
        self.B_old.x.Dz = self.B_old.x.Dz_B
        
        self.B_old.y.Dx = self.B_old.y.Dx_B
#        self.B_old.y.Dy = self.B_old.y.Dy_B
        self.B_old.y.Dz = self.B_old.y.Dz_B
        
        self.B_old.z.Dx = self.B_old.z.Dx_B
        self.B_old.z.Dy = self.B_old.z.Dy_B
#        self.B_old.z.Dz = self.B_old.z.Dz_B    
        
    
    
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
        for ll in np.arange(0,nz-1):
            for kk in np.arange(0,ny-1):
                for jj in np.arange(0,nx-1):
#                    print('jj',jj,'\n',
#                          'kk',kk,'\n',
#                          'll',ll)
                    x = (jj+1/2)*2*dx
                    y = kk*2*dy
                    z = ll*2*dz
                    F_x[jj + nx*kk + nx*ny*ll] = self.fx(x,y,z,t)
                    
        return F_x
        
    def Fy(self,t):
        F_y = np.zeros(shape=(self.E_old.y.value.shape[0],1))
        dx = self.disc[0]
        dy = self.disc[1]
        dz = self.disc[2]
        
        nx = self.E_old.y.nx
        ny = self.E_old.y.ny
        nz = self.E_old.y.nz
        for ll in np.arange(0,nz-1):
            for kk in np.arange(0,ny-1):
                for jj in np.arange(0,nx-1):
                    x = jj*2*dx
                    y = (kk+1/2)*2*dy
                    z = ll*2*dz
                    
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
        for ll in np.arange(0,nz-1):
            for kk in np.arange(0,ny-1):
                for jj in np.arange(0,nx-1):
                    x = jj*2*dx
                    y = (kk+1/2)*2*dy
                    z = ll*2*dz
                    
                    F_z[jj + nx*kk + nx*ny*ll] = self.fy(x,y,z,t)

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
        Bz = self.B_old.z
        
        ### Indexing for Even node grid
#        ind_Ex = np.array([0]) #First node is always a boundary
#        for ll in np.arange(0,Ex.nz):
#            for kk in np.arange(0,Ex.ny):
#                for jj in np.arange(0,Ex.nx):
#                    if ll == 0:
#                        ind = [ll*Ex.nx*Ex.ny + kk*Ex.nx + jj]
#                        ind_Ex= np.concatenate((ind_Ex, ind))
#                    elif kk == 0:
#                        ind = [ll*Ex.nx*Ex.ny + kk*Ex.nx + jj]
#                        ind_Ex= np.concatenate((ind_Ex, ind))
#                    elif jj == Ex.nx-1: 
#                    # Note: This does NOT double count nodes if ll==0 and jj==0
#                        ind = [ll*Ex.nx*Ex.ny + kk*Ex.nx + jj]
#                        ind_Ex= np.concatenate((ind_Ex, ind))
#                        
#        ind_By = np.array([0])
#        for ll in np.arange(0,By.nz):
#            for kk in np.arange(0,By.ny):
#                for jj in np.arange(0,By.nx):
#                    ind = [ll*By.nx*By.ny + kk*By.nx + jj]
#                    if ll == Bx.nz-1:
#                        ind_By = np.concatenate((ind_By, ind))
#                    elif kk == 0:
#                        ind_By = np.concatenate((ind_By, ind))
#                    elif jj == Ex.nx-1: 
#                        ind_By = np.concatenate((ind_By, ind))
#        
#        ind_Ey = np.array([0])
#        for ll in np.arange(0,Ey.nz):
#            for kk in np.arange(0,Ey.ny):
#                for jj in np.arange(0,Ey.nx):
#                    ind = [ll*Ey.nx*Ey.ny + kk*Ey.nx + jj]
#                    if ll == 0:
#                        ind_Ey = np.concatenate((ind_Ey, ind))
#                    elif kk == Ex.ny-1:
#                        ind_Ey = np.concatenate((ind_Ey, ind))
#                    elif jj == 0:
#                        ind_Ey = np.concatenate((ind_Ey, ind))
#                        
#        ind_Bx = np.array([0])
#        for ll in np.arange(0,Bx.nz):
#            for kk in np.arange(0,Bx.ny):
#                for jj in np.arange(0,Bx.nx):
#                    ind = [ll*Bx.nx*Bx.ny + kk*Bx.nx + jj]
#                    if ll == Bx.nz-1:
#                        ind_Bx = np.concatenate((ind_Bx, ind))
#                    elif kk == Ex.ny-1:
#                        ind_Bx = np.concatenate((ind_Bx, ind))
#                    elif jj == 0:
#                        ind_Bx = np.concatenate((ind_Bx, ind))
#        
#        ind_Ez = np.array([0])
#        for ll in np.arange(0,Ez.nz):
#            for kk in np.arange(0,Ez.ny):
#                for jj in np.arange(0,Ez.nx):
#                    ind = [ll*Ez.nx*Ez.ny + kk*Ez.nx + jj]
#                    if ll == Ez.nz-1:
#                        ind_Ez = np.concatenate((ind_Ez, ind))
#                    elif kk == 0:
#                        ind_Ez = np.concatenate((ind_Ez, ind))
#                    elif jj == 0:
#                        ind_Ez = np.concatenate((ind_Ez, ind))
#                        
#        ind_Bz = np.array([0])
#        for ll in np.arange(0,Bz.nz):
#            for kk in np.arange(0,Bz.ny):
#                for jj in np.arange(0,Bz.nx):
#                    ind = [ll*Bz.nx*Bz.ny + kk*Bz.nx + jj]
#                    if ll == Bz.nz-1:
#                        ind_Bz = np.concatenate((ind_Bz, ind))
#                    elif kk == 0:
#                        ind_Bz = np.concatenate((ind_Bz, ind))
#                    elif jj == 0:
#                        ind_Bz = np.concatenate((ind_Bz, ind))
#                    
#        ind_Ex = np.unique(ind_Ex)
#        ind_Ey = np.unique(ind_Ey)
#        ind_Ez = np.unique(ind_Ez)
#        
#        ind_Bx = np.unique(ind_Bx)
#        ind_By = np.unique(ind_By)
#        ind_Bz = np.unique(ind_Bz)
        
        
        
        
        
        
        ### Indexing for Odd node count
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
#        for ll in np.arange(0,Bz.nz):
#            for kk in np.arange(0,Bz.ny):
#                for jj in np.arange(0,Bz.nx):
#                    ind = [jj+kk*Bz.nx + ll*Bz.nx*Bz.ny]
#                    if 
                        
                        
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
        
        T: at what time step is this?
            
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
        
        ## Actual computation of time stepping
        F = np.array([self.Fx(t), self.Fy(t), self.Fz(t)])
        E_new_values = E_old.values + dt*H_old.curl()
        
        #Setting E boundaries to 0
        for j in b_ind[0]:
            E_new_values[0][j] = 0
        for k in b_ind[1]:
            E_new_values[1][k] = 0
        for l in b_ind[2]:
            E_new_values[2][l] = 0
        
        #Forcing term and boundary conditions inside F
        E_new_values = E_new_values+F
        self.E_new = E_new_values
        
        B_new_values = B_old.values - dt*self.E_new.curl()
        self.B_new = B_new_values
        
        B_on = (B_old.values + B_new_values)/2
        
        f = 2*M_old.values
        a = -(abs(gamma)*self.dt/2)*(B_on/mu0 + self.H_s.values) - alpha*M_old.values
        lam = -K*abs(gamma)*self.dt/4
        
        a_dot_f = np.array([a[0].T@f[0], a[1].T@f[1], a[2].T@f[2]])
        
        if K == 0:
            M_new_values = (f + (a_dot_f)*a - np.cross(a.T,f.T).T) 
            Mnv_denom = (1+np.linalg.norm(a,axis=1)**2)
            
            for k in np.arange(0,M_new_values.shape[0]):
                M_new_values[k] = M_new_values[k]/Mnv_denom[k]
            
            
        elif type(K)!= int or type(K) != float:
            print('Error in K, not a number?')
            
        else:
            print('Currently under construction')
            
        self.M_new = M_new_values
        
        self.H_new = B_new_values/mu0 - M_new_values
        #Need to add in how we find M_new for K ~= 0
        
        
        
    def plot_xy_slice(self,field,comp,num_of_slice):
        '''
        Plots the num_of_slice slice (in the z-direction) of the 
        component of the field given
        
        available fields: E,H,M,B
        available comps: x,y,z
        
        E.g.
        '''
        
        
        
        
    def Ex_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 1:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 0:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break')
            raise Exception
        else:
            ind1 = k*((nx+1)/2*(nx-1)/2)
            ind2 = j/2*(nx-3)/2
            ind = ind1+ind2+i
            
        return ind
    
    def Ey_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 0:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 1:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break')            
            raise Exception
        
        else:
            ind1 = k*((nx+1)/2*(nx-1)/2)
            ind2 = (j-1)/2*(nx-1)/2
            ind = ind1+ind2+i
            
        return ind
        
    def Ez_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 1:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 1:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break')            
            raise Exception
        else:            
            
            ind1 = k*((nx-3) + (nx+1)/2*(nx-3)/2)
            
            if j == 1:
                ind2 = 0
            else:
                ind2 = (j-3)/2*(nx+1)/2*(nx-3)/2 + (nx-3)/2
            ind = ind1+ind2+i
            
        return ind
        
    
    def Bx_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 1:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 0:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
            
        elif i == 0:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there, boundary node. Break now')
            raise Exception
        elif i == nx-1:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there, boundary node. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break') 
            raise Exception
        
        else:
            ind1 = k*((nx-1)/2*(nx-3)/2)
            ind2 = (j-1)/2*(nx-3)/2
            ind = ind1+ind2+i
            
            return ind
        
    def By_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 0:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 1:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
            
        elif j == 0:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there, boundary node. Break now')
            raise Exception
        elif j == nx-1:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there, boundary node. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break')   
            raise Exception
        
        else:
            ind1 = k*((nx-1)/2*(nx-3)/2)
            ind2 = j/2*(nx-1)/2
            ind = ind1+ind2+i
            
            return ind
        
    def Bz_ind(self,x,y,z):
        nx = self.nx
        dx = self.dx
        
        i = round(x/dx)
        j = round(y/dx)
        k = round(z/dx)
        
        if i%2 != 0:
            print('*'*40,'\n','Mistake in Ex-x input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif j%2 != 1:
            print('*'*40,'\n','Mistake in Ex-y input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif k%2 != 0:
            print('*'*40,'\n','Mistake in Ex-z input. Nodal value not','\n',...
                  ,' assigned there. Break now')
            raise Exception
        elif i > nx-1 or j > nx-1 or k > nx-1:
            print('*'*40,'\n','Mistake. Some component is too large. Break')     
            raise Exception
        
        else:
            ind1 = k*((nx-1)/2)**2
            ind2 = (j-1)/2*(nx-1)/2
            ind = ind1+ind2+i
            
            return ind
        
        










































        
