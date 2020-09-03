#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:50:32 2019

@author: evanraj
"""

import numpy as np
#import scipy as sp
from scipy.sparse import lil_matrix
    
class Vector(object):
    ''' 
    This will be the storing on one vector. Could be a component of a field, 
        or stand alone. 
        
    -- Think F = <F_x, F_y, F_z>
        This will be either F_x or F_y or F_z
    
    It will 
    
    Attributes:
    -----------
    value: np.ndarray
        Value for the vector as stored
    nodal_count: np.ndarray 
        Stores the nodal count in any direction
        - If length == 1, uniform
        - Elif length == 3, [nx ny nz] storage
        *** Note that a Field requires the length to be 3 ***
    nx: float
        Number of nodes in a single row (x-direction)
    ny: float
        Number of nodes in a single column (y-direction)
    nz: float
        Number of nodes in z-direction (z-direction)
    d_: float
        Step-size in _-direction
    index: function
        Converts (x,y,z) into (nx, ny, nz) specific to this Vector
    length: int
        Length of array being stored
        
    TODO:
    -----
    - (done) Add in check that values being given to Vector are always a np.arrray of size(x,1), not a matrix    
    - Think about possibly just saving the dX,dY,dZ matrices? And then just reusing those, 
        rather then generating them over and over?
    **  Ans: Still not sure how, but good to think about. 
    
    Questions:
    ----------
    Can the index function be changed? Or will I just have to assign this elsewhere? 
        Ans: Index function can be changed. Once the Vector is created, it's as simple as reassigning 
            the index function. This will also change the derivatives to be based upon the new
            index function. For now, I am writing the derivative functions solely for the ferro-
            magnetic paper, as the derivatives aren't a square matrix. Maybe this will also carry over
            as the number of nodes in the array d_/dx should be smaller than the original array of the 
            Vector. 
        
    Comments:
    ---------
    If using another index function, make sure the only inputs are (x,y,z)
    and that nx,ny,nz are used as fixed parameters, to maintain consistency. 
    '''
    
    def __init__(self,nodal_count,disc,init_value):
        size = nodal_count
        if type(nodal_count) != np.ndarray:
            print('Warning: Not a numpy array. Break')
            raise Exception
        if nodal_count.size == 3:
            self.nx = size[0]
            self.ny = size[1]
            self.nz = size[2]
        elif nodal_count.size == 1:
            self.nx = size[0]
            self.ny = size[0]
            self.nz = size[0]
        
        self.value = init_value
        
        self.dx = disc[0]
        self.dy = disc[1]
        self.dz = disc[2]
        
        self.index = self.myind_std
        self.index_rev = self.myind_std_rev
    
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self,val):
        self._value = val
        if type (val) == np.ndarray:
            self.length = val.size 
            if val.shape[0] != self.nx*self.ny*self.nz:
                print('Error. Incorrect dimensions or value given. Abort')
                print('Initiating value array length: ',val.shape[0],'\n'
                      ,'nx: ',self.nx,'\n'
                      ,'ny: ',self.ny,'\n'
                      ,'nz: ',self.nz,'\n'
                      ,'product: ',self.nx*self.ny*self.nz)
                print('Note: proper initiation requires \n Initiating value array length = product')
                raise Exception
            
        else:
            print('Error. Value not given as a numpy array. Abort')
            raise Exception 
        
    def myind_std(self,x,y,z):
        '''
        Returns index of value in vector associated with a given (x,y,z) value
        This is a standard ordering index, with each slice counted first, and then 
        cont'd on to the next. See below for first slice ordering.
        
        index_value
           .
         (x,y)
         
         
        
          2      3
          .      .
        (1,0)  (1,1)  
          
          0      1              
          .      .
        (0,0)  (1,0)
        
        '''
        
        j = x/self.dx
        k = y/self.dy
        l = z/self.dz
        
        val = j + k*(self.nx) + l*(self.nx*self.ny)
        
#        print('j: ',j,'\n',
#              'k: ',k,'\n',
#              'l: ',l)
        
        return np.int(np.round(val))
    
    def myind_std_rev(self, m):
        '''
        Returns the (x,y,z) coordinate of some node, assuming standard grid
        as used in myind_std. 
        '''
        nx = self.nx
        ny = self.ny
        nz = self.nz
        
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        
        l = (m-m%(nx*ny))/(nx*ny)
        m2 = m-l*nx*ny
        k = (m2-m2%nx)/nx
        j = m2-k*nx
        if l >= nz or k >= ny or j >= nx:
            print('Error, node not within array. Abort')
            print('l: ',l,' nz: ',nz,'\n',
                  'k: ',k,' ny: ',ny,'\n',
                  'j: ',j,' nx: ',nx,'\n'
                  'm: ',m)
            raise Exception        
        
        return np.array([j*dx,k*dy,l*dz])
        
    
    def Dx(self):
        '''
        Gives vector approximation to derivative w.r.t. x
                
        '''
        ind = self.myind_std # See Running notes p.20 for more explanation
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        if self.nz != 1:
            row_num = (self.nx-1)*self.ny*self.nz
        elif self.nz == 1 and self.nx > self.ny:
            row_num = (self.nx-1)*(self.ny)*(self.nz)
        elif self.nz == 1 and self.ny > self.nx:
            row_num = (self.nx)*(self.ny-1)*(self.nz)
            '''
            This is done as the Ef_z is assumed to match the Ef_x layout
            and a sizing issue becomes apparent if this is not done. 
            '''
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        
        if self.nz !=1:
            for ll in range(0,self.nz-1): #Moving through each inner slice
                for kk in range(0,self.ny-1): #Moving through each inner row
                    for jj in range(1,self.nx-1): #Moving through each inner node
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
        else:
            for ll in range(0,self.nz): #Moving through each inner slice
                for kk in range(0,self.ny-1): #Moving through each inner row
                    for jj in range(1,self.nx-1): #Moving through each inner node
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
                        
        A1 = Al.tocsc()
        
        return A1*self.value
    
   
    
    def Dy(self):
        '''
        Gives vector approximation to derivative w.r.t. y                
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        row_num = self.nx*(self.ny-1)*self.nz
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        if self.nz != 1:
            for ll in range(0,self.nz-1): #Moving through each inner slice
                for kk in range(1,self.ny-1): #Moving through each inner row
                    for jj in range(0,self.nx-1): #Moving through each inner node
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
        else:
            for ll in range(0,self.nz): #Moving through only slice
                for kk in range(1,self.ny-1): #Moving through each inner row
                    for jj in range(0,self.nx-1): #Moving through each inner node
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
                        
        A1 = Al.tocsc()
        
        return A1*self.value
    
    def Dz(self):
        '''
        Gives vector approximation to derivative w.r.t. z               
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        if self.nz != 1:
            row_num = self.nx*self.ny*(self.nz-1)
        elif self.nz == 1 and self.nx > self.ny:
            row_num = (self.nx-1)*(self.ny)*(self.nz)
        elif self.nz == 1 and self.ny > self.nx:
            row_num = (self.nx)*(self.ny-1)*(self.nz)
            
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        if self.nz!=1:    
            for ll in range(1,self.nz-1): #Moving through each slice
                for kk in range(0,self.ny-1): #Moving through each row
                    for jj in range(0,self.nx-1): #Moving through each node
                        diff = self.nx #Could also be ny I believe
                        
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll)*dz)+kk+diff*ll] = -1/(2*dz)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)+kk+diff*ll] = 1/(2*dz)
        else:
            for ll in range(1,self.nz): #Moving through only slice
                for kk in range(0,self.ny-1): #Moving through each row
                    for jj in range(0,self.nx-1): #Moving through each node
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll)*dz)+kk] = -1/(2*dz)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)+kk] = 1/(2*dz)
        A1 = Al.tocsc()
        
        return A1*self.value
    
    def ind_dx(self,x,y,z):
        '''
        Returns the Bz (for Ey) or By for (Ex) node to the *left* of given
        (x,y,z), to be used solely for the Dx_E
        
        Note that the sizing is assumed to be:
            E_nx = B_nx+1
            E_ny = B_ny
            E_nz = B_nz
        
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*(nx-1) + l*(nx-1)*ny
        return np.int(np.round(val))
    
    
    def Dx_E_mat(self):
        '''
        Gives matrix operator to compute derivative of E w.r.t. x, 
        based on the following scheme
                
            .   x      ...
          E_nx+1
        ^     E_nx
        |
        y   .  x   .  x   .  x   . ... x   .
        x-> 0      1      2      3        E_nx (E nodes)
               0      1      2       (E_nx-1)  (DE/Dx nodes)
               
        Note: This is to be used for only Ey, Ez. Will not work for Ex
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
        
        ind = self.ind_dx
        if nz!=1:
            if ny > nz:
                row_num = nx*(ny-1)*nz
            elif nz > ny:
                row_num = nx*ny*(nz-1)
        else:
            if nx > ny:
                row_num = (nx-1)*ny
            elif ny > nx:
                row_num = nx*(ny-1)
        
            '''
            This is done as the Ef_z is assumed to match the Ef_x layout
            and a sizing issue becomes apparent if this is not done. 
            '''
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        
        for ll in np.arange(0,nz):
           for kk in np.arange(0,ny):
               for jj in np.arange(0,nx-1):
                   #abs(Ey.ny*Ey.nx - Bz.nx*Bz.ny) = abs(Ey.ny*Ey.nx - Ey.ny*(Ey.nx-1))
                   diff = ny ## Adjust for each slice number discrepancy. 
                   diff2 = 1 #(Ey.nx - Bz.nx), adjust for each row wrapping
                   Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,ll*dz)+ll*diff+kk*diff2] = -1/(2*dx)
                   Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)+ll*diff+kk*diff2] = 1/(2*dx)
#                   print(jj,kk,ll,\
#                            ind_y3(jj*dx,kk*dy,ll*dz),\
#                            ind_y3(jj*dx,kk*dy,ll*dz)+ll*diff+kk*diff2,\
#                            ind_y3((jj+1)*dx,kk*dy,ll*dz)+ll*diff+kk*diff2)
                   
        A1 = Al.tocsc()
        
        return A1
    
    def Dx_E(self):
        '''
        Returns derivative of E w.r.t. x
        '''
        A1 = self.Dx_E_mat()
        return A1*self.value
        

    def ind_dy(self,x,y,z):
        '''
        Returns the Bz (for Ex) or Bx for (Ez) node to *above* given
        (x,y,z), to be used solely for the Dy_E
        
        Note that the sizing is assumed to be:
            E_nx = B_nx
            E_ny = B_ny+1
            E_nz = B_nz
        
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*nx + l*nx*(ny-1)
        return np.int(np.round(val))
                   
    def Dy_E_mat(self):
        '''
        Gives matrix operator to approximates derivative of E w.r.t. y, 
        based on the following scheme
                
            .   x      ...
            1
        ^       1
        |
        x   .  x   .     x      .    ...       x        .
        y-> 0    E_nx+1      2*(E_nx+1)            (E_ny)(E_nx+1)     (E nodes)
               0      E_nx+1              (E_ny)(E_nx+1)              (DE/Dy nodes)
               
        Note: This is to be used for only Ex, Ez. Will not work for Ey
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
        
        ind = self.ind_dy
        if nz!=1:
            if nx > nz:
                row_num = (nx-1)*ny*nz
            elif nz > nx:
                row_num = nx*ny*(nz-1)
        else:
            if nx > ny:
                row_num = (nx-1)*ny
            elif ny > nx:
                row_num = nx*(ny-1)
            '''
            This is done as the Ef_z is assumed to match the Ef_x layout
            and a sizing issue becomes apparent if this is not done. 
            '''
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        
        for ll in np.arange(0,nz):
            for kk in np.arange(0,ny-1):
                for jj in np.arange(0,nx):
                    #abs(Ey.ny*Ey.nx - Bz.nx*Bz.ny) = abs(Ey.ny*Ey.nx - Ey.nx*(Ey.ny-1))
#                    print(jj,kk,ll,\
#                    ind(jj*dx,kk*dy,ll*dz),\
#                    ind(jj*dx,kk*dy,ll*dz),\
#                    ind(jj*dx,(kk+1)*dy,ll*dz))
#                    
#                    wait = input('Press Enter to continue')
                    
                    diff = nx 
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,ll*dz)+ll*diff] = -1/(2*dy)
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)+ll*diff] = 1/(2*dy)
                    
        A1 = Al.tocsc()
        
        return A1
    
    def Dy_E(self):
        '''
        Returns derivative w.r.t. y of E_x, E_z
        '''
        A1 = self.Dy_E_mat()
        return A1*self.value
    
    
    def ind_dz(self,x,y,z):
        '''
        Returns the Bz (for Ex) or Bx for (Ez) node to *above* given
        (x,y,z), to be used solely for the Dy_E
        
        Note that the sizing is assumed to be:
            E_nx = B_nx
            E_ny = B_ny+1
            E_nz = B_nz
        
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*nx + l*nx*ny
        return np.int(np.round(val))
    
    def Dz_E_mat(self):
        '''
        Gives matrix operator that approximates derivative of E w.r.t. z,
        based on the similar scheme to above
               
        Note: This is to be used for only Ex, Ez. Will not work for Ey
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
        
        ind = self.ind_dz
        
        if nx > ny:
            row_num = (nx-1)*ny*nz
        elif ny > nx:
            row_num = nx*(ny-1)*nz
        
#        if self.nz != 1:
#            row_num = nx*(ny-1)*nz
#        elif self.nz == 1 and self.nx > self.ny:
#            row_num = (self.nx-1)*(self.ny)*(self.nz)
#        elif self.nz == 1 and self.ny > self.nx:
#            row_num = (self.nx)*(self.ny-1)*(self.nz)
            '''
            This is done as the Ef_z is assumed to match the Ef_x layout
            and a sizing issue becomes apparent if this is not done. 
            '''
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        
        for ll in np.arange(0,nz-1):
            for kk in np.arange(0,ny):
                for jj in np.arange(0,nx):
#                    print(jj,kk,ll,\
#                    ind(jj*dx,kk*dy,ll*dz),\
#                    ind(jj*dx,kk*dy,ll*dz),\
#                    ind(jj*dx,(kk+1)*dy,ll*dz))
#                    
#                    wait = input('Press Enter to continue')
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,ll*dz)] = -1/(2*dz)
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)] = 1/(2*dz)
                    
        A1 = Al.tocsc()
        
        return A1
        
    def Dz_E(self):
        '''
        Returns derivative w.r.t. to z of E_x, E_y
        '''
        A1 = self.Dy_E_mat()
        return A1*self.value
    
    def Dx_B_v1(self):
        '''
        Gives vector approximation to derivative w.r.t. x for By, Bz
        but includes the boundary nodes of E
        
        These boundary nodes will lie on the x == 0, x == gnx-1 nodes
        where gnx is the global node count in the x-direction. 
        
        Note: the 'zero-values' being added = 2*nx*ny, for a uniform grid
                
        '''
        ind = self.myind_std # See Running notes p.20 for more explanation
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
                    # Interior 'actual' values     zero values
        row_num = int((self.nx-1)*self.ny*self.nz + 2*self.ny*self.nz)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        if self.nz != 1:
            for ll in range(0,self.nz-1): #Moving through each inner slice
                for kk in range(0,self.ny-1): #Moving through each inner row
                    for jj in range(0,self.nx): #Moving through each inner node
                        if jj == 0 or jj == self.nx-1:
                            pass
                        else:
                            Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
        else:
            for ll in range(0,self.nz): #Moving through each inner slice
                for kk in range(0,self.ny-1): #Moving through each inner row
                    for jj in range(0,self.nx): #Moving through each inner node
                        if jj == 0 or jj == self.nx-1:
                            pass
                        else:
                            Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
                    
        A1 = Al.tocsc()
        
        return A1*self.value
    
    def ind_dx_B(self,x,y,z):
        '''
        Returns the Ey (for Bz) or Ex for (By) node to the *left* of given
        (x,y,z), to be used solely for the Dx_B
        
        Note that the sizing is assumed to be:
            E_nx = B_nx+1
            E_ny = B_ny
            E_nz = B_nz
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*(nx+1) + l*(nx+1)*ny
        return np.int(np.round(val))
    
    
    #### Correction as done in Dx_E but reverse?
    def Dx_B_mat(self):
        '''
        Gives vector approximation to derivative w.r.t. x for By, Bz
        but includes the boundary nodes of E
        
        These boundary nodes will lie on the x == 0, x == gnx-1 nodes
        where gnx is the global node count in the x-direction. 
        
        Note: the 'zero-values' being added = 2*nx*ny, for a uniform grid
                
        '''
        ind = self.ind_dx_B
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
        
                    # Interior 'actual' values  +  zero values
        row_num = int((nx-1)*ny*nz + 2*ny*nz)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        
        for ll in range(0,self.nz): #Moving through each inner slice
            for kk in range(0,self.ny): #Moving through each inner row
                for jj in range(0,self.nx+1): #Moving through each inner node
                    if jj == 0 or jj == self.nx:
                        pass
                    else:
                        if nz < ny:
                            diff = nz
                        elif ny < nz:
                            diff = ny
                        diff2 = 1
#                        print(jj,kk,ll,\
#                            ind(jj*dx,kk*dy,ll*dz),\
#                            ind((jj-1)*dx,kk*dy,ll*dz)-ll*diff-kk*diff2,\
#                            ind(jj*dx,kk*dy,ll*dz)-ll*diff-kk*diff2)
#                            
#                        wait = input('Press Enter to continue')
                        
                        Al[ind(jj*dx,kk*dy,ll*dz)\
                           ,ind((jj-1)*dx,kk*dy,ll*dz)-ll*diff-kk*diff2] = -1/(2*dx)
                        Al[ind(jj*dx,kk*dy,ll*dz),\
                           ind(jj*dx,kk*dy,ll*dz)-ll*diff-kk*diff2] = 1/(2*dx)
                    
        A1 = Al.tocsc()
        
        return A1
    
    def Dx_B(self):
        '''
        you know
        '''
    
        A1 = self.Dx_B_mat()
        return A1*self.value
    
    def Dy_B_v1(self):
        '''
        Gives vector approximation to derivative w.r.t. y for Bx, Bz
        but includes the boundary nodes of E              
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz

                    # Interior 'actual' values     zero values        
        row_num = int(self.nx*(self.ny-1)*self.nz + 2*self.nx*self.nz)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        if self.nz != 1:
            for ll in range(0,self.nz-1): #Moving through each inner slice
                for kk in range(0,self.ny): #Moving through each inner row
                    for jj in range(0,self.nx-1): #Moving through each inner node
                        if kk == 0 or kk == self.ny-1:
                            pass
                        else:
    #                        print('a: ',ind(jj*dx,kk*dy,ll*dz),'\n'
    #                              'b: ',ind(jj*dx,(kk+1)*dy,ll*dz),'\n'
    #                              'c: ',ind(jj*dx,(kk-1)*dy,ll*dz))
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
        else:
            for ll in range(0,self.nz): #Moving through each inner slice
                for kk in range(0,self.ny): #Moving through each inner row
                    for jj in range(0,self.nx-1): #Moving through each inner node
                        if kk == 0 or kk == self.ny-1:
                            pass
                        else:
    #                        print('a: ',ind(jj*dx,kk*dy,ll*dz),'\n'
    #                              'b: ',ind(jj*dx,(kk+1)*dy,ll*dz),'\n'
    #                              'c: ',ind(jj*dx,(kk-1)*dy,ll*dz))
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
                        
        A1 = Al.tocsc()
        
        return A1*self.value
    
    def ind_dy_B(self,x,y,z):
        '''
        Returns the Ex (for Bz) or Ez for (Bx) node to *above* given
        (x,y,z), to be used solely for the Dy_E
        
        Note that the sizing is assumed to be:
            E_nx = B_nx
            E_ny = B_ny+1
            E_nz = B_nz
        
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*nx + l*nx*(ny+1)
        return np.int(np.round(val))
    
    ### Correction in progress
    def Dy_B_mat(self):
        '''
        Gives vector approximation to derivative w.r.t. y for Bx, Bz
        but includes the boundary nodes of E              
        '''
        ind = self.ind_dy_B
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz

                    # Interior 'actual' values     zero values        
        row_num = int(self.nx*(self.ny-1)*self.nz + 2*self.nx*self.nz)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        
        for ll in range(0,self.nz): #Moving through each inner slice
            for kk in range(0,self.ny+1): #Moving through each inner row
                for jj in range(0,self.nx): #Moving through each inner node
                    if kk == 0 or kk == self.ny:
                        pass
                    else:
                        if nx > nz:
                            diff = nz
                        elif nz > nx:
                            diff = nx

                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)-ll*diff] = -1/(2*dy)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,ll*dz)-ll*diff] = 1/(2*dy)
        
                        
        A1 = Al.tocsc()
        
        return A1
    
    def Dy_B(self):
        A1 = self.Dy_B_mat()
        return A1*self.value
    
    def Dz_B_v1(self):
        '''
        Gives vector approximation to derivative w.r.t. z for Bx, By
        but includes the boundary nodes of E          
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz

        if self.nz != 1:
                        # Interior 'actual' values     zero values        
            row_num = int(self.nx*self.ny*(self.nz-1) + 2*self.nx*self.ny)
        else:
            '''
            If nz = 1, then no dz derivatives will actually matter. 
            This is merely a place-holder for the zeros, so I can 
            continue to use my other code more efficiently, i.e. curl etc. 
            '''
            row_num = int(self.nx*(self.ny-1)*(self.nz) + 2*self.nx)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        
        if self.nz != 1:
            for ll in range(0,self.nz): #Moving through each slice
                for kk in range(0,self.ny-1): #Moving through each row
                    for jj in range(0,self.nx-1): #Moving through each node
                        if ll == 0 or ll == self.nz-1:
                            pass
                        else:
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)] = 1/(2*dz)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll-1)*dz)] = -1/(2*dz)
        else: 
            for ll in range(0,self.nz): #Moving through each slice
                for kk in range(0,self.ny-1): #Moving through each row
                    for jj in range(0,self.nx-1): #Moving through each node
                        if ll == 0 or ll == self.nz-1:
                            pass
                        else:
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)] = 1/(2*dz)
                            Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll-1)*dz)] = -1/(2*dz)
                        
        A1 = Al.tocsc()
            
        return A1*self.value
    
    def ind_dz_B(self,x,y,z):
        '''
        Returns the Ex (for By) or Ey for (Bx) node to *above* given
        (x,y,z), to be used solely for the Dy_E
        
        Note that the sizing is assumed to be:
            E_nx = B_nx
            E_ny = B_ny
            E_nz = B_nz+1
        
        
        '''
        dx = self.dx
        dy = self.dy
        dz = self.dz

        j = x/dx
        k = y/dy
        l = z/dz
        nx = self.nx
        ny = self.ny
        val = j + k*nx + l*nx*ny
        return np.int(np.round(val))
    
    def Dz_B_mat(self):
        '''
        Gives vector approximation to derivative w.r.t. z for Bx, By
        but includes the boundary nodes of E          
        '''
        ind = self.ind_dz_B
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        nx = self.nx
        ny = self.ny
        nz = self.nz

        if self.nz != 1:
                        # Interior 'actual' values     zero values        
            row_num = int(self.nx*self.ny*(self.nz-1) + 2*self.nx*self.ny)
        else:
            '''
            If nz = 1, then no dz derivatives will actually matter. 
            This is merely a place-holder for the zeros, so I can 
            continue to use my other code more efficiently, i.e. curl etc. 
            '''
            row_num = int(self.nx*(self.ny-1)*(self.nz) + 2*self.nx)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        

        for ll in range(0,self.nz+1): #Moving through each slice
            for kk in range(0,self.ny): #Moving through each row
                for jj in range(0,self.nx): #Moving through each node
                    if ll == 0 or ll == self.nz:
                        pass
                    else:
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll-1)*dz)] = -1/(2*dz)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,ll*dz)] = 1/(2*dz)
                 
        A1 = Al.tocsc()
            
        return A1
    
    def Dz_B(self):
        
        A1 = self.Dz_B_mat()
        
        return A1*self.value
    
        
        
class Field_v3(object):
    ''' 
    This class will store the values of a given field, as three vectors
    The size will be determined by initialization
    where each component is located will be based on my_ind()
    
    It will be F = <F_x, F_y, F_z>
    
    Attributes:
    -----------
    node_grid: dict
        Contains the number of nodes in the x,y,z direction of each spatial 
        direction of the field
    disc: np_array of length 3
        Contains the spatial step-size
        in the x,y,z direction, in that order
    x: Vector
        stores the first component of the field
    y,z similar to x
    
    vals: list of 3 np.arrays of 1 by n
        Stores the values to initialize the Vectors x,y,z respectively. 
    
    TODO:
            
    '''
    
    def __init__(self,node_grid = {}, disc = [], vals = []):
#        self.index = index
        if node_grid['x'].size !=3 or node_grid['y'].size !=3 or node_grid['z'].size !=3:
            print('*'*40,'\n','Error in field initialization.',
                  'Incorrect nodal_grid given. Abort. ')
            raise Exception
        if disc.size !=3:
            print('*'*40,'\n','Error in field initialization.',
                  'Incorrect discretization given. Abort. ')
            raise Exception        
        
        self.node_grid_x = np.array(node_grid['x'])
        self.node_grid_y = np.array(node_grid['y'])
        self.node_grid_z = np.array(node_grid['z'])
        self.disc = disc

        # self.values = values
        
        if type(vals[0]) != np.ndarray:
            print('Error with Field initialization. x not np array')
            raise Exception
            
        if type(vals[1]) != np.ndarray:
            print('Error with Field initialization. y not np array')
            raise Exception
            
        if type(vals[2]) != np.ndarray:
            print('Error with Field initialization. z not np array')
            raise Exception
        
        self.x = vals[0]
        self.y = vals[1]
        self.z = vals[2]
        self.values = vals
        
    @property 
    def x(self):
        return self._x
    @x.setter
    def x(self, val):
        self._x = Vector(self.node_grid_x, self.disc, val)
        
    @property 
    def y(self):
        return self._y
    @y.setter
    def y(self, val):
        self._y = Vector(self.node_grid_y, self.disc, val)
        
    @property 
    def z(self):
        return self._z
    @z.setter
    def z(self, val):
        self._z = Vector(self.node_grid_z, self.disc, val)
        
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, vals):
        self._values = vals
        self.x = vals[0]
        self.y = vals[1]
        self.z = vals[2]
        
        # return np.array(vals)
        
        

    def curl(self):
        '''
        Finds an approximation to the curl, given the index and x,y,z values
        in Cartesian coordinates
        
        Returns as three vectors. Can be used to generate a field 
        '''
        
        curl_x = self.z.Dy() - self.y.Dz()
        curl_y = self.x.Dz() - self.z.Dx()
        curl_z = self.y.Dx() - self.x.Dy()
        
        return np.array([curl_x,curl_y,curl_z])
    
