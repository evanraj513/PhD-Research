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
        or a stand alone thing
        
    -- Think F = <F_x, F_y, F_z>
        This will be either F_x or F_y or F_z
    
    It will 
    
    Attributes:
    -----------
    value: np.ndarray
        Value for the vector as stored
    nx: float
        Number of nodes in a single row
    ny: float
        Number of nodes in a single column
    nz: float
        Number of nodes in z-direction       
    d_: float
        Step-size in _-direction
    index: function
        Function that is used to order spatial nodes into one vector
    length: int
        Length of array being stored
        
    TODO:
    -----
    - (done) Add in check that values being given to Vector are always a np.arrray of size(x,1), not a matrix    
    - As of now, first slice is included? Shouldn't be, but necessary for reduction to 2D? 
        Ans: With current code, first slice is always required
    - Think about possibly just saving the dX,dY,dZ matrices? And then just reusing those, 
        rather then generating them over and over?
        
        Ans: Still not sure how, but good to think about. 
    
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
    
    def __init__(self,size,disc,init_value):
        self.nx = size[0]
        self.ny = size[1]
        self.nz = size[2]
        
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
                print('size of value: ',val.shape[0],'\n'
                      ,'nx: ',self.nx,'\n'
                      ,'ny: ',self.ny,'\n'
                      ,'nz: ',self.nz,'\n'
                      ,'product: ',self.nx*self.ny*self.nz)
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
         
         
        
          3      4
          .      .
        (1,0)  (1,1)  
          
          1      2              
          .      .
        (0,0)  (1,0)
        
        '''
        
        j = x/self.dx
        k = y/self.dy
        l = z/self.dz
        
        val = j + k*self.nx + l*self.nx*self.ny
        
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
        
        row_num = (self.nx-1)*self.ny*self.nz
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        for ll in range(0,self.nz-1): #Moving through each inner slice
            for kk in range(0,self.ny-1): #Moving through each inner row
                for jj in range(1,self.nx-1): #Moving through each inner node
                    Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                    Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
                    
        A1 = Al.tocsr()
        
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
        for ll in range(0,self.nz-1): #Moving through each inner slice
            for kk in range(1,self.ny-1): #Moving through each inner row
                for jj in range(0,self.nx-1): #Moving through each inner node
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
                        
        A1 = Al.tocsr()
        
        return A1*self.value
    
    def Dz(self):
        '''
        Gives vector approximation to derivative w.r.t. z               
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz
        
        row_num = self.nx*self.ny*(self.nz-1)
                        ## rows       columns
        Al = lil_matrix((int(row_num), int(self.length)), dtype='float')
        for ll in range(1,self.nz-1): #Moving through each slice
            for kk in range(0,self.ny-1): #Moving through each row
                for jj in range(0,self.nx-1): #Moving through each node
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)] = 1/(2*dz)
                    Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll-1)*dz)] = -1/(2*dz)
                    
        A1 = Al.tocsr()
            
        return A1*self.value
    
    def Dx_B(self):
        '''
        Gives vector approximation to derivative w.r.t. x for By, Bz
        but includes the boundary nodes of E
        
        These boundary nodes will lie on the x == 0, x == nx-1 nodes
        where nx is the global node count in the x-direction. 
        
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
        for ll in range(0,self.nz-1): #Moving through each inner slice
            for kk in range(0,self.ny-1): #Moving through each inner row
                for jj in range(0,self.nx): #Moving through each inner node
                    if jj == 0 or jj == self.nx-1:
                        pass
                    else:
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj+1)*dx,kk*dy,ll*dz)] = 1/(2*dx)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind((jj-1)*dx,kk*dy,ll*dz)] = -1/(2*dx)
                    
        A1 = Al.tocsr()
        
        return A1*self.value
    
    
    def Dy_B(self):
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
        for ll in range(0,self.nz-1): #Moving through each inner slice
            for kk in range(0,self.ny): #Moving through each inner row
                for jj in range(0,self.nx-1): #Moving through each inner node
                    if kk == 0 or kk == self.ny:
                        pass
                    else:
#                        print('a: ',ind(jj*dx,kk*dy,ll*dz),'\n'
#                              'b: ',ind(jj*dx,(kk+1)*dy,ll*dz),'\n'
#                              'c: ',ind(jj*dx,(kk-1)*dy,ll*dz))
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk+1)*dy,ll*dz)] = 1/(2*dy)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,(kk-1)*dy,ll*dz)] = -1/(2*dy)
                        
        A1 = Al.tocsr()
        
        return A1*self.value
    
    def Dz_B(self):
        '''
        Gives vector approximation to derivative w.r.t. z for Bx, By
        but includes the boundary nodes of E          
        '''
        ind = self.myind_std
        dx = self.dx
        dy = self.dy
        dz = self.dz

                    # Interior 'actual' values     zero values        
        row_num = int(self.nx*self.ny*(self.nz-1) + 2*self.nx*self.ny)
                        ## rows       columns
        Al = lil_matrix((row_num, int(self.length)), dtype='float')
        for ll in range(0,self.nz): #Moving through each slice
            for kk in range(0,self.ny-1): #Moving through each row
                for jj in range(0,self.nx-1): #Moving through each node
                    if ll == 0 or ll == self.nz-1:
                        pass
                    else:
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll+1)*dz)] = 1/(2*dz)
                        Al[ind(jj*dx,kk*dy,ll*dz),ind(jj*dx,kk*dy,(ll-1)*dz)] = -1/(2*dz)
                    
        A1 = Al.tocsr()
            
        return A1*self.value
    
        
        
class Field_2(object):
    ''' 
    This class is an update to the Field class above, for testing purposes. 
    
    ------- In Development -----------
    
    Attributes:
    -----------
    index: function
        Converts (x,y,z) to k component for the vector fed into it
    size_(): np_array of length 3
        Contains the number of nodes in the x,y,z direction, in that order 
        of the () component of the field
    disc: np_array of length 3
        Contains the uniform discretization in the x,y,z direction, in that order
    x: Vector
        stores the first component of the field
    y,z similar to x
    
    values: np.array of 3 by x
        Stores the values to initialize the Vectors x,y,z respectively. 
    
    TODO:
        - (done) Add in setter properties for x,y,z? This will force the user
            to use the Vector class.
            
        - (done) Boundary conditions need to be implemented. Think more about how
            to implement these
            
            Ans: Done in system
            
    '''
    
    def __init__(self,size_x, size_y, size_z,disc,values):
#        self.index = index
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.disc = disc
        self.values = values
        
    @property 
    def x(self):
        return self._x
    @x.setter
    def x(self, val):
        self._x = Vector(self.size_x, self.disc, val)
        
    @property 
    def y(self):
        return self._y
    @y.setter
    def y(self, val):
        self._y = Vector(self.size_y, self.disc, val)
        
    @property 
    def z(self):
        return self._z
    @z.setter
    def z(self, val):
        self._z = Vector(self.size_z, self.disc, val)
                
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, vals):
        self._values = vals
        self.x = vals[0]
        self.y = vals[1]
        self.z = vals[2]
#        if vals.shape[0] == 3 or vals.shape[1] == 3:
#            self.z = vals[2]
#        elif vals.shape[0] == 2  or vals.shape[1] == 2:
#            print('Note, reduction to two-dimensions. In construction')
#        else:
#            print('Error. Values not given in proper format')
        

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
    


            
        
    
        
        
        
        