#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:50:32 2019

@author: evanraj
"""

import numpy as np
#import scipy as sp
from scipy.sparse import lil_matrix

''' TODO
    1. Reverse index function
    '''


class Field(object):
    ''' 
    This class will store the values of a given field, as three vectors
    The size will be determined by initialization
    where each component is located will be based on my_ind()
    It will be F = <F_x, F_y, F_z>
    
    Attributes:
    -----------
    index: function
        Converts (x,y,z) to k component for the vector fed into it
    size: np_array of length 3
        Contains the number of nodes in the x,y,z direction, in that order
    disc: np_array of length 3
        Contains the uniform discretization in the x,y,z direction, in that order
    x: Vector
        stores the first component of the field
    y,z similar to x
    
    TODO:
        - Add in setter properties for x,y,z? This will force the user
            to use the Vector class...
            
        - Boundary conditions need to be implemented. Think more about how
            to implement these
            
    '''
    
    def __init__(self,size,disc,x,y,z):
#        self.index = index
        self.size = size
        self.disc = disc
        self.x = Vector(x,self.size,self.disc)
        self.y = Vector(y,self.size,self.disc)
        self.z = Vector(z,self.size,self.disc)
        
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self,ind):
        self._index = ind
        self.x.index = ind
        self.y.index = ind
        self.z.index = ind
        
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
    - Add in check that values being given to Vector are always a vector, not a matrix
    
        Somewhat done?
    
    - As of now, first slice is included? Shouldn't be, but necessary for reduction to 2D? 
    - Think about possibly just saving the dX,dY,dZ matrices? And then just reusing those, 
        rather then generating them over and over?
    
    Questions:
    ----------
    Can the index function be changed? Or will I just have to assign this elsewhere? 
        -Think more about this
        Ans: Index function can be changed. Once the Vector is created, it's as simple as reassigning 
            the index function. This will also change the derivatives to be based upon the new
            index function. For now, I am writing the derivative functions solely for the ferro-
            magnetic paper, as the derivatives aren't a square matrix. Maybe this will also carry over
            as the number of nodes in the array d_/dx should be smaller than the original array of the 
            Vector. 
        
    Comments:
    ---------
    If using another index function, need to make sure the only inputs are (x,y,z)
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

#    @property
#    def nx(self):
#        return self._nx
#    @nx.setter
#    def nx(self,val):
#        self._nx = val
#        if val == 1:
#            print('Error, change discretization so 2D is in z-axis')
#        elif val == 2:
#            print('Error, discretization not available')
#        else:
#            pass   
#        
#    @property
#    def ny(self):
#        return self._ny
#    @ny.setter
#    def ny(self,val):
#        self._ny = val
#        if val == 1:
#            print('Error, change discretization so 2D is in z-axis')
#        elif val == 2:
#            print('Error, discretization not available')
#        else:
#            pass 
#        
#    @property
#    def nz(self):
#        return self._nz
#    @nz.setter
#    def nz(self,val):
#        self._nz = val
#        if val == 1:
#            print('Reduction to 2-dimensions apparent. \n',
#                  'Altering derivatives. Dz derivative no ',
#                  'longer available')
#        elif val == 2:
#            print('Discretization not available, please define nz == 1 or >= 3 only')
#        else:
#            pass
        
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
        
        return val
    
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
    
    TODO:
        - (done) Add in setter properties for x,y,z? This will force the user
            to use the Vector class.
            
        - Boundary conditions need to be implemented. Think more about how
            to implement these
            
    '''
    
    def __init__(self,size_x, size_y, size_z,disc,values):
#        self.index = index
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.disc = disc
        self.values = values
#        self.x = values[0]
#        self.y = values[1]
#        self.z = values[2]
        
#    @property
#    def index(self):
#        return self._index
#    @index.setter
#    def index(self,ind):
#        self._index = ind
#        self.x.index = ind
#        self.y.index = ind
#        self.z.index = ind
        
        self.E = False
        
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
        
    def conv_DEx_to_Bz(self,arr):
        '''
        Removes first and last slice of an derivative approximation of Ex or Ey
        so that it's size matches Bz
        '''
        
        m = self.x.nx*self.y.ny
        f_s = np.arange(0,m)
        l_s = np.arange(arr.shape[0]-m, arr.shape[0])
        
        obj = np.concatenate([f_s,l_s])
        
        arr = np.delete(arr, [obj],axis=0)
#        print(arr,'\n','f_s: ',f_s, 'l_s: ',l_s, 'obj: ',obj)
        
        return arr
    
    def conv_DEz_to_By(self,arr):
        '''
        Removes boundary derivative approximations of Ez w.r.t x
        on rows of inner slices
        
        Can also be used to reduce z-derivative approximations of Ex
        
        Note: DEz w.r.t. x now lives on Ex nodes \pm a slice. So need to
        get rid of anywhere Ex is on a boundary. 
        '''
        
        Ez = self.z
        Ex = self.x
        rem = np.array([])
        
        for ll in np.arange(0,Ez.nz):
            for kk in np.arange(0,Ex.ny):
                if kk == 0 or kk == Ex.ny-1:
                    ind = kk*Ex.nx+ll*Ex.nx*Ex.ny
                    f_r = np.arange(ind,ind+Ex.nx)
                    rem = np.concatenate([rem, f_r])
#                    print('ind: ',ind,'\n',
#                          'll: ',ll,'\n',
#                          'kk: ',kk,'\n',
#                          'rem: ',rem,'\n')
                    
        arr = np.delete(arr, [rem], axis=0)
        
        return arr
    
    def conv_DEz_to_Bx(self,arr):
        '''
        Removes boundary derivative approximations of Ez w.r.t y
        on columns of inner slices
        
        Can also be used to reduce z-derivative approximations of Ey
        
        Note: DEz w.r.t. y now lives on Ey nodes \pm a slice. So need to
        get rid of anywhere Ey is on a boundary. 
        '''
        
        Ez = self.z
        Ey = self.y
        rem = np.array([])
        
        for ll in np.arange(0,Ez.nz):
            for kk in np.arange(0,Ey.ny):
                for jj in np.arange(0,Ey.nx):
                    if jj == 0 or jj == Ey.nx-1:
                        ind = np.arange(jj + kk*Ey.nx + ll*Ey.nx*Ey.ny,
                                        jj + kk*Ey.nx + ll*Ey.nx*Ey.ny+1)
                        rem = np.concatenate([rem, ind])
                            
        arr = np.delete(arr, [rem], axis=0)
        
        return arr
    
#    def conv_DBx_to_Ez(self,arr):
#        ''' 
#        Add the empty first and last rows (or columns, same number) 
#        to each slice of derivative approximations of Bx,By
#        so that this can be added to Ez properly
#        '''
#        Bx = self.x
#        Bz = self.z
#        
#        for ll in np.arange(0,Bx.nz):
#            for kk in np.arange(0,Bx.ny+2):
#                for jj in np.arange(0,Bx.nx+2):
#                    if jj == 0 or jj == Bx.nx+1:
##                        add = np.zeros((Bx.nx+2,1)).T
#                        ind = 
#                        ind_total = np.concatenate([ind_total, ind])
#                        
#                        
        
        
    def curl(self):
        '''
        Finds an approximation to the curl, given the index and x,y,z values
        in Cartesian coordinates
        
        Returns as three vectors. Can be used to generate a field 
        '''
        
        if self.E == True:
            curl_x = self.conv_DEz_to_Bx(self.z.Dy()) - self.conv_DEz_to_Bx(self.y.Dz())
            curl_y = self.conv_DEz_to_By(self.y.Dz()) - self.conv_DEz_to_By(self.z.Dx())
            curl_z = self.conv_DEx_to_Bz(self.y.Dx()) - self.conv_DEx_to_Bz(self.x.Dy())
        else:
        
            curl_x = self.z.Dy() - self.y.Dz()
            curl_y = self.x.Dz() - self.z.Dx()
            curl_z = self.y.Dx() - self.x.Dy()
        
        return np.array([curl_x,curl_y,curl_z])

            
        
    
        
        
        
        