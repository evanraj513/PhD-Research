#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:16:47 2020

@author: evanraj
"""

    def res_func(self,val):
        '''
        Returns the residual for a given guess of M
        
        value must be an array of proper size. 
        '''
        
        M_old, B_old, B_new, H_s = self.M_old, self.B_old, self.B_new, self.H_s
        dt = self.dt
        mu0 = self.mu0
        gamma = self.gamma
        K = self.K
        alpha = self.alpha
        
        M_new_values = val
        
        #####
        P = 0
        #####
        
        M_on = (M_new_values + M_old.values)/2
        B_on = (B_old.values + B_new.values)/2
        
        if type(val) != np.array:
            print('Error, function cannot be evaluated for this input. Abort')
            raise Exception
            
        elif val.shape != M_old.values.shape:
            print('Error in input size. Abort')
            raise Exception
        
        a =1/dt*(M_new_values - M_old.values)
        b = abs(gamma)*( (1/mu0)*B_on - M_on + H_s.values + K*P*M_on)
        
        bt = np.cross(b.T,M_on.T).T
        
        c = alpha/np.norm(M_on) * M_on
        ct = np.cross(c.T, a.T).T
        
        print('Warning: Still undergoing work. P needs to be further developed')
        
        return a-(bt+ct)


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
        
    
    
    
    
    
### Code for removing boundary nodes
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
        
        
    
### Potentially unusable code for 3d to 2d conversion of vector
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
        
        
        
        ## Indexing Code for odd grid style
    
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
        
                