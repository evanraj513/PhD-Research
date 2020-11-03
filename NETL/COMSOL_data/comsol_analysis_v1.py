#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:50:20 2020

This will function as an analysis tool to work with the COMSOL code. 

It is not designed to work with anything in particular as of yet. It is
    currently being written to give insight into the matrices and vectors
    used in the code, such as the condition number of a matrix, what it actually
    looks like, etc. 
    
Update 10/11/2020

This code now includes the ability to generate plots based on some file-input. 
The goal is to use this code to validate COMSOL's code with Rosa's equations. 
This implies to generate plots where beta_e = beta_i = 0, and vary K,sigma,u_x,B_z
    to check that power out = K(1-K)sigma*u_x^2*B_z^2
    


@author: evanraj
"""

import numpy as np
import pandas as pd
import os
import sys
import time

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
close = plt.close
# plt.rcParams['backend'] = "Qt4Agg"
plt.rcParams['figure.max_open_warning'] = 100

l_path = '..'
m_path = os.path.abspath(l_path)
if not os.path.exists(m_path):
    print('Error importing modules. Need to specify new path')
    raise Exception
else:
    sys.path.append(m_path)

#######################################################3
## Old code to manually inspect condition number of M
# mob_e = 0.1 ## Mobility of electrons
# u_x = 340 # m/s
# B_z = 6 # Tesla
# beta_e = mob_e ## Hall parameter
# beta_i = 1/(B_z**2) ## Ion-slip parameter
# sigma = 60 #(S/m^2), conductivity of plasma

# gamma = (1-beta_i**2*B_z**2)**2+beta_e**2*B_z**2 ## Denominator for M^-1
# M = np.array([np.array([1-beta_i*B_z**2, -beta_e*B_z, 0]),
#               np.array([beta_e*B_z, 1-beta_i*B_z**2, 0]),
#               np.array([0, 0, gamma])])

# M *= sigma/gamma

# J_e = np.array([sigma/gamma*beta_e*B_z**2*u_x, sigma/gamma*(1-beta_i*B_z**2)*B_z**2*u_x, 0])

# cond = np.linalg.cond(M)

#######################################################3
### Code to build LHS matrix for a given B, beta_e, beta_i 
    
# def cpb(x,y,z):
#     '''
#     This is a function to auto-generate the cross-product matrix
#     to test some invertibility stuff. 
    
#     Hopefully it will help provide insight
#     '''
    
#     return np.array([[0, z, -y],[-z, 0, x],[y, -x, 0]])

# def cpb2(x,y,z):
#     '''
#     This is a function to auto-generate the cross-product matrix
#     to test some invertibility stuff. 
    
#     Hopefully it will help provide insight
#     '''
    
#     return np.array([[-(y**2+z**2), x*y, x*z],
#                       [y*x, -(x**2+z**2), y*z],
#                       [z*x, z*y, -(x**2+y**2)]])
    
# def norm_B(x,y,z):
#     return (x**2+y**2+z**2)**(1/2)

# def LHS(x = 0,y = 0,z = 6,mu_e = 0,mu_i = 0):
#     b_e = mu_e*norm_B(x,y,z)
#     b_i = mu_e*mu_i*(norm_B(x,y,z)**2)
#     I = np.eye(3)
#     return I+b_e/norm_B(x,y,z)*cpb(x,y,z)+b_i/(norm_B(x,y,z)**2)*cpb2(x,y,z)
    
    
# det = np.linalg.det
# root = np.sqrt
#######################################################3
    
'''
This code will be used to analyze the csv's that comsol auto-generates, and
create necessary plots. 
'''   

def generate_plots(x_keys = [], y_keys = [], file_name = ''):
    '''
    Can generate plots for all pairs of keys given, based on filename
    ** Note: filename should not include .csv **
    '''
    try:
        df = pd.read_csv(file_name+'.csv')
    except FileNotFoundError:
        print('Error importing filename. Need other name')
        raise Exception
    except pd.errors.ParserError:
        print('CSV in possible format issue. Attempting to rectify')
        try:
            df = pd.read_csv(file_name+'.csv', skiprows=4)
        except pd.errors.ParserError:
            print('Failed to reformat. Abort')
            raise Exception
        
    def make_plot(x_key, y_key):
        '''
        Generates a plot from file_neame using x_key, y_key
        '''
        
        fig,ax = plt.subplots(1,1)
        ax.plot(df[x_key],df[y_key])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return fig
    
    for x in x_keys:
        fig,ax = plt.subplots(len(y_keys),1,sharex=True)
        ticker=0
        for y in y_keys:
            if len(y_keys) != 1:
                ax[ticker].plot(df[x],df[y])
                ax[ticker].set_ylabel(y)
            else:
                ax.plot(df[x],df[y])
                ax.set_ylabel(y)
            ticker += 1
        if len(y_keys) != 1:
            ax[ticker].set_xlabel(x)
        else:
            ax.set_xlabel(x)
            
    return fig,ax

    
def get_keys(file_name = ''):
    '''
    Parameters
    ----------
    file_name : STR 
       file_name for which you will use to generate plots or whatnot

    Returns
    -------
    List of keys for specific filename
    '''
    try:
        df = pd.read_csv(file_name+'.csv')
    except FileNotFoundError:
        print('Error importing filename. Need other name')
        raise Exception
    except pd.errors.ParserError:
        print('CSV in possible format issue. Attempting to rectify')
        try:
            df = pd.read_csv(file_name+'.csv', skiprows=4)
        except pd.errors.ParserError:
            print('Failed to reformat. Abort')
            raise Exception
    
    return df.keys()
    
    

# file_list = ['evan_B_con_far_run1',
#              'evan_u_con_far_run1',
#              'evan_sig_con_far_run1',
#              'evan_res_con_far_run1',
#              'evan_mob_con_far_v3_run2',
#              'evan_mob_seg_far_run1',
#              'evan_mobe_con_far_run1'] 

### As I forgot to add the probes for sigma, u, B, this is necessary for 
### comparing to Rosa's equations. Otherwise, this would be done in a for loop
    
######## Hall Runs ##########
# file_list_hall = ['evan_B_seg_hal_run1',
#                   'evan_u_seg_hal_run1',
#                   'evan_sig_seg_hal_run1',
#                   'evan_res_seg_hal_run1']

# file = file_list_hall[0] ## file currently being used

# fig_hal,ax_hal = plt.subplots(2,3,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys()
# ## COMSOL power
# ax_hal[0][0].plot(df_hal[key_hal[0]],-df_hal[key_hal[12]],label = 'COMSOL') ## Total
# ax_hal[0][1].plot(df_hal[key_hal[0]],-df_hal[key_hal[10]],label = 'COMSOL') ## Faraday
# ax_hal[0][2].plot(df_hal[key_hal[0]],-df_hal[key_hal[11]],label = 'COMSOL') ## Hall
# ## Ideal power
# K_hal = df_hal[key_hal[6]]
# sig_hal = df_hal[key_hal[9]]
# B_hal = df_hal[key_hal[8]]
# u_hal = df_hal[key_hal[7]]
# power_ideal = K_hal*(1-K_hal)*sig_hal*(u_hal**2)*(B_hal**2)*(.5*6)**2/((1+0)**2+(.5*6)**2) ## Needs to be corrected
# power_ideal_hal = -np.array(power_ideal)
# ax_hal[1][0].plot(df_hal[key_hal[0]],-power_ideal_hal,label = 'Ideal') ## Total
# ax_hal[1][1].plot(df_hal[key_hal[0]],-power_ideal_hal,label = 'Ideal') ## Faraday
# ax_hal[1][2].plot(df_hal[key_hal[0]],-power_ideal_hal,label = 'Ideal') ## Hall
# ax_hal[0][0].legend()
# ax_hal[0][1].legend()
# ax_hal[0][2].legend()
# ## Relative Difference
# dif_hal_tot = abs(abs(df_hal[key_hal[12]]) - abs(power_ideal_hal))/abs(power_ideal_hal) ## Total diff
# dif_hal_far = abs(abs(df_hal[key_hal[10]]) - abs(power_ideal_hal))/abs(power_ideal_hal) ## Fara diff
# dif_hal_hal = abs(abs(df_hal[key_hal[11]]) - abs(power_ideal_hal))/abs(power_ideal_hal) ## Hal diff
# ax_B[1].plot(df_B[key_B[0]],diff_B)
# ax_B[1].set_xlabel(r'$B_z$ [T]')
# ax_B[1].set_ylabel('Rel. difference (Measured-idea)/ideal')
# fig_B.set_figheight(10)
# ax_B[0].legend()












######### Some Faraday runs: needs to be altered for con vs. seg ####


# key_B = get_keys(file_list[0])
# fig_B,ax_B = plt.subplots(2,1,sharex = True)
# df_B = pd.read_csv(file_list[0]+'.csv', skiprows=4)
# ax_B[0].plot(df_B[key_B[0]],-df_B[key_B[1]],label = 'COMSOL')
# K_B = df_B[key_B[4]]
# power_ideal_B = K_B*(1-K_B)*60*(1800**2)*(df_B[key_B[0]]**2)
# power_ideal_B = -np.array(power_ideal_B)
# ax_B[0].plot(df_B[key_B[0]],-power_ideal_B,'--',label='Ideal Power output')
# ax_B[0].set_title(r'Con. Faraday; Comparing Power Ouput: $B_z$')
# ax_B[0].set_ylabel('Ideal Power Output [W]')
# diff_B = abs((np.array(df_B[key_B[1]]) - power_ideal_B)/power_ideal_B)
# ax_B[1].plot(df_B[key_B[0]],diff_B)
# ax_B[1].set_xlabel(r'$B_z$ [T]')
# ax_B[1].set_ylabel('Rel. difference (Measured-idea)/ideal')
# fig_B.set_figheight(10)
# ax_B[0].legend()
    
# key_sig = get_keys(file_list[2])
# fig_sig,ax_sig = plt.subplots(2,1,sharex = True)
# df_sig = pd.read_csv(file_list[2]+'.csv', skiprows=4)
# ax_sig[0].plot(df_sig[key_sig[0]],-df_sig[key_sig[1]],label = 'COMSOL')
# K_sig = df_sig[key_sig[4]]
# power_ideal_sig = K_sig*(1-K_sig)*(df_sig[key_sig[0]])*(1800**2)*6**2
# power_ideal_sig = -np.array(power_ideal_sig)
# ax_sig[0].set_ylabel('Ideal Power Output [W]')
# ax_sig[0].plot(df_sig[key_sig[0]],-power_ideal_sig,'--',label='Ideal Power output')
# ax_sig[0].set_title(r'Con. Faraday; Comparing Power Ouput: $\sigma$')
# diff_sig = abs((np.array(df_sig[key_sig[1]]) - power_ideal_sig)/power_ideal_sig)
# ax_sig[1].plot(df_sig[key_sig[0]],diff_sig)
# ax_sig[1].set_xlabel(r'$\sigma$ [S/m]')
# ax_sig[1].set_ylabel('Rel. difference (Measured-idea)/ideal')
# fig_sig.set_figheight(10)
# ax_sig[0].legend()

# key_u = get_keys(file_list[1])
# fig_u,ax_u = plt.subplots(2,1,sharex = True)
# df_u = pd.read_csv(file_list[1]+'.csv', skiprows=4)
# ax_u[0].plot(df_u[key_u[0]],-df_u[key_u[1]],label = 'COMSOL')
# K_u = df_u[key_u[4]]
# power_ideal_u = K_u*(1-K_u)*60*(df_u[key_u[0]])**2*6**2
# power_ideal_u = -np.array(power_ideal_u)
# ax_u[0].set_ylabel('Ideal Power Output [W]')
# ax_u[0].plot(df_u[key_u[0]],-power_ideal_u,'--',label='Ideal Power output')
# ax_u[0].set_title(r'Seg. Faraday; Comparing Power Ouput: $u_x$')
# diff_u = abs((np.array(df_u[key_u[1]]) - power_ideal_u)/power_ideal_u)
# ax_u[1].plot(df_u[key_u[0]],diff_u)
# ax_u[1].set_xlabel(r'$u_x$ [m/s]')
# ax_u[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_u.set_figheight(10)
# ax_u[0].legend()

# key_hal = get_keys('evan_res_con_far_run1')
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv('evan_res_con_far_run1.csv', skiprows=4)
# ax_hal[0].plot(df_hal[key_hal[4]],-df_hal[key_hal[1]],label = 'COMSOL')
# K_hal = df_hal[key_hal[4]]
# power_ideal_hal = K_hal*(1-K_hal)*60*1800**2*6**2
# power_ideal_hal = -np.array(power_ideal_hal)
# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].plot(df_hal[key_hal[4]],-power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: $K$')
# diff_hal = abs((np.array(df_hal[key_hal[1]]) - power_ideal_hal)/power_ideal_hal)
# ax_hal[1].plot(df_hal[key_hal[4]],diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)
# ax_hal[0].legend()    
    
###### Segmented Runs #######
# file_list = ['evan_B_seg_far_run1',
#              'evan_u_seg_far_run1',
#              'evan_sig_seg_far_run1'] 

# ### As I forgot to add the probes for sigma, u, B, this is necessary for 
# ### comparing to Rosa's equations. Otherwise, this would be done in a for loop
    
# key_B = get_keys(file_list[0])
# fig_B,ax_B = plt.subplots(2,1,sharex = True)
# df_B = pd.read_csv(file_list[0]+'.csv', skiprows=4)
# ax_B[0].plot(df_B[key_B[0]],-df_B[key_B[1]],label = 'COMSOL')
# K_B = df_B[key_B[4]]
# power_ideal_B = K_B*(1-K_B)*60*(1800**2)*(df_B[key_B[0]]**2)
# power_ideal_B = -np.array(power_ideal_B)
# ax_B[0].plot(df_B[key_B[0]],-power_ideal_B,'--',label='Ideal Power output')
# ax_B[0].set_title(r'Seg. Faraday; Comparing Power Ouput: $B_z$')
# ax_B[0].set_ylabel('Ideal Power Output [W]')
# diff_B = abs((np.array(df_B[key_B[1]]) - power_ideal_B)/power_ideal_B)
# ax_B[1].plot(df_B[key_B[0]],diff_B)
# ax_B[1].set_xlabel(r'$B_z$ [T]')
# ax_B[1].set_ylabel('Rel. difference (Measured-idea)/ideal')
# fig_B.set_figheight(10)
    
# key_sig = get_keys(file_list[2])
# fig_sig,ax_sig = plt.subplots(2,1,sharex = True)
# df_sig = pd.read_csv(file_list[2]+'.csv', skiprows=4)
# ax_sig[0].plot(df_sig[key_sig[0]],-df_sig[key_sig[1]],label = 'COMSOL')
# K_sig = df_sig[key_sig[4]]
# power_ideal_sig = K_sig*(1-K_sig)*(df_sig[key_sig[0]])*(1800**2)*6**2
# power_ideal_sig = -np.array(power_ideal_sig)
# ax_sig[0].set_ylabel('Ideal Power Output [W]')
# ax_sig[0].plot(df_sig[key_sig[0]],-power_ideal_sig,'--',label='Ideal Power output')
# ax_sig[0].set_title(r'Seg. Faraday; Comparing Power Ouput: $\sigma$')
# diff_sig = abs((np.array(df_sig[key_sig[1]]) - power_ideal_sig)/power_ideal_sig)
# ax_sig[1].plot(df_sig[key_sig[0]],diff_sig)
# ax_sig[1].set_xlabel(r'$\sigma$ [S/m]')
# ax_sig[1].set_ylabel('Rel. difference (Measured-idea)/ideal')
# fig_sig.set_figheight(10)

# key_u = get_keys(file_list[1])
# fig_u,ax_u = plt.subplots(2,1,sharex = True)
# df_u = pd.read_csv(file_list[1]+'.csv', skiprows=4)
# ax_u[0].plot(df_u[key_u[0]],-df_u[key_u[1]],label = 'COMSOL')
# K_u = df_u[key_u[4]]
# power_ideal_u = K_u*(1-K_u)*60*(df_u[key_u[0]])**2*6**2
# power_ideal_u = -np.array(power_ideal_u)
# ax_u[0].set_ylabel('Ideal Power Output [W]')
# ax_u[0].plot(df_u[key_u[0]],-power_ideal_u,'--',label='Ideal Power output')
# ax_u[0].set_title(r'Seg. Faraday; Comparing Power Ouput: $u_x$')
# diff_u = abs((np.array(df_u[key_u[1]]) - power_ideal_u)/power_ideal_u)
# ax_u[1].plot(df_u[key_u[0]],diff_u)
# ax_u[1].set_xlabel(r'$u_x$ [m/s]')
# ax_u[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_u.set_figheight(10)

# key_hal = get_keys('evan_res_seg_far_run1')
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv('evan_res_seg_far_run1.csv', skiprows=4)
# ax_hal[0].plot(df_hal[key_hal[4]],-df_hal[key_hal[1]],label = 'COMSOL')
# K_hal = df_hal[key_hal[4]]
# power_ideal_hal = K_hal*(1-K_hal)*60*1800**2*6**2
# power_ideal_hal = -np.array(power_ideal_hal)
# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].plot(df_hal[key_hal[4]],-power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[0].set_title(r'Seg. Faraday, Comparing Power Ouput: $K$')
# diff_hal = abs((np.array(df_hal[key_hal[1]]) - power_ideal_hal)/power_ideal_hal)
# ax_hal[1].plot(df_hal[key_hal[4]],diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)    

# df_mob = pd.read_csv(file_list[4]+'.csv', skiprows=4)
# key_mob = df_mob.keys()

# fig_mob = plt.figure()
# ax_mob = fig_mob.add_subplot(111,projection='3d')

# beta_i = np.array(df_mob[key_mob[1]])
# power = np.array(df_mob[key_mob[2]])
# K_mob = df_mob[key_mob[5]]
# sigma_mob = df_mob[key_mob[15]]
# vel_mob = df_mob[key_mob[13]]
# B_mob = df_mob[key_mob[14]]
# beta_e = np.array(df_mob[key_mob[0]])
# power_ideal_mob = (K_mob*(1-K_mob)*60*(1-beta_i)*1800**2*6**2)/ \
#     ((1-beta_i)**2 + beta_e**2)

# surf = ax_mob.plot_trisurf(beta_e,beta_i,-power,label='COMSOL Power-out')
# surf2 = ax_mob.plot_trisurf(beta_e,beta_i,power_ideal_mob,label='Ideal Power-out')
# ax_mob.set_xlabel(r'$\beta_e$')
# ax_mob.set_ylabel(r'$\beta_i$')
# ax_mob.set_zlabel(r'Measured Power-out')

# ## To account for a bug ##
# surf._edgecolors2d=surf._edgecolors3d
# surf._facecolors2d=surf._facecolors3d
# surf2._edgecolors2d=surf2._edgecolors3d
# surf2._facecolors2d=surf2._facecolors3d
# ## To account for a bug ##

# ax_mob.legend()

# fig_mob2 = plt.figure()
# ax_mob2 = fig_mob2.add_subplot(111,projection='3d')
# diff = abs(power-power_ideal_mob)/abs(power_ideal_mob)
# surf3 = ax_mob2.plot_trisurf(beta_e,beta_i,np.log((diff)))
# ax_mob2.set_xlabel(r'$\beta_e$')
# ax_mob2.set_ylabel(r'$\beta_i$')
# ax_mob2.set_zlabel(r'Log(Rel. difference)')

# key_u = get_keys(file_list[5])
# fig_u,ax_u = plt.subplots(2,1,sharex = True)
# df_u = pd.read_csv(file_list[5]+'.csv', skiprows=4)
# ax_u[0].plot(6*df_u[key_u[0]],df_u[key_u[2]],label = 'COMSOL')
# beta_e = np.array(df_u[key_u[0]])*6
# beta_i = np.array(df_u[key_u[1]])
# power = np.array(df_u[key_u[2]])
# K_mob = df_u[key_u[5]]
# sigma_mob = df_u[key_u[17]]
# vel_mob = df_u[key_u[13]]
# B_mob = df_u[key_u[14]]
# power_ideal_mob = (K_mob*(1-K_mob)*sigma_mob*(1-beta_i)*vel_mob**2*B_mob**2)/ \
#     ((1-beta_i)**2+beta_e**2)
# ax_u[0].set_ylabel('Ideal Power Output [W]')
# ax_u[0].plot(6*df_u[key_u[0]],power_ideal_mob,'--',label='Ideal Power output')
# ax_u[0].set_title(r'Seg. Faraday; Comparing Power Ouput: $\beta_e$')
# ax_u[0].legend()
# diff_u = abs((np.array(df_u[key_u[1]]) - power_ideal_u)/power_ideal_u)
# ax_u[1].plot(df_u[key_u[0]],diff_u)
# ax_u[1].set_xlabel(r'$u_x$ [m/s]')
# ax_u[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_u.set_figheight(10)
    
    
# df_mob = pd.read_csv('evan_mob_con_far_v3_run2.csv', skiprows=4)
# key_mob = df_mob.keys()

# fig_mob = plt.figure()
# ax_mob = fig_mob.add_subplot(111,projection='3d')

# beta_i = np.array(df_mob[key_mob[1]])
# power = np.array(df_mob[key_mob[2]])
# K_mob = df_mob[key_mob[5]]
# sigma_mob = df_mob[key_mob[15]]
# vel_mob = df_mob[key_mob[13]]
# B_mob = df_mob[key_mob[14]]
# beta_e = np.array(df_mob[key_mob[0]])
# power_ideal_mob = (K_mob*(1-K_mob)*60*(1-beta_i)*1800**2*6**2)/ \
#     ((1-beta_i)**2 + beta_e**2)

# surf = ax_mob.plot(beta_e,beta_i,-power,label='COMSOL Power-out')
# surf2 = ax_mob.plot(beta_e,beta_i,power_ideal_mob,label='Ideal Power-out')
# ax_mob.set_xlabel(r'$\beta_e$')
# ax_mob.set_ylabel(r'$\beta_i$')
# ax_mob.set_zlabel(r'Measured Power-out')

# ## To account for a bug ##
# surf._edgecolors2d=surf._edgecolors3d
# surf._facecolors2d=surf._facecolors3d
# surf2._edgecolors2d=surf2._edgecolors3d
# surf2._facecolors2d=surf2._facecolors3d
# ## To account for a bug ##

# ax_mob.legend()

# fig_mob2 = plt.figure()
# ax_mob2 = fig_mob2.add_subplot(111,projection='3d')
# diff = abs(power-power_ideal_mob)/abs(power_ideal_mob)
# surf3 = ax_mob2.plot_trisurf(beta_e,beta_i,np.log((diff)))
# ax_mob2.set_xlabel(r'$\beta_e$')
# ax_mob2.set_ylabel(r'$\beta_i$')
# ax_mob2.set_zlabel(r'Log(Rel. difference)')

    
    
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv('evan_K_con_far_run7.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# ax_hal[0].plot(df_hal[key_hal[5]],-df_hal[key_hal[3]],label = 'COMSOL')
# K_hal = df_hal[key_hal[5]]
# power_ideal_hal = K_hal*(1-K_hal)*60*1800**2*6**2
# power_ideal_hal = -np.array(power_ideal_hal)
# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].plot(K_hal,-power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: $K$')
# ax_hal[0].legend()
# diff_hal = abs((np.array(df_hal[key_hal[3]]) - power_ideal_hal)/power_ideal_hal)
# ax_hal[1].plot(df_hal[key_hal[5]],diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10) 
    
# df_hal_2 = pd.read_csv('evan_K_con_far_run4.csv', skiprows=4)
# df_hal_3 = pd.read_csv('evan_K_con_far_run3.csv', skiprows=4)
    
    
# fig_2, ax_2 = plt.subplots(1,1)
# # ax_2.plot(df_hal[key_hal[5]],-df_hal[key_hal[3]],label='Faraday Power')
# # ax_2.plot(df_hal[key_hal[5]],-df_hal[key_hal[8]],label='Total Power')
# ax_2.plot(df_hal[key_hal[5]],-df_hal[key_hal[7]],label='Hall Power')
# ax_2.legend()

# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv('evan_K_con_far_run9.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# ax_hal[0].plot(df_hal[key_hal[4]],-df_hal[key_hal[3]],label = 'COMSOL')
# K_hal = df_hal[key_hal[4]]
# power_ideal_hal = K_hal*(1-K_hal)*60*1800**2*6**2
# power_ideal_hal = -np.array(power_ideal_hal)

# K_hal_new = df_hal[key_hal[12]]/(1800*6)
# power_ideal_hal_new = K_hal_new*(1-K_hal_new)*60*1800**2*6**2
# power_ideal_hal_new = -np.array(power_ideal_hal_new)

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].plot(K_hal,-power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[0].plot(K_hal_new,-power_ideal_hal_new,'-+',label='COMSOL Power output v.2')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: $K$')
# ax_hal[0].legend()

# diff_hal = abs((np.array(df_hal[key_hal[3]]) - power_ideal_hal)/power_ideal_hal)
# diff_hal_new = abs((np.array(df_hal[key_hal[3]]) - power_ideal_hal_new)/power_ideal_hal)

# ax_hal[1].plot(K_hal,diff_hal, label='K comp. COMSOL')
# ax_hal[1].plot(K_hal_new,diff_hal_new,label='K comp. outside')
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# ax_hal[1].legend()
# fig_hal.set_figheight(10) 

fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
df_hal = pd.read_csv('evan_K_seg_far_run6.csv', skiprows=4)
key_hal = df_hal.keys().to_numpy()

K_hal = df_hal[key_hal[8]]
COMSOL_power = df_hal[key_hal[5]]

power_ideal_hal = K_hal*(1-K_hal)*60*1800**2*6**2
power_ideal_hal = -np.array(power_ideal_hal)

K_hal_new = -df_hal[key_hal[10]]/(1800*6)
power_ideal_hal_new = K_hal_new*(1-K_hal_new)*60*1800**2*6**2
power_ideal_hal_new = -np.array(power_ideal_hal_new)

ax_hal[0].set_ylabel('Ideal Power Output [W]')
ax_hal[0].plot(K_hal,-COMSOL_power,label = 'COMSOL')
ax_hal[0].plot(K_hal,-power_ideal_hal,'--',label='Ideal Power out')
# ax_hal[0].plot(K_hal_new,-power_ideal_hal_new,'-+',label='Ideal Power out v.2')
ax_hal[0].set_title(r'Seg. Faraday, Comparing Power Ouput: $K$')
ax_hal[0].legend()

# diff_hal = abs((np.array(df_hal[key_hal[3]]) - power_ideal_hal)/power_ideal_hal)
# diff_hal_new = abs((np.array(df_hal[key_hal[3]]) - power_ideal_hal_new)/power_ideal_hal)

# ax_hal[1].plot(K_hal,diff_hal, label='K comp. COMSOL')
# ax_hal[1].plot(K_hal_new,diff_hal_new,label='K comp. outside')
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# ax_hal[1].legend()
# fig_hal.set_figheight(10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    