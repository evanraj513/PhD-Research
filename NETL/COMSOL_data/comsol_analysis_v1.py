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
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('x-small')
import matplotlib.backends.backend_pdf ### Specifically for saving multiple graphs into one PDF

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
# ## Code to build LHS matrix for a given B, beta_e, beta_i 
    
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
    # b_e = mu_e*norm_B(x,y,z)
    # b_i = mu_e*mu_i*(norm_B(x,y,z)**2)
    # I = np.eye(3)
    # return I+b_e/norm_B(x,y,z)*cpb(x,y,z)+b_i/(norm_B(x,y,z)**2)*cpb2(x,y,z)


# def RHS(x = 0, y = 0, z = 6, mu_e=0, mu_i=0, sig = 60, u = 1800):
#     b_e = mu_e*norm_B(x,y,z)
#     b_i = mu_e*mu_i*(norm_B(x,y,z)**2)
#     gamma = (1-b_i)**2+b_e**2
#     val = np.array([[b_e*sig/gamma*u*norm_B(x,y,z)],
#                     [-(1-b_i)*sig/gamma*u*norm_B(x,y,z)],
#                     [0]])
#     return val
    
# det = np.linalg.det
# root = np.sqrt
# cond = np.linalg.cond
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
####### Seg. Hall ##########
# file_list = [['evan_B_seg_hal_run3','B'],
#               ['evan_u_seg_hal_run3','u'],
#               ['evan_sig_seg_hal_run2',r'$\sigma$']]

# file_list = [['evan_u_seg_hal_run5','u']]

# ticker = 1
# for fl in file_list:
#     print('\n','*'*40,'\n')
#     if ticker == 10:
#         pass
#     file=fl[0]
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     df_hal = pd.read_csv(file+'.csv', skiprows=4)
#     key_hal = df_hal.keys().to_numpy()
#     if ticker == 0 or ticker == 2:        
#         x_plot = df_hal[key_hal[0]] ## x-value for plottings
#         comsol_power = df_hal[key_hal[9]]
#         print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
        
        
#         K = df_hal[key_hal[4]]
#         u = df_hal[key_hal[5]]
#         B = df_hal[key_hal[6]]
#         sig = df_hal[key_hal[7]]
#         beta = df_hal[key_hal[1]] 
        
#         print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name+
#               '\n beta:'+beta.name)
        
#     elif ticker == 1:         
#         x_plot = df_hal[key_hal[1]] ## x-value for plottings
#         comsol_power = abs(df_hal[key_hal[9]])
#         print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
        
        
#         K = df_hal[key_hal[4]]
#         u = df_hal[key_hal[5]]
#         B = df_hal[key_hal[6]]
#         sig = df_hal[key_hal[7]]
#         beta = df_hal[key_hal[2]] 
        
#         print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#               '\n beta:'+beta.name)
        
#     power_ideal_hal = K*(1-K)*sig*u**2*B**2*beta**2/(1+beta**2)
    
#     ax_hal[0].plot(x_plot, comsol_power,label = 'COMSOL')
#     ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')
    
#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Seg. Hall, Comparing Power Ouput: '+fl[1])
#     ax_hal[0].legend()
    
#     diff_hal = abs((np.array(comsol_power) - np.array(power_ideal_hal))/power_ideal_hal)
    
#     ax_hal[1].plot(x_plot,diff_hal)
#     ax_hal[1].set_xlabel(fl[1])
#     ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
#     fig_hal.set_figheight(10)
    
#     ticker += 1
    
# file='evan_K_seg_hal_run2'
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[8]] ## x-value for plottings
# comsol_power = df_hal[key_hal[4]]
# print('Plotting: '+key_hal[8]+' vs. '+key_hal[4])

# K = df_hal[key_hal[8]]
# u = df_hal[key_hal[9]]
# B = df_hal[key_hal[10]]
# sig = df_hal[key_hal[11]]

# print('\n K: '+key_hal[8]+'\n u: '+key_hal[1]+'\n B: '+key_hal[2]+'\n sigma: '+key_hal[3])

# power_ideal_hal = K*(1-K)*sig*u**2*B**2

# ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
# ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+'K')
# ax_hal[0].legend()

# diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)

# ax_hal[1].plot(x_plot,diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)
    
    



####### Con. Hall ##########

## both beta ##
# file='evan_both_beta_con_hal_run1'
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# fig_hal,ax_hal = plt.subplots(3,1,sharex = True)

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = abs(df_hal[key_hal[7]])
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
# comsol_power = abs(comsol_power.to_numpy())

# K = df_hal[key_hal[3]]
# m = 4 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = x_plot
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[1]]
# beta_i = mob_e*mob_i*B**2

# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i : '+mob_i.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*beta_e**2/(((1-beta_i)**2+beta_e**2)*(beta_i-1))

# ax_hal[0].plot(x_plot,comsol_power,label = r'COMSOL')
# ax_hal[0].plot(x_plot, -power_ideal_hal,'--',label='Ideal Far. Power')

# ax_hal[0].set_ylabel('Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+r'$\beta_e$')
# ax_hal[0].legend()

# diff_hal = abs((comsol_power - power_ideal_hal)/power_ideal_hal)


# ax_hal[1].plot(x_plot, -df_hal[key_hal[5]],label='Hall power')
# ax_hal[1].set_ylabel('Hall Power [W]')

# # ax_hal[1].plot(x_plot, diff_hal)
# # ax_hal[1].set_ylabel('Relative Difference')
 
# ax_hal[2].plot(x_plot,K) 
# ax_hal[2].set_ylabel('K []')  
# ax_hal[2].set_xlabel(r'$\beta_e$ []')

# fig_hal.set_figheight(10)












######### Continuous Faraday ####
# file_list = [['evan_B_con_far_run2','B'],
#               ['evan_sig_con_far_run2',r'$\sigma$'],
#               ['evan_u_con_far_run3','u']]

# ticker=0
# for fl in file_list:
#     file=fl[0]
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     df_hal = pd.read_csv(file+'.csv', skiprows=4)
#     key_hal = df_hal.keys().to_numpy()
    
#     if ticker != 2:
#         x_plot = df_hal[key_hal[0]] ## x-value for plottings
#     else:
#         x_plot = df_hal[key_hal[1]] ## x-value for plottings
#     comsol_power = df_hal[key_hal[3]]
#     print('*'*40+'\n Plotting: '+x_plot.name+' vs. '+comsol_power.name)
    
    
#     K = df_hal[key_hal[4]]
#     u = df_hal[key_hal[8]]
#     B = df_hal[key_hal[9]]
#     sig = df_hal[key_hal[10]]
    
#     print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name)
    
#     power_ideal_hal = K*(1-K)*sig*u**2*B**2
    
#     ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
#     ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')
    
#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+fl[1])
#     ax_hal[0].legend()
    
#     diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)
    
#     ax_hal[1].plot(x_plot,diff_hal)
#     ax_hal[1].set_xlabel(fl[1])
#     ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
#     fig_hal.set_figheight(10)
    
#     ticker+=1
    
# file='evan_K_con_far_run13'
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[8]] ## x-value for plottings
# comsol_power = df_hal[key_hal[4]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)

# K = df_hal[key_hal[8]]
# u = df_hal[key_hal[9]]
# B = df_hal[key_hal[10]]
# sig = df_hal[key_hal[11]]

# print('\n K: '+key_hal[8]+'\n u: '+key_hal[1]+'\n B: '+key_hal[2]+'\n sigma: '+key_hal[3])

# power_ideal_hal = K*(1-K)*sig*u**2*B**2

# ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
# ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+'K')
# ax_hal[0].legend()

# diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)

# ax_hal[1].plot(x_plot,diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)



## beta_e run ##
# file='evan_beta_e_con_far_run5'
# fig_hal,ax_hal = plt.subplots(3,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = abs(df_hal[key_hal[2]])
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)

# K = df_hal[key_hal[6]]
# m = 7 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[0]]
# beta_e = mob_e*B

# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'Comsol Power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2/(1+beta_e**2)

# ax_hal[0].plot(x_plot, comsol_power,label = r'COMSOL: resistance $= 7E-4$')
# ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Far. Power')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+r'$\beta_e$')
# ax_hal[0].legend()

# diff_hal = abs((comsol_power - power_ideal_hal)/power_ideal_hal)


# # ax_hal[1].plot(x_plot, -df_hal[key_hal[5]],label='Hall power')
# # ax_hal[1].set_ylabel('Hall Power [W]')

# ax_hal[1].plot(x_plot, diff_hal)
# ax_hal[1].set_ylabel('Relative Difference')
 
# ax_hal[2].plot(x_plot,K) 
# ax_hal[2].set_ylabel('K []')  
# ax_hal[2].set_xlabel(r'$\beta_e$ []')

# fig_hal.set_figheight(10)



# ### both beta_e and beta_i
# file='evan_both_beta_con_far_run12'

# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# comsol_power = abs(df_hal[key_hal[2]])

# K = df_hal[key_hal[3]]
# m = 7 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[0]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[1]]
# beta_i = mob_i*mob_e*B*B


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))
# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal

# # # ##############
# # # ## 2D plots ##
# # # ##############

# x_plot = beta_i

# rs1 = int(beta_e.unique().size) ## Number of beta_e tested
# rs2 = int(beta_e.shape[0]/rs1) ## Number of beta_i for specific beta_e

# beta_e = beta_e.to_numpy().reshape(rs1,rs2)
# x_plot = x_plot.to_numpy().reshape(rs1,rs2)
# comsol_power = comsol_power.to_numpy().reshape(rs1,rs2)
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2)

# diff_hal = diff_hal.to_numpy().reshape(rs1,rs2)

# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)

# for a in np.arange(0,rs1):
#     ax_hal[0].plot(x_plot[a], comsol_power[a],'-o',label = r'COMSOL: $\beta_e = $'+str(beta_e[a][0]))
#     ax_hal[0].plot(x_plot[a], power_ideal_hal[a],'-x',label=r'Ideal Far. Power, $\beta_e = $'+str(beta_e[a][0]))
    
#     ax_hal[1].plot(x_plot[a], diff_hal[a], label=r'Diff. for $\beta_e = $'+str(beta_e[a][0]))
    

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+r'$\beta_e$ and $\beta_i$')
# ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
# ax_hal[1].set_ylabel('Rel. Diff')
# ax_hal[1].set_xlabel(r'$\beta_i$')
# ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
# fig_hal.set_size_inches(10,10)

##############
## 3D plots ##
##############

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# fig2 = plt.figure()
# ax2 = fig2.gca(projection='3d')

# ax.plot_trisurf(beta_e,beta_i,comsol_power)
# ax.plot_trisurf(beta_e,beta_i,power_ideal_hal)
# ax.set_xlabel(r'$\beta_e$ []')
# ax.set_ylabel(r'$\beta_i$ []')
# ax.set_zlabel(r'Power [MW]')

# # ax2.plot_trisurf(beta_e,beta_i,diff_hal)
# ax2.set_xlabel(r'$\beta_e$')
# ax2.set_ylabel(r'$\beta_i$')
# ax2.set_zlabel(r'Rel. Difference')
# ax2.set_title('Relative Difference between measured and ideal Power out')

# ax2.plot_trisurf(beta_e,beta_i,diff_hal)




















    
###### Segmented Faraday Runs #######
# file_list = [['evan_B_seg_far_run3','B'],
#                ['evan_sig_seg_far_run2',r'$\sigma$'],
#                ['evan_u_seg_far_run2','u']]

# file_list = [['evan_res_height_seg_far_run2','wire_rad [mm]']]

# for fl in file_list:
#     file=fl[0]
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     df_hal = pd.read_csv(file+'.csv', skiprows=4)
#     key_hal = df_hal.keys().to_numpy()
    
#     x_plot = df_hal[key_hal[0]] ## x-value for plottings
#     comsol_power = df_hal[key_hal[4]]
#     print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
    
#     K = df_hal[key_hal[8]]
#     u = df_hal[key_hal[9]]
#     B = df_hal[key_hal[10]]
#     sig = df_hal[key_hal[11]]
#     beta = df_hal[key_hal[1]]
    
#     print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#           '\n beta: '+beta.name)
    
#     power_ideal_hal = K*(1-K)*sig*u**2*B**2/(1+beta**2)
    
#     ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
#     ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')
    
#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Seg. Faraday, Comparing Power Ouput: '+fl[1])
#     ax_hal[0].legend()
    
#     diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)
    
#     ax_hal[1].plot(x_plot,diff_hal)
#     ax_hal[1].set_xlabel(fl[1])
#     ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
#     fig_hal.set_figheight(10)
    
# file='evan_K_seg_far_run8'
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[8]] ## x-value for plottings
# comsol_power = df_hal[key_hal[4]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
# K = df_hal[key_hal[8]]
# u = df_hal[key_hal[9]]
# B = df_hal[key_hal[10]]
# sig = df_hal[key_hal[11]]

# print('\n K: '+key_hal[8]+'\n u: '+key_hal[1]+'\n B: '+key_hal[2]+'\n sigma: '+key_hal[3])

# power_ideal_hal = K*(1-K)*sig*u**2*B**2

# ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
# ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Seg. Faraday, Comparing Power Ouput: '+'K')
# ax_hal[0].legend()

# diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)

# ax_hal[1].plot(x_plot,diff_hal)
# ax_hal[1].set_xlabel(r'$K$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)

#### beta_e run
# file='evan_beta_e_seg_far_run1'
# fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = df_hal[key_hal[3]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
# K = df_hal[key_hal[7]]
# m = 8 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# beta= df_hal[key_hal[0]]

# print('\n K: '+key_hal[8]+'\n u: '+key_hal[1]+'\n B: '+key_hal[2]+'\n sigma: '+key_hal[3])

# power_ideal_hal = K*(1-K)*sig*u**2*B**2/(1+beta**2)

# ax_hal[0].plot(x_plot,-comsol_power,label = 'COMSOL')
# ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Seg. Faraday, Comparing Power Ouput: '+r'$\beta$')
# ax_hal[0].legend()

# diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)

# ax_hal[1].plot(x_plot,diff_hal)
# ax_hal[1].set_xlabel(r'$\beta$ []')
# ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)

    
#### Varying CW:EW and IW:EW

# file='evan_chan_len_seg_far_run1'
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# fig_hal,ax_hal = plt.subplots(2,2)

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = df_hal[key_hal[4]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)

# K = df_hal[key_hal[8]]
# u = df_hal[key_hal[9]]
# B = df_hal[key_hal[10]]
# sig = df_hal[key_hal[11]]

# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2

# ax_hal[0][0].plot(x_plot,-comsol_power,label = 'COMSOL')
# ax_hal[0][0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')

# ax_hal[0][1].plot(df_hal[key_hal[15]],-comsol_power,label = 'COMSOL')
# ax_hal[0][1].plot(df_hal[key_hal[15]], power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[0][1].set_xlabel('CW:EW []')

# ax_hal[1][1].plot(df_hal[key_hal[16]],-comsol_power,label = 'COMSOL')
# ax_hal[1][1].plot(df_hal[key_hal[16]], power_ideal_hal,'--',label='Ideal Power output')
# ax_hal[1][1].set_xlabel('IW:EW []')

# ax_hal[0][0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0][0].set_title(r'Seg. Faraday, Comparing Power Ouput: '+'CW')
# ax_hal[0][0].legend()

# diff_hal = abs((np.array(comsol_power) + np.array(power_ideal_hal))/power_ideal_hal)

# ax_hal[1][0].plot(x_plot,diff_hal)
# ax_hal[1][0].set_xlabel(r'Channel Width (x) [mm]')
# ax_hal[1][0].set_ylabel('Rel. difference (Measured-ideal)/ideal')
# fig_hal.set_figheight(10)

# file='evan_both_beta_seg_far_run2'
# # fig_hal,ax_hal = plt.subplots(3,1,sharex = True)

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# fig2 = plt.figure()
# ax2 = fig2.gca(projection='3d')

# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = df_hal[key_hal[3]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
# comsol_power = abs(comsol_power.to_numpy())

# K = df_hal[key_hal[7]]
# m = 8 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = x_plot
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[1]]
# beta_i = mob_i*mob_e*B*B
# # beta_i = df_hal[key_hal[1]]


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# ax.plot_trisurf(beta_e,beta_i,comsol_power)
# ax.plot_trisurf(beta_e,beta_i,power_ideal_hal)
# ax.set_xlabel(r'$\beta_e$ []')
# ax.set_ylabel(r'$\beta_i$ []')
# ax.set_zlabel(r'Power [MW]')
# ax.set_title(r'Measured and Ideal power: Seg. Far.')

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal

# ax2.plot_trisurf(beta_e,beta_i,diff_hal)
# ax2.set_xlabel(r'$\beta_e$')
# ax2.set_ylabel(r'$\beta_i$')
# ax2.set_zlabel(r'Rel. Difference')
# ax2.set_title('Relative Difference between measured and ideal Power: Seg. Far.')

# ax2.plot_trisurf(beta_e,beta_i,diff_hal)

# x_plot = x_plot.to_numpy().reshape(5,5).T
# comsol_power = comsol_power.reshape(5,5).T
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(5,5).T

# a = 4
# ax_hal[0].plot(x_plot[a],comsol_power[a],'+',label = r'COMSOL: resistance $= 7E-4$')
# ax_hal[0].plot(x_plot[a], power_ideal_hal[a],'x',label='Far. Power')

# ax_hal[0].set_ylabel('Ideal Power Output [W]')
# ax_hal[0].set_title(r'Con. Faraday, Comparing Power Ouput: '+r'$\beta_e$ and $\beta_i$')
# ax_hal[0].legend()

# K = K.to_numpy().reshape(5,5).T

# a = 1

# fig3,ax3 = plt.subplots(1,1)
# ax3.plot(K[a],power_ideal_hal[a],'+')
# ax3.plot(K[a],comsol_power[a],'x')


###############################################################################################
########################### Sensitivity Analyses ##############################################
###############################################################################################

######### Continuous Faraday ############

#########
### u ###
#########

# file='sens_con_far_u_run7'

# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# comsol_power = abs(df_hal[key_hal[3]])

# K = df_hal[key_hal[4]]
# m = 8 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[1]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[2]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[0]]
# beta_i = mob_i*mob_e*B*B


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = beta_i

# rs1 = int(mob_e.unique().size) ## Number of beta_e tested
# rs2 = int(u.unique().size) ## Number of u's tested
# rs3 = int(beta_e.shape[0]/rs1/rs2) ## Number of beta_i for each unique pair (beta_e, u)

# beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# u = u.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# for a in np.arange(0,rs1):
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     for b in np.arange(0,rs2):
#         ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
#                         label=r'COMSOL Power'+
#                         r' $u = $'+str(u[0][b][a]))
#         ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x',
#                         label=r'Ideal Power' +
#                         r' $u = $'+str(u[0][b][a]))
    
#         ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
#                         label=r'Diff. for'+
#                         r' $u = $'+str(u[0][b][a]))
    

#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, u, $\beta_E = $'+str(beta_e[a][0][0]))
#     ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     ax_hal[1].set_ylabel('Rel. Diff')
#     ax_hal[1].set_xlabel(r'$\beta_i$')
#     ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     fig_hal.set_size_inches(10,10)

#########
### B ###
#########

# file='sens_con_far_B_run1'

# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# comsol_power = abs(df_hal[key_hal[4]])

# K = df_hal[key_hal[7]]
# m = 8 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[1]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[2]]
# beta_i = mob_i*mob_e*B*B


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = beta_i

# rs1 = int(mob_e.unique().size) ## Number of beta_e tested
# rs2 = int(B.unique().size) ## Number of u's tested
# rs3 = int(beta_e.shape[0]/rs1/rs2) ## Number of beta_i for each unique pair (beta_e, u)

# mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
# beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# B = B.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# for a in np.arange(0,rs1):
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     for b in np.arange(0,rs2):
#         ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
#                         label = r'COMSOL:'+
#                         r' $B = $'+str(B[0][b][a]))
#         ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x',
#                         label=r'Ideal Far. Power,'+
#                         r' $B = $'+str(B[0][b][a]))
    
#         ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
#                         label=r'Diff. for'+
#                         r' $B = $'+str(B[0][b][a]))
    

#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, B, $\mu_e = $'+str(mob_e[a][0][0]))
#     ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     ax_hal[1].set_ylabel('Rel. Diff')
#     ax_hal[1].set_xlabel(r'$\beta_i$')
#     ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     fig_hal.set_size_inches(10,10)

#########
## sig ##
#########

# file='sens_con_far_sig_run4'

# df_hal = pd.read_csv(file+'.csv',skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# comsol_power = abs(df_hal[key_hal[3]])

# K = df_hal[key_hal[4]]
# m = 8 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[1]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[2]]
# beta_i = mob_i*mob_e*B*B


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = beta_i

# rs1 = int(mob_e.unique().size) ## Number of beta_e tested
# rs2 = int(sig.unique().size) ## Number of u's tested
# rs3 = int(beta_e.shape[0]/rs1/rs2) ## Number of beta_i for each unique pair (beta_e, u)

# mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
# beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# sig = sig.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# for a in np.arange(0,rs1):
#     fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     for b in np.arange(0,rs2):
#         ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
#                         label = r'COMSOL: '+
#                         r'$\sigma = $'+str(sig[0][b][a]))
#         ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x',
#                         label=r'Ideal Far. Power, '+
#                         r'$\sigma = $'+str(sig[0][b][a]))
    
#         ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
#                         label=r'Diff. for '+
#                         r'$\sigma = $'+str(sig[0][b][a]))
    

#     ax_hal[0].set_ylabel('Ideal Power Output [W]')
#     ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, $\sigma$, $\beta_e = $'+str(beta_e[a][0][0]))
#     ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     ax_hal[1].set_ylabel('Rel. Diff')
#     ax_hal[1].set_xlabel(r'$\beta_i$')
#     ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     fig_hal.set_size_inches(10,10)

#############
## L_inlet ##
#############

##### Cont. Faraday #####

file_name='con_far_L_sens_run1_2'

file='../comsol_data_v2/'+file_name
# file='sens_seg_far_L_run2'

# plot = 'Joule Heating'
# plot = 'Resistor'
# plot = 'Resistor_Joule Heating'

# plot_all = ['Joule Heating','Resistor','Resistor_Joule Heating']

# saving = 1 ## saves pdf in output file, named by file_name. 0 = off, 1 = on. 
    
# if saving == 1:
    # pdf = matplotlib.backends.backend_pdf.PdfPages('../saved_plots_2/'+file_name+'_'+ plot+'plot.pdf')
    # pdf = matplotlib.backends.backend_pdf.PdfPages(file_name+'_'+ plot+'plot.pdf')

def plot_data(plot):
    
    plot_type = 0
    '''
    Sets the way plots are viewed: 0 = all separate, 1 = separated by diff. beta_e, 2 = all together
    '''
    
    saving = 1 ## saves pdf in output file, named by file_name. 0 = off, 1 = on. 
    
    if saving == 1:
        pdf = matplotlib.backends.backend_pdf.PdfPages('../saved_plots_2/'+file_name+'_'+ plot+'_plot.pdf')
        # pdf = matplotlib.backends.backend_pdf.PdfPages(file_name+'_'+ plot+'plot.pdf')
    
    df_hal = pd.read_csv(file+'.csv',skiprows=4)
    key_hal = df_hal.keys().to_numpy()
    df_hal = df_hal.sort_values(by=[key_hal[1],key_hal[2]])
    
    if plot == 'Resistor':
        comsol_power = abs(df_hal[key_hal[7]])
        y_plot_name = comsol_power.name
    elif plot == 'Joule Heating':
        comsol_power = abs(df_hal[key_hal[12]])
        y_plot_name = comsol_power.name
    elif plot == 'Resistor_Joule Heating':
        comsol_power = abs(df_hal[key_hal[7]]) - abs(df_hal[key_hal[12]])
        y_plot_name = 'Resistor Power - Joule Heating [W]'
    
    K = df_hal[key_hal[3]]
    m = 4 ## to make redefining u,B,sig easier. 
    u = df_hal[key_hal[m]]
    B = df_hal[key_hal[m+1]]
    sig = df_hal[key_hal[m+2]]
    mob_e = df_hal[key_hal[1]]
    beta_e = mob_e*B
    mob_i = df_hal[key_hal[2]]
    beta_i = mob_e*mob_i*B**2
    
    
    print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
          '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
          '\n'+'Plotting: '+y_plot_name)
    
    power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))
    
    x_plot = df_hal[key_hal[0]] ## L_input
    x_plot_name = x_plot.name
    
    rs1 = int(beta_e.unique().size) ## Number of beta_e tested
    rs3 = int(x_plot.unique().size) ## Number of L's tested
    rs2 = int(beta_e.shape[0]/rs1/rs3) ## Number of beta_i for each unique pair (beta_e, u)
    
    mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
    beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
    beta_i = beta_i.to_numpy().reshape(rs1,rs2,rs3)
    x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
    comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
    power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
    sig = sig.to_numpy().reshape(rs1,rs2,rs3)
    
    diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
    diff_hal = diff_hal.reshape(rs1,rs2,rs3)
    
    if plot_type == 2:
        print('Plotting all plots inlaid')
        fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
    
    for a in np.arange(0,rs1):
        # fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
        if plot_type == 1:
            fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
        for b in np.arange(0,rs2):
            
            if plot_type == 0:
                fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
                ax_hal.plot(x_plot[a][b],comsol_power[a][b])
                ax_hal.set_ylabel(y_plot_name)
                ax_hal.set_xlabel(x_plot_name)
                ax_hal.set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3))
                                 +r' $\beta_i = $'+str(round(beta_i[a][b][0],3)))
                # ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
                fig_hal.set_size_inches(10,10)
                
                if saving == 1:
                    pdf.savefig(fig_hal)
                    
        
            # fig_hal.set_size_inches(10,10)
            # ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
            #                 label = r'COMSOL: '+
            #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
            # ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x'),
            #                 # label=r'Ideal Far. Power, '+
            #                 # r'$\beta_i = $'+str(beta_i[0][b][a]))
        
            # ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
            #                 label=r'Diff. for '+
            #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
            
            elif plot_type == 1:
                ax_hal.plot(x_plot[a][b], comsol_power[a][b],'-o',
                                label = r'COMSOL: '+
                                r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
                                # r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))
            elif plot_type == 2:
                ax_hal.plot(x_plot[a][b], comsol_power[a][b],'-o',
                            label = r'COMSOL: '+
                            r'$\beta_i = $'+str(round(beta_i[a][b][0],2))+
                            r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        
    
        # ax_hal[0].set_ylabel('Power Output [W]')
        # ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        # ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        # ax_hal[1].set_ylabel('Rel. Diff')
        # ax_hal[1].set_xlabel(r'L [m]')
        # ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        
        if plot_type == 1:
            ax_hal.set_ylabel(y_plot_name)
            ax_hal.set_xlabel(x_plot_name)
            ax_hal.set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
            ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
            
            fig_hal.set_size_inches(10,10)
            
            if saving == 1:
                    pdf.savefig(fig_hal) 
                    
                    
    if plot_type == 2:
        ax_hal.set_ylabel(y_plot_name)
        ax_hal.set_xlabel(x_plot_name)
        ax_hal.set_title(r'Con. Faraday, Sensitivity Analysis, L, all plotted')
        ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        fig_hal.set_size_inches(10,10)
        
        if saving == 1:
            pdf.savefig(fig_hal)
            
    if saving == 1:
        pdf.close()
            
            
plot_all = ['Joule Heating','Resistor','Resistor_Joule Heating']
for x in plot_all:
    plot_data(x)

##### Con. Far. beta_e insensitivity ####
# file='sens_con_far_beta_e_insensitivity_run1'

# df_hal = pd.read_csv(file+'.csv',skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# df_hal = df_hal.sort_values(by=[key_hal[0],key_hal[1]])

# comsol_power = abs(df_hal[key_hal[11]])

# K = df_hal[key_hal[2]]
# m = 5 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[0]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[1]]
# beta_i = mob_e*mob_i*B**2


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = beta_e ## beta_e

# rs1 = int(beta_e.unique().size) ## Number of beta_e tested
# rs3 = int(x_plot.unique().size) ## Number of L's tested
# rs2 = int(beta_e.shape[0]/rs1/rs3) ## Number of beta_i for each unique pair (beta_e, u)

# # mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
# # beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# # beta_i = beta_i.to_numpy().reshape(rs1,rs2,rs3)
# # x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# # comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# # power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# # sig = sig.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# # diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# # fig_hal,ax_hal = plt.subplots(1,1,sharex = True)

# fig_hal,ax_hal = plt.subplots(1,1,sharex = True)

# # ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
# #                 label = r'COMSOL: '+
# #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
# # ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x'),
# #                 # label=r'Ideal Far. Power, '+
# #                 # r'$\beta_i = $'+str(beta_i[0][b][a]))

# # ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
# #                 label=r'Diff. for '+
# #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))

# ax_hal.plot(x_plot, comsol_power,'-o',
#                 label = r'COMSOL: '+
#                 r'$\beta_i = $'+str(round(beta_i[0],2)))
#                 # r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))


# # ax_hal[0].set_ylabel('Power Output [W]')
# # ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
# # ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
# # ax_hal[1].set_ylabel('Rel. Diff')
# # ax_hal[1].set_xlabel(r'L [m]')
# # ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))

# ax_hal.set_ylabel(r'Faraday Power [W]')
# ax_hal.set_xlabel(r'$\beta_e$ []')
# ax_hal.set_title(r'Seg. Faraday, Sensitivity Analysis, L,')
# ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))

# fig_hal.set_size_inches(10,10)

    
    
####### Segmented Faraday #######
# file='../comsol_data_v2/seg_far_L_sens_run1_4'
# # file='sens_seg_far_L_run2'

# # plot = 'Joule Heating'
# # plot = 'Resistor'
# plot = 'Resistor - Joule Heating'


# plot_type = 1
# '''
# Sets the way plots are viewed: 0 = all separate, 1 = separated by diff. beta_e, 2 = all together
# '''

# df_hal = pd.read_csv(file+'.csv',skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# df_hal = df_hal.sort_values(by=[key_hal[1],key_hal[2]])

# if plot == 'Resistor':
#     comsol_power = abs(df_hal[key_hal[7]])
#     y_plot_name = comsol_power.name
# elif plot == 'Joule Heating':
#     comsol_power = abs(df_hal[key_hal[12]])
#     y_plot_name = comsol_power.name
# elif plot == 'Resistor - Joule Heating':
#     comsol_power = abs(df_hal[key_hal[7]]) - abs(df_hal[key_hal[12]])
#     y_plot_name = 'Resistor Power - Joule Heating [W]'

# K = df_hal[key_hal[3]]
# m = 4 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[1]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[2]]
# beta_i = mob_e*mob_i*B**2


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'Plotting: '+y_plot_name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = df_hal[key_hal[0]] ## L_input
# x_plot_name = x_plot.name

# rs1 = int(beta_e.unique().size) ## Number of beta_e tested
# rs3 = int(x_plot.unique().size) ## Number of L's tested
# rs2 = int(beta_e.shape[0]/rs1/rs3) ## Number of beta_i for each unique pair (beta_e, u)

# mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
# beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# beta_i = beta_i.to_numpy().reshape(rs1,rs2,rs3)
# x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# sig = sig.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# if plot_type == 2:
#     print('Plotting all plots inlaid')
#     fig_hal,ax_hal = plt.subplots(1,1,sharex = True)

# for a in np.arange(0,rs1):
#     # fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
#     if plot_type == 1:
#         fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
#     for b in np.arange(0,rs2):
        
#         if plot_type == 0:
#             fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
#             ax_hal.set_ylabel(y_plot_name)
#             ax_hal.set_xlabel(x_plot_name)
#             ax_hal.set_title(r'Seg. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3))
#                              +r' $\beta_i = $'+str(round(beta_i[a][b][0],3)))
#             ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#             fig_hal.set_size_inches(10,10)
    
#         # fig_hal.set_size_inches(10,10)
#         # ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
#         #                 label = r'COMSOL: '+
#         #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
#         # ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x'),
#         #                 # label=r'Ideal Far. Power, '+
#         #                 # r'$\beta_i = $'+str(beta_i[0][b][a]))
    
#         # ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
#         #                 label=r'Diff. for '+
#         #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
        
#         ax_hal.plot(x_plot[a][b], comsol_power[a][b],'-o',
#                         label = r'COMSOL: '+
#                         r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
#                         # r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))
    

#     # ax_hal[0].set_ylabel('Power Output [W]')
#     # ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
#     # ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
#     # ax_hal[1].set_ylabel('Rel. Diff')
#     # ax_hal[1].set_xlabel(r'L [m]')
#     # ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
    
#     if plot_type == 1:
#         ax_hal.set_ylabel(y_plot_name)
#         ax_hal.set_xlabel(x_plot_name)
#         ax_hal.set_title(r'Seg. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
#         ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        
#         fig_hal.set_size_inches(10,10)

#### Seg. Far. Insensitivty #####
# file='sens_seg_far_beta_e_insensitivity_run2'

# df_hal = pd.read_csv(file+'.csv',skiprows=4)
# key_hal = df_hal.keys().to_numpy()
# df_hal = df_hal.sort_values(by=[key_hal[0],key_hal[1]])

# comsol_power = abs(df_hal[key_hal[6]])

# K = df_hal[key_hal[2]]
# m = 3 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# mob_e = df_hal[key_hal[0]]
# beta_e = mob_e*B
# mob_i = df_hal[key_hal[1]]
# beta_i = mob_e*mob_i*B**2


# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
#       '\n'+'comsol_power: '+comsol_power.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))

# x_plot = beta_e ## beta_e

# rs1 = int(beta_e.unique().size) ## Number of beta_e tested
# rs3 = int(x_plot.unique().size) ## Number of L's tested
# rs2 = int(beta_e.shape[0]/rs1/rs3) ## Number of beta_i for each unique pair (beta_e, u)

# # mob_e = mob_e.to_numpy().reshape(rs1,rs2,rs3)
# # beta_e = beta_e.to_numpy().reshape(rs1,rs2,rs3)
# # beta_i = beta_i.to_numpy().reshape(rs1,rs2,rs3)
# # x_plot = x_plot.to_numpy().reshape(rs1,rs2,rs3)
# # comsol_power = comsol_power.to_numpy().reshape(rs1,rs2,rs3)
# # power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs1,rs2,rs3)
# # sig = sig.to_numpy().reshape(rs1,rs2,rs3)

# diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
# # diff_hal = diff_hal.reshape(rs1,rs2,rs3)

# # fig_hal,ax_hal = plt.subplots(1,1,sharex = True)

# fig_hal,ax_hal = plt.subplots(1,1,sharex = True)

# # ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-o',
# #                 label = r'COMSOL: '+
# #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
# # ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b],'--x'),
# #                 # label=r'Ideal Far. Power, '+
# #                 # r'$\beta_i = $'+str(beta_i[0][b][a]))

# # ax_hal[1].plot(x_plot[a][b], diff_hal[a][b], 
# #                 label=r'Diff. for '+
# #                 r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))

# ax_hal.plot(x_plot, comsol_power,'-o',
#                 label = r'COMSOL: '+
#                 r'$\beta_i = $'+str(round(beta_i[0],2)))
#                 # r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))


# # ax_hal[0].set_ylabel('Power Output [W]')
# # ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
# # ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
# # ax_hal[1].set_ylabel('Rel. Diff')
# # ax_hal[1].set_xlabel(r'L [m]')
# # ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))

# ax_hal.set_ylabel(r'Joule Heating [W]')
# ax_hal.set_xlabel(r'$\beta_e$ []')
# ax_hal.set_title(r'Seg. Faraday, Sensitivity Analysis, L,')
# ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))

# fig_hal.set_size_inches(10,10)



