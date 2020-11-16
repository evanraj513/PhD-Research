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
# file_list = [['evan_B_seg_hal_run3','B'],
#               ['evan_u_seg_hal_run3','u'],
#               ['evan_sig_seg_hal_run2',r'$\sigma$']]

file_list = [['evan_u_seg_hal_run5','u']]

ticker = 1
for fl in file_list:
    print('\n','*'*40,'\n')
    if ticker == 10:
        pass
    file=fl[0]
    fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
    df_hal = pd.read_csv(file+'.csv', skiprows=4)
    key_hal = df_hal.keys().to_numpy()
    if ticker == 0 or ticker == 2:        
        x_plot = df_hal[key_hal[0]] ## x-value for plottings
        comsol_power = df_hal[key_hal[9]]
        print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
        
        
        K = df_hal[key_hal[4]]
        u = df_hal[key_hal[5]]
        B = df_hal[key_hal[6]]
        sig = df_hal[key_hal[7]]
        beta = df_hal[key_hal[1]] 
        
        print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name+
              '\n beta:'+beta.name)
        
    elif ticker == 1:         
        x_plot = df_hal[key_hal[1]] ## x-value for plottings
        comsol_power = abs(df_hal[key_hal[9]])
        print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
        
        
        K = df_hal[key_hal[4]]
        u = df_hal[key_hal[5]]
        B = df_hal[key_hal[6]]
        sig = df_hal[key_hal[7]]
        beta = df_hal[key_hal[2]] 
        
        print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
              '\n beta:'+beta.name)
        
    power_ideal_hal = K*(1-K)*sig*u**2*B**2*beta**2/(1+beta**2)
    
    ax_hal[0].plot(x_plot, comsol_power,label = 'COMSOL')
    ax_hal[0].plot(x_plot, power_ideal_hal,'--',label='Ideal Power output')
    
    ax_hal[0].set_ylabel('Ideal Power Output [W]')
    ax_hal[0].set_title(r'Seg. Hall, Comparing Power Ouput: '+fl[1])
    ax_hal[0].legend()
    
    diff_hal = abs((np.array(comsol_power) - np.array(power_ideal_hal))/power_ideal_hal)
    
    ax_hal[1].plot(x_plot,diff_hal)
    ax_hal[1].set_xlabel(fl[1])
    ax_hal[1].set_ylabel('Rel. difference (Measured-ideal)/ideal')
    fig_hal.set_figheight(10)
    
    ticker += 1
    
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
# file='evan_beta_e_con_far_run4'
# fig_hal,ax_hal = plt.subplots(3,1,sharex = True)
# df_hal = pd.read_csv(file+'.csv', skiprows=4)
# key_hal = df_hal.keys().to_numpy()

# x_plot = df_hal[key_hal[0]] ## x-value for plottings
# comsol_power = df_hal[key_hal[2]]
# print('Plotting: '+x_plot.name+' vs. '+comsol_power.name)
# comsol_power = abs(comsol_power.to_numpy())

# K = df_hal[key_hal[3]]
# m = 7 ## to make redefining u,B,sig easier. 
# u = df_hal[key_hal[m]]
# B = df_hal[key_hal[m+1]]
# sig = df_hal[key_hal[m+2]]
# beta_e = x_plot

# print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
#       '\n'+'beta_e : '+beta_e.name)

# power_ideal_hal = K*(1-K)*sig*u**2*B**2/(1+beta_e**2)

# ax_hal[0].plot(x_plot,comsol_power,label = r'COMSOL: resistance $= 7E-4$')
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    