#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:08:04 2021

@author: rajbhane
"""

import numpy as np
import pandas as pd
import difflib
import os
import sys
import time

find_key = difflib.get_close_matches

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

##########################################################################
######################### Optimal K ######################################
##########################################################################

def sens_comb_plots(file_list, sens_par_key,
                    label_list,
          save_question=0, plot_x=0, title_name='',
          plot_type=2, plot_ideal=False):
    '''
    Parameters
    ----------
    file_list : list of strings
        names of files to go through and plot, and then combine lines
        
    label_list : list of strings
        labels for the different lines
    
    The other parameters are described in the plot_data function

    Returns
    -------
    fig: plt.figure
        combined figure
    ax: plt.axes
        axes of combined figure
    '''
    ticker = 0 ## To initiate figures 
    
    for file_name in file_list:
        df_hal = pd.read_csv(file_name+'.csv',skiprows=4)
        key_hal = df_hal.keys().to_numpy()
        sp = df_hal['mob_i (m^2/(V*s))']
        
        comsol_power = abs(df_hal['ec.Ey*ec.Jy (W/m^3), Channel Faraday Power'])
        y_plot_name = r'Optimal Power Out for varying $\beta_i$'
    
        K = df_hal[find_key('abs(ec.Ey/(U_x(y)*B_z(x))) (V/m), K ', key_hal)[0]]
        # m = 4 ## to make redefining u,B,sig easier. 
        u = df_hal[find_key('Max fluid flow (m/s)',key_hal)[0]]
        B = df_hal[find_key('Max Applied Magnetic Field Strength (T)',key_hal)[0]]
        sig = df_hal[find_key('Conductivity of plasma (S/m)',key_hal)[0]]
        mob_e = df_hal['% mob_e (m^2/(V*s))']
        # beta_e = mob_e*B
        mob_i = df_hal['mob_i (m^2/(V*s))']
        beta_i = mob_e*mob_i*B**2
        beta_i = beta_i.to_numpy()
        
        ideal_power = K*(1-K)*u**2*B**2*sig
        
        ##### Plotting for r_r ######
        
        x_plot = df_hal['r_r (Ω)']
        x_plot_name = 'r_r (Ω)'
        
        rs2 = int(sp.unique().size)
        rs1 = int(x_plot.unique().size)
        
        x_plot = x_plot.to_numpy().reshape(rs2,rs1)
        comsol_power = abs(comsol_power.to_numpy().reshape(rs2,rs1))
        ideal_power = abs(ideal_power.to_numpy().reshape(rs2,rs1))
        beta_i = beta_i.reshape(rs2,rs1) ## For labelling only
        
        sp = sp.to_numpy().reshape(rs1,rs2).T
        
        if ticker == 0:
            fig1,ax1 = plt.subplots(1,1)
        
        for k in np.arange(0,rs2):
            ax1.plot(x_plot[k],comsol_power[k],
                     label = sens_par_key + ': '+str(round(beta_i[k][0],4)) 
                     + ' , ' + label_list[ticker])
            ymax = max(comsol_power[k])
            xpos = np.where(comsol_power[k] == (ymax))[0][0]
            xmax = x_plot[k][xpos]
            ax1.plot(xmax,ymax,'x')
            ## Also plot ideal
            # ax1.plot(x_plot[k],ideal_power[k],
            #          label=sens_par_key + ':'+str(round(beta_i[k][0],4)) 
            #          + ' , ' + label_list[ticker]+
            #          ', ideal')
        
        ax1.set_ylabel('Channel Faraday Power [W]')
        ax1.set_xlabel(x_plot_name)
        ax1.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        ax1.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        ax1.set_ylabel(y_plot_name)
        fig1.set_size_inches(8,8)
        
        # plt.figure(1)
        # for k in np.arange(0,rs2):
        #     plt.plot(x_plot[k],comsol_power[k],label = sens_par_key + ': '+str(round(beta_i[k][0],4)))
        #     ymax = max(comsol_power[k])
        #     xpos = np.where(comsol_power[k] == (ymax))[0][0]
        #     xmax = x_plot[k][xpos]
        #     plt.plot(xmax,ymax,'x')
        
        # plt.set_ylabel('Channel Faraday Power [W]')
        # plt.set_xlabel(x_plot_name)
        # plt.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        # plt.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        # plt.set_ylabel(y_plot_name)
        
        if ticker == 0:
            fig3,ax3 = plt.subplots(1,1)
        
        for k in np.arange(0,rs2):
            ax3.plot(x_plot[k],K,label = label_list[ticker])
        
        ax3.set_ylabel(r'K vs $r_r$')
        ax3.set_xlabel(x_plot_name)
        # ax3.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        ax3.legend(fontsize = 'x-small')#
        fig3.set_size_inches(8,8)
        
        
         #### Second Plot being created for K ####
        
        try:
            x_plot = df_hal['abs(ec.Ey/(U_x(y)*B_z(x))) (V/m), K ']
        except:
            pass
        try:
            x_plot = df_hal['abs(ec.Ey/(u_x*B_app)) (1), K ']
        except:
            print('Error in determining K.')
            raise Exception
        x_plot_name = 'K'
        
        x_plot = x_plot.to_numpy().reshape(rs2,rs1)
        
        if ticker == 0:
            fig2,ax2 = plt.subplots(1,1)
        
        for k in np.arange(0,rs2):
            ax2.plot(x_plot[k],comsol_power[k],label = sens_par_key
                     + ': '+str(round(beta_i[k][0],4))
                     + ' , ' + label_list[ticker])
            ymax = max(comsol_power[k])
            xpos = np.where(comsol_power[k] == (ymax))[0][0]
            xmax = x_plot[k][xpos]
            ax2.plot(xmax,ymax,'x')
            ## Also plot ideal
            # ax2.plot(x_plot[k],ideal_power[k],
            #          label=sens_par_key + ':'+str(round(beta_i[k][0],4)) 
            #          + ' , ' + label_list[ticker]+
            #          ', ideal')
        
        ax2.set_ylabel('Channel Faraday Power [W]')
        ax2.set_xlabel(x_plot_name)
        ax2.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        ax2.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        ax2.set_ylabel(y_plot_name)
        fig2.set_size_inches(8,8)
        
        # plt.figure(2)
        
        # for k in np.arange(0,rs2):
        #     plt.plot(x_plot[k],comsol_power[k],label = sens_par_key + ': '+str(round(beta_i[k][0],4)))
        #     ymax = max(comsol_power[k])
        #     xpos = np.where(comsol_power[k] == (ymax))[0][0]
        #     xmax = x_plot[k][xpos]
        #     plt.plot(xmax,ymax,'x')
        
        # plt.set_ylabel('Channel Faraday Power [W]')
        # plt.set_xlabel(x_plot_name)
        # plt.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        # plt.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        # plt.set_ylabel(y_plot_name)
        
        ticker += 1
    return

'''
A list of keys to use:
    mob_e (m^2/(V*s))
    mob_i (m^2/(V*s))
    r_r (Ω)
    
    
    
Some notes:
    : cw_eh = 10
    _fine: same range as above, just more points
    _2: cw_eh = 100
        any additional _k are just additional runs. Notable for R
'''


# sens_comb_plots('seg_far_optimal_r_in0',r'$\beta_i$')
sens_comb_plots(['seg_far_non_optimal_r_sweep_i0.5_1_10_fine_mesh',
                 'seg_far_non_optimal_r_sweep_i0.5_1_10_coarse_mesh']
                ,r'$\beta_i$',['Fine','Coarse'])



