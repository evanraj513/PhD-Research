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

def sens_comb_plots(file_name, sens_par_key,
          save_question=0, plot_x=0, title_name='',
          plot_type=2, plot_ideal=False):
    '''
    Parameters
    ----------
    file_list : list of strings
        names of files to go through and plot, and then combine lines
    
    The other parameters are described in the plot_data function

    Returns
    -------
    fig: plt.figure
        combined figure
    ax: plt.axes
        axes of combined figure
    '''
    
    df_hal = pd.read_csv(file_name+'.csv',skiprows=4)
    key_hal = df_hal.keys().to_numpy()
    sp = df_hal['mob_i (m^2/(V*s))']
    
    comsol_power = abs(df_hal['ec.Ey*ec.Jy (W/m^3), Channel Faraday Power'])
    y_plot_name = r'Optimal Power Out for varying $\beta_i$'

    K = df_hal[find_key('abs(ec.Ey/(U_x(y)*B_z(x))) (V/m), K ', key_hal)[0]]
    # m = 4 ## to make redefining u,B,sig easier. 
    u = df_hal['Max fluid flow (m/s)']
    B = df_hal['Max Applied Magnetic Field Strength (T)']
    sig = df_hal['Conductivity of plasma (S/m)']
    mob_e = df_hal['% mob_e (m^2/(V*s))']
    # beta_e = mob_e*B
    mob_i = df_hal['mob_i (m^2/(V*s))']
    beta_i = mob_e*mob_i*B**2
    beta_i = beta_i.to_numpy()
    
    ##### Plotting for r_r ######
    
    x_plot = df_hal['r_r (Ω)']
    x_plot_name = 'r_r (Ω)'
    
    rs2 = int(sp.unique().size)
    rs1 = int(x_plot.unique().size)
    
    x_plot = x_plot.to_numpy().reshape(rs2,rs1)
    comsol_power = abs(comsol_power.to_numpy().reshape(rs2,rs1))
    beta_i = beta_i.reshape(rs2,rs1) ## For labelling only
    
    sp = sp.to_numpy().reshape(rs1,rs2).T
 
    fig,ax = plt.subplots(1,1)
    
    for k in np.arange(0,rs2):
        ax.plot(x_plot[k],comsol_power[k],label = sens_par_key + ': '+str(round(beta_i[k][0],4)))
        ymax = max(comsol_power[k])
        xpos = np.where(comsol_power[k] == (ymax))[0][0]
        xmax = x_plot[k][xpos]
        ax.plot(xmax,ymax,'x')
    
    ax.set_ylabel('Channel Faraday Power [W]')
    ax.set_xlabel(x_plot_name)
    ax.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
    ax.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylabel(y_plot_name)
    
     #### Second Plot being created for K ####
    
    x_plot = df_hal['abs(ec.Ey/(U_x(y)*B_z(x))) (V/m), K ']
    x_plot_name = 'K'
    
    x_plot = x_plot.to_numpy().reshape(rs2,rs1)
 
    fig,ax = plt.subplots(1,1)
    
    for k in np.arange(0,rs2):
        ax.plot(x_plot[k],comsol_power[k],label = sens_par_key + ': '+str(round(beta_i[k][0],4)))
        ymax = max(comsol_power[k])
        xpos = np.where(comsol_power[k] == (ymax))[0][0]
        xmax = x_plot[k][xpos]
        ax.plot(xmax,ymax,'x')
    
    ax.set_ylabel('Channel Faraday Power [W]')
    ax.set_xlabel(x_plot_name)
    ax.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
    ax.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylabel(y_plot_name)
    
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


sens_comb_plots('seg_far_optimal_r_in0',r'$\beta_i$')



