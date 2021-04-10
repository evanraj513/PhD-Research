#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:27:37 2021

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
    
def plot_data(plot_y, file, save_question = 0, plot_x = 0, file_name = 'Unnamed', plot_type=0, plot_ideal=False, title_name = ''):
    '''
    Plots the data contained in the file provided. 
    '''
    
    if file_name == 'Unnamed':
        file_name = file
        
    if title_name == '':
        title_name = file_name
    
    # plot_type = 0
    '''
    Sets the way plots are viewed: 0 = all separate, 1 = separated by diff. beta_e, 2 = all together
    '''
    
    saving = save_question ## saves pdf in output file, named by file_name. 0 = off, 1 = on. 
    
    if saving == 1:
        # pdf = matplotlib.backends.backend_pdf.PdfPages('../saved_plots_2/'+file_name+'_'+ plot+'_plot.pdf')
        # pdf = matplotlib.backends.backend_pdf.PdfPages(file_name+'_'+ plot+'plot.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(file+'_plot.pdf')
    
    df_hal = pd.read_csv(file+'.csv',skiprows=4)
    key_hal = df_hal.keys().to_numpy()
    # df_hal = df_hal.sort_values(by=[key_hal[1],key_hal[2]])
    
    if plot_y == 'Resistor':
        key = find_key('ec.Ex*ec.Jx+ec.Ey*ec.Jy+ec.Ez*ec.Jz (W), Resistor Power',key_hal)
        comsol_power = abs(df_hal[key[0]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Joule Heating':
        key = find_key('ec.Jx*ec.Ex + ec.Jy*ec.Ey + ec.Jz*ec.Ez (W), Joule Heating',key_hal)
        comsol_power = abs(df_hal[key[0]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Resistor_Joule Heating':
        key_R = find_key('ec.Ex*ec.Jx+ec.Ey*ec.Jy+ec.Ez*ec.Jz (W), Resistor Power',key_hal)
        key_J = find_key('ec.Jx*ec.Ex + ec.Jy*ec.Ey + ec.Jz*ec.Ez (W), Joule Heating',key_hal)
        comsol_power = abs(df_hal[key_R[0]]) - abs(df_hal[key_J[0]])
        y_plot_name = 'Resistor Power - Joule Heating [W]'
    else:
        try:
            key = find_key(plot_y, key_hal)
            comsol_power = abs(df_hal[plot_y])
            y_plot_name = 'Power [W]'
        except:
            print('Error. Did not find key given in y_plot')
            raise Exception
                
    
    K = df_hal['abs(ec.Ey/(u_x*B_app)) (1), K hall']
    # m = 4 ## to make redefining u,B,sig easier. 
    u = df_hal['plasma velocity (m/s), Point: (0, 0, 0)']
    B = df_hal['Applied Magnetic Field (T), Point: (0, 0, 0)']
    sig = df_hal['plasma conductivity (S/m), Point: (0, 0, 0)']
    mob_e = df_hal[key_hal[0]]
    beta_e = mob_e*B
    mob_i = df_hal['mob_i (m^2/(V*s))']
    beta_i = mob_e*mob_i*B**2
    
    
    print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
          '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name,
          '\n'+'Plotting: '+y_plot_name)
    
    power_ideal_hal = K*(1-K)*sig*u**2*B**2*((1-beta_i)/((1-beta_i)**2+beta_e**2))
    
    try:
        x_plot = beta_e ## L_input
    except:
        # x_plot = df_hal[key_hal[0]]
        print('Error. Input not valid.')
        raise Exception
    
    x_plot_name = r'$\beta_e$ []'
    
    rs1 = int(mob_i.unique().size) ## Number of mob_i tested
    rs2 = int(beta_e.shape[0]/rs1)
    
    mob_e = mob_e.to_numpy().reshape(rs2,rs1).T
    beta_e = beta_e.to_numpy().reshape(rs2,rs1).T
    beta_i = beta_i.to_numpy().reshape(rs2,rs1).T
    x_plot = x_plot.to_numpy().reshape(rs2,rs1).T
    comsol_power = comsol_power.to_numpy().reshape(rs2,rs1).T
    power_ideal_hal = power_ideal_hal.to_numpy().reshape(rs2,rs1).T
    sig = sig.to_numpy().reshape(rs2,rs1).T
    
    diff_hal = abs(comsol_power - power_ideal_hal)/power_ideal_hal
    diff_hal = diff_hal.reshape(rs2,rs1).T
    

    fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
    
    for a in np.arange(0,rs1):
        # fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
        ax_hal[0].semilogx(x_plot[a], comsol_power[a],'-',
                       label = r'COMSOL: ')

        ax_hal[0].semilogx(x_plot[a], power_ideal_hal[a], '-x',
                        label = r'Ideal Power ')
        
        ax_hal[1].semilogx(x_plot[a], diff_hal[a])

    ax_hal[0].set_ylabel(y_plot_name)
    ax_hal[1].set_xlabel(x_plot_name)
    ax_hal[0].set_title(title_name+r' $\mu_i$: '+str(round(mob_i.to_numpy()[a],4)))
    ax_hal[0].legend(fontsize = 'x-small', loc = 'upper center', bbox_to_anchor=(0.9, 1))
    ax_hal[1].set_ylabel('Rel. Difference')
            
    fig_hal.set_size_inches(10,10)
    
    if saving == 1:
        pdf.savefig(fig_hal)
            
    if saving == 1:
        pdf.close()
        
file_name = 'seg_far_both_beta_sweep_1'
file='../comsol_data_v2/'+file_name

# file_name = 'evan_both_beta_seg_far_run2'
# file='../Imp_COMSOL_data/'+file_name

plot_data('ec.Ey*ec.Jy (W/m^3), Faraday Power', file, save_question=0, plot_x=0, title_name='Segmented Faraday: Faraday Power',
          plot_type=2, plot_ideal=False)
        