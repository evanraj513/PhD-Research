#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:32:08 2021

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
    
def plot_data(plot_y, file, save_question = 1, plot_x = 0, file_name = 'Unnamed', plot_type=0):
    '''
    Plots the data contained in the file provided. 
    '''
    
    if file_name == 'Unnamed':
        file_name = file
    
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
    df_hal = df_hal.sort_values(by=[key_hal[1],key_hal[2]])
    
    if plot_y == 'Resistor':
        comsol_power = abs(df_hal[key_hal[7]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Joule Heating':
        comsol_power = abs(df_hal[key_hal[12]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Resistor_Joule Heating':
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
    
    try:
        x_plot = df_hal[key_hal[plot_x]] ## L_input
    except:
        # x_plot = df_hal[key_hal[0]]
        print('Error. Input not valid.')
        raise Exception
    
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
                ax_hal.set_title(file_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3))
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
            ax_hal.set_title(file_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
            ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
            
            fig_hal.set_size_inches(10,10)
            
            if saving == 1:
                    pdf.savefig(fig_hal) 
                    
                    
    if plot_type == 2:
        ax_hal.set_ylabel(y_plot_name)
        ax_hal.set_xlabel(x_plot_name)
        ax_hal.set_title(file_name)
        ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        fig_hal.set_size_inches(10,10)
        
        if saving == 1:
            pdf.savefig(fig_hal)
            
    if saving == 1:
        pdf.close()
        
        
file_name='seg_far_L_sens_run1_5'

file='../comsol_data_v2/'+file_name

plot_all = ['Joule Heating','Resistor_Joule Heating']
for x in plot_all:
    plot_data(x, file, save_question = 0, plot_x = 0, file_name = file_name+': '+x, plot_type=0)