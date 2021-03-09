#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:32:08 2021

@author: evanraj
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
    
def plot_data(plot_y, file, save_question = 1, plot_x = 0, file_name = 'Unnamed', plot_type=0, plot_ideal=False, title_name = ''):
    '''
    Plots the data contained in the file provided. 
    
    plot_y: str
        name of string trying to plot as y
    file: str
        name of csv file containing data
    save_question: binary
        turn on to save figures automatically
    file_name: str
        what the title name of the plot should be
    plot_type: 0,1,2
        0 == all separate plots
        1 == all with same beta_e overlaid
        2 == all on same plot
    plot_ideal: Binary
        Turns on and off the ideal plots. Will also auto turn on the 2 plots. 
    title_name: str
        overrides file_name for title
    
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
    df_hal = df_hal.sort_values(by=[key_hal[1],key_hal[2]])
    
    if plot_y == 'Resistor':
        key = find_key('ec.Ex*ec.Jx+ec.Ey*ec.Jy+ec.Ez*ec.Jz (W), Resistor Power',key_hal)
        comsol_power = abs(df_hal[key[0]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Joule Heating':
        key = find_key('ec.Jx*ec.Ex + ec.Jy*ec.Ey + ec.Jz*ec.Ez (W), Joule Heating',key_hal)
        comsol_power = abs(df_hal[key[0]])
        y_plot_name = comsol_power.name
    elif plot_y == 'Resistor-Joule Heating' or plot_y == 'Resistor - Joule Heating':
        key_R = find_key('ec.Ex*ec.Jx+ec.Ey*ec.Jy+ec.Ez*ec.Jz (W), Resistor Power',key_hal)
        key_J = find_key('ec.Jx*ec.Ex + ec.Jy*ec.Ey + ec.Jz*ec.Ez (W), Joule Heating',key_hal)
        comsol_power = abs(df_hal[key_R[0]]) - abs(df_hal[key_J[0]])
        y_plot_name = 'Resistor Power - Joule Heating [W]'
    else:
        try:
            key = find_key(plot_y, key_hal)
            comsol_power = abs(df_hal[key[0]])
            y_plot_name = plot_y
        except:
            print('Error. Did not find key given in y_plot')
            raise Exception
                
    
    K = df_hal[find_key('abs(ec.Ey/(u_x*B_app)) (1), K hall', key_hal)[0]]
    # m = 4 ## to make redefining u,B,sig easier. 
    u = df_hal[find_key('plasma velocity (m/s), Point: (0, 0, 0)', key_hal)[0]]
    B = df_hal[find_key('Applied Magnetic Field (T), Point: (0, 0, 0)', key_hal)[0]]
    sig = df_hal[find_key('plasma conductivity (S/m), Point: (0, 0, 0)', key_hal)[0]]
    mob_e = df_hal[find_key('% mob_e (m^2/(V*s))', key_hal)[0]]
    beta_e = mob_e*B
    mob_i = df_hal[find_key('mob_i (m^2/(V*s))', key_hal)[0]]
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
    
    # x_plot_name = x_plot.name
    x_plot_name = 'L [m]'
    
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
        if plot_ideal == False:
            fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
        else:
            fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
    
    for a in np.arange(0,rs1):
        # fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
        if plot_type == 1:
            if plot_ideal == False:
                fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
            else:
                fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
        for b in np.arange(0,rs2):
            
            if plot_type == 0:
                if plot_ideal == False:
                    fig_hal,ax_hal = plt.subplots(1,1,sharex = True)
                    ax_hal.plot(x_plot[a][b],comsol_power[a][b])
                    ax_hal.set_ylabel(y_plot_name)
                    ax_hal.set_xlabel(x_plot_name)
                    ax_hal.set_title(title_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3))
                                     +r' $\beta_i = $'+str(round(beta_i[a][b][0],3)))
                    # ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
                    fig_hal.set_size_inches(10,10)
                else:
                    fig_hal,ax_hal = plt.subplots(2,1,sharex = True)
                    ax_hal[0].plot(x_plot[a][b],comsol_power[a][b])
                    ax_hal[0].set_ylabel(y_plot_name)
                    ax_hal[0].set_title(title_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3))
                                     +r' $\beta_i = $'+str(round(beta_i[a][b][0],3)))
                    
                    ax_hal[1].set_xlabel(x_plot_name)
                    ax_hal[1].set_ylabel('Rel. Difference')
                    ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b], '-x')
                    
                    ax_hal[1].plot(x_plot[a][b], diff_hal[a][b])
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
                if plot_ideal == False:
                    ax_hal.plot(x_plot[a][b],comsol_power[a][b], label = r'COMSOL: '+
                            r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
                else:
                    ax_hal[0].plot(x_plot[a][b],comsol_power[a][b], label = r'COMSOL: '+
                            r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))

                    ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b], '-x', label = r'Ideal Power: '+
                            r'$\beta_i = $'+str(round(beta_i[a][b][0],2)))
                    
                    ax_hal[1].plot(x_plot[a][b], diff_hal[a][b])

            elif plot_type == 2:
                
                if plot_ideal == False:
                    ax_hal.plot(x_plot[a][b], comsol_power[a][b],'-x',
                            label = r'COMSOL: '+
                            r'$\beta_i = $'+str(round(beta_i[a][b][0],2))+
                            r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))
                else:
                    ax_hal[0].plot(x_plot[a][b], comsol_power[a][b],'-',
                                   label = r'COMSOL: '+
                                   r'$\beta_i = $'+str(round(beta_i[a][b][0],2))+
                                   r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))

                    ax_hal[0].plot(x_plot[a][b], power_ideal_hal[a][b], '-x',
                                    label = r'COMSOL: '+
                                    r'$\beta_i = $'+str(round(beta_i[a][b][0],2))+
                                    r', $\beta_e = $'+str(round(beta_e[a][0][0],3)))
                    
                    ax_hal[1].plot(x_plot[a][b], diff_hal[a][b])
        
    
        # ax_hal[0].set_ylabel('Power Output [W]')
        # ax_hal[0].set_title(r'Con. Faraday, Sensitivity Analysis, L, $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        # ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        # ax_hal[1].set_ylabel('Rel. Diff')
        # ax_hal[1].set_xlabel(r'L [m]')
        # ax_hal[1].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        
        if plot_type == 1:
            if plot_ideal == False:
                ax_hal.set_ylabel(y_plot_name)
                ax_hal.set_xlabel(x_plot_name)
                ax_hal.set_title(title_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
                ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
                
                fig_hal.set_size_inches(10,10)
            else:
                ax_hal[0].set_ylabel(y_plot_name)
                ax_hal[1].set_xlabel(x_plot_name)
                ax_hal[0].set_title(title_name + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
                ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
                ax_hal[1].set_ylabel('Rel. Difference')
                
                fig_hal.set_size_inches(10,10)
            
            if saving == 1:
                    pdf.savefig(fig_hal) 
                    
                    
    if plot_type == 2:
        if plot_ideal == False:
            ax_hal.set_ylabel(y_plot_name)
            ax_hal.set_xlabel(x_plot_name)
            ax_hal.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
            ax_hal.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
            
            fig_hal.set_size_inches(10,10)
        else:
            ax_hal[0].set_ylabel(y_plot_name)
            ax_hal[1].set_xlabel(x_plot_name)
            ax_hal[0].set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
            ax_hal[0].legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
            ax_hal[1].set_ylabel('Rel. Difference')
            
            fig_hal.set_size_inches(10,10)
        
        if saving == 1:
            pdf.savefig(fig_hal)
            
    if saving == 1:
        pdf.close()
        
    return fig_hal, ax_hal
        
        
# file_name='seg_far_L_sens_run1_5'
# file_name = 'seg_far_v7_linear_B_run1'
file_name = 'seg_far_v6_2_hall_10_L_sweep'
# file_name = 'seg_far_v6_2_L_020_100_run1'

file='../comsol_data_v2/'+file_name

plot_y_str = 'Resistor-Joule Heating'
# plot_y_str = file_name
# plot_y_str = 'ec.Ey*ec.Jy (W/m^3), Faraday Power'

# plot_all = ['Joule Heating','Resistor_Joule Heating']
# for x in plot_all:
#     plot_data(x, file, save_question = 0, plot_x = 0, file_name = file_name+': '+x, plot_type=2,plot_ideal=True)

# fig_hal, ax_hal = plot_data(plot_y_str, 
#           file, save_question=0, plot_x=0, title_name='Segmented Faraday:'+file_name,
#           plot_type=2, plot_ideal=False)

def comb_plots(file_list, plot_y_str, 
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
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    
    count = 0
    
    for k in file_list:
        fig1,ax1 = plot_data(plot_y_str, 
          k, save_question=0, plot_x=0, title_name='Segmented Faraday:'+file_name,
          plot_type=2, plot_ideal=False)
        close(fig1)
        line1 = ax1.get_lines()
        handles,labels = ax1.get_legend_handles_labels()
        # for j in np.count_nonzero(line1):
        elec_num = 2**(count+2)
        x_plot = line1[0].get_data()[0]
        y_plot = line1[0].get_data()[1]
        ax.plot(x_plot,y_plot,
                label = labels[0]+': '+str(elec_num)+' Electrodes')
        ymax = max(line1[0].get_data()[1])
        xpos = np.where(line1[0].get_data()[1] == (ymax))[0][0]
        xmax = line1[0].get_data()[0][xpos]
        
        ax.plot(xmax,ymax,'x')
        
        if count == 0:
            ax.plot(x_plot,np.ones(x_plot.shape)*1.622E6,'--',label='Bz constant, 4 electrodes')
        if count == 1:
            ax.plot(x_plot,np.ones(x_plot.shape)*3.003E6,'--', label='Bz constant, 8 electrodes')
        if count == 2:
            ax.plot(x_plot,np.ones(x_plot.shape)*5.705E6,'--', label='Bz constant, 16 electrodes')
        count += 1
        
        
    ax.set_ylabel(plot_y_str)
    ax.set_xlabel('L [m]')
    ax.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
    ax.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylabel('Rel. Difference')
    
    return fig,ax

file_list = ['../comsol_data_v2/'+'seg_far_v6_2_hall_10_L_fine_sweep',
             '../comsol_data_v2/'+'seg_far_v6_3_hall_10_L_fine_sweep',
             '../comsol_data_v2/'+'seg_far_v6_4_hall_10_L_fine_sweep']

fig,ax = comb_plots(file_list,plot_y_str, 
           save_question=0, plot_x=0, title_name='Segmented Faraday:'+file_name,
           plot_type=2, plot_ideal=False)












