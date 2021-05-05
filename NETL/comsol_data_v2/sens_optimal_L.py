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
################ Sens. of L to other parameter ###########################
##########################################################################

def sens_comb_plots(file_name, sens_par_key, 
          save_question=0, plot_x=0, title_name='',
          plot_type='all in one', plot_ideal=False, flip = 0):
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
    sp = df_hal[find_key(sens_par_key, key_hal)[0]]
    
    key_R = find_key('ec.Ex*ec.Jx+ec.Ey*ec.Jy+ec.Ez*ec.Jz (W), Resistor Power',key_hal)
    key_J = find_key('ec.Jx*ec.Ex + ec.Jy*ec.Ey + ec.Jz*ec.Ez (W), Joule Heating',key_hal)
    comsol_power = abs(df_hal[key_R[0]]) - abs(df_hal[key_J[0]])
    y_plot_name = 'Resistor Power - Joule Heating [W]'

    K = df_hal[find_key('abs(ec.Ey/(u_x*B_app)) (1), K hall', key_hal)[0]]
    # m = 4 ## to make redefining u,B,sig easier. 
    u = df_hal[find_key('plasma velocity (m/s), Point: (0, 0, 0)', key_hal)[0]]
    B = df_hal[find_key('Applied Magnetic Field (T), Point: (0, 0, 0)', key_hal)[0]]
    sig = df_hal[find_key('plasma conductivity (S/m), Point: (0, 0, 0)', key_hal)[0]]
    # mob_e = df_hal[find_key('% mob_e (m^2/(V*s))', key_hal)[0]]
    # beta_e = mob_e*B
    # mob_i = df_hal[find_key('mob_i (m^2/(V*s))', key_hal)[0]]
    # beta_i = mob_e*mob_i*B**2
    
    # print('\n K: '+K.name+'\n u: '+u.name+'\n B: '+B.name+'\n sigma: '+sig.name,
    #       '\n'+'mob_e : '+mob_e.name,'\n'+'mob_i: '+mob_i.name)
    
    ## ocl = outer_chan_len, making sure only plotting relevant L values. 
    try:
        ocl = df_hal['Insulating boundary added to ends of channel (outdated, use L now) (mm)']
        ocl = ocl.to_numpy()/1000
    except:
        print('Trying other unit')
        try:
            ocl = df_hal['Insulating boundary added to ends of channel (outdated, use L now) (m)']
            ocl = ocl.to_numpy()
        except:
            print('OCL not recovered. Check L to be sure it lies lower than OCL manually')
            ocl = np.ones(u.to_numpy().shape)*100
    
    x_plot = df_hal[key_hal[0]]
    x_plot_name = x_plot.name
    
    rs2 = int(sp.unique().size)
    rs1 = int(x_plot.unique().size)
    
    if rs2 == 1:
        flip == 2
    
    ## Try changing flip if plots look weird. 
    if flip == 0:
        x_plot = x_plot.to_numpy().reshape(rs1,rs2).T
        comsol_power = abs(comsol_power.to_numpy().reshape(rs1,rs2).T)
        sp = sp.to_numpy().reshape(rs1,rs2).T
        ocl = ocl.reshape(rs1,rs2).T
    elif flip == 1:
        x_plot = x_plot.to_numpy().reshape(rs2,rs1)
        comsol_power = abs(comsol_power.to_numpy().reshape(rs2,rs1))
        sp = sp.to_numpy().reshape(rs2,rs1)
        ocl = ocl.reshape(rs2,rs1)
    elif flip == 2:
        pass
    
    ## Redoing sp to be the K value
    if sens_par_key == 'resistance':
        sens_par_key = 'K'
        sp = df_hal['abs(ec.Ey/(u_x*B_app)) (1), K hall'].to_numpy()
        sp = np.around(sp,3)
        sp = sp.reshape(rs1,rs2).T
    elif sens_par_key == 'mob_e (m^2/V*s))':
        sp = np.around(sp,3)
    
    
    ## Actually plotting
    if plot_type == 'all in one' or plot_type == 'All in one':
        fig,ax = plt.subplots(1,1)
    
    '''
    Could add functionality to only include plots up to ocl. 
      -- Added --
    '''
    
    for k in np.arange(0,rs2):
        if plot_type == 'All on own' or plot_type == 'all on own':
            fig,ax = plt.subplots(1,1)
        if x_plot[k][x_plot[k].shape[0]-1] > ocl[k][0]:
            ind1 = np.where(x_plot[k] <= ocl[k][0])
            x_plot_2 = x_plot[k][ind1]
            y_plot_2 = comsol_power[k][ind1]
            ax.plot(x_plot_2,y_plot_2,label = sens_par_key + ': '+str(sp[k][0]))
            ymax = max(y_plot_2)
            xpos = np.where(y_plot_2 == (ymax))[0][0]
            xmax = x_plot_2[xpos]
            ax.plot(xmax,ymax,'x')
        else:
            ax.plot(x_plot[k],comsol_power[k],label = sens_par_key + ': '+str(sp[k][0]))
            ymax = max(comsol_power[k])
            xpos = np.where(comsol_power[k] == (ymax))[0][0]
            xmax = x_plot[k][xpos]
            ax.plot(xmax,ymax,'x')
            
        if plot_type == 'All on own' or plot_type == 'all on own':
             ax.set_ylabel('Net Power [W]')
             ax.set_xlabel('L [m]')
             ax.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
             ax.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
             ax.set_ylabel(y_plot_name)
    
    if plot_type == 'All in one' or plot_type == 'all in one':
        ax.set_ylabel('Net Power [W]')
        ax.set_xlabel('L [m]')
        ax.set_title(title_name)# + r' $\beta_e = $'+str(round(beta_e[a][0][0],3)))
        ax.legend(fontsize = 'x-small')#, loc = 'upper left', bbox_to_anchor=(1.05, 1))
        ax.set_ylabel(y_plot_name)
    
    return fig,ax

'''
A list of keys to use:
    sig_b (S/m)
    outer_chan_len (m)
    mob_e (m^2/(V*s))
    u_x (m/s)
    B_app (T)
    cw_eh_ratio
    
Notes:
    - You can switch the plots from all in one to all separate by changing the plot_type
        appropriately, i.e. named "all in one" or "all on own"
    - If the plots look bad, try changing flip to 0 or 1. 
    - A single run with no varying of a sens. parameter can be done, but
        you need to still select the one of the keys from above. Any should work
    
'''



### Basic sensitivity analyses
# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_ocl_sweep'
#                 ,'outer_chan_len (m)',
#                 title_name = r'Sens. Of Opt. L to ocl',flip = 1)

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_u_sweep_ocl_050'
#                 ,'u_x (m/s)', title_name = r'Sens. Of Opt. L to $u_x$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_sig_sweep_ocl_050'
#                 ,'sig_b (S/m)', title_name = r'Sens. Of Opt. L to $\sigma$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_B_sweep_ocl_050'
#                 ,'B_app (T)', title_name = r'Sens. Of Opt. L to $B_z$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_cweh_sweep_ocl_050'
#                 ,'cw_eh_ratio', title_name = r'Sens. Of Opt. L to CW:EH')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_u_sweep_ocl_025_coarse'
#                 ,'u_x (m/s)', title_name = r'Sens. Of Opt. L to $u_x$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_sig_sweep_ocl_025'
#                 ,'sig_b (S/m)', title_name = r'Sens. Of Opt. L to $\sigma$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_B_sweep_ocl_025'
#                 ,'B_app (T)', title_name = r'Sens. Of Opt. L to $B_z$')

# sens_comb_plots('./Tapered_sensitivity_analyses_v2/seg_far_cweh_sweep_ocl_025'
#                 ,'cw_eh_ratio', title_name = r'Sens. Of Opt. L to CW:EH')






































###### Older Joule Heating Definition ######

'''
    
Some notes:
    : cw_eh = 10
    _fine: same range as above, just more points
    _2: cw_eh = 100
        any additional _k are just additional runs. Notable for R
'''

## outer_chan_len = 100
sens_comb_plots('seg_far_v6_2_hall_10_L_and_u_sweep','u_x (m/s)',
                title_name = r'Sens. Of Opt. L to $u_x$')
sens_comb_plots('seg_far_v6_2_hall_10_L_and_sig_sweep', 'sig_b (S/m)',
                title_name=r'Sens. Of Opt. L to $\sigma$')
sens_comb_plots('seg_far_v6_2_hall_10_L_and_outer_chan_len_sweep_fine','outer_chan_len (m)',
                title_name = r'Sens. Of Opt. L to outer_chan_len')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_beta_e_sweep','mob_e (m^2/V*s))',
                title_name = r'Sens. Of Opt. L to $\mu_e$')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_B_sweep','B_app (T)',
                title_name = r'Sens. Of Opt. L to $B_z$')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_cw_eh_sweep','cw_eh_ratio',
                title_name = r'Sens. Of Opt. L to cw_eh_ratio')

### outer_chan_len = 500
sens_comb_plots('seg_far_v6_2_hall_10_L_and_sig_sweep_2', 'sig_b (S/m)',
                title_name=r'Sens. Of Opt. L to $\sigma$')
sens_comb_plots('seg_far_v6_2_hall_10_L_and_outer_chan_len_sweep_2','outer_chan_len (m)',
                title_name = r'Sens. Of Opt. L to outer_chan_len')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_beta_e_sweep_2','mob_e (m^2/V*s))',
                title_name = r'Sens. Of Opt. L to $\mu_e$')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_B_sweep_2','B_app (T)',
                title_name = r'Sens. Of Opt. L to $B_z$')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_R_sweep_2_2','resistance',
                title_name = r'Sens. Of Opt. L to Resistivity')
sens_comb_plots('seg_far_v6_2_hall_10_L_and_R_sweep_2_3','resistance',
                title_name = r'Sens. Of Opt. L to Resistivity')

sens_comb_plots('seg_far_v6_2_hall_10_L_and_cw_eh_sweep_2','cw_eh_ratio_2',
                title_name = r'Sens. Of Opt. L to cw_eh_ratio')




