#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:25:33 2020

@author: evanraj
"""

import numpy as np
# import pandas as pd
import os
import sys
# import time

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('x-small')

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
    
from math import factorial as fact
import scipy.integrate as integrate
from scipy.special import gamma
from mpmath import beta
from mpmath import betainc
from sympy import Float
    
    
###############################################################
         ################ Attempt 1 ###################
###############################################################

# Le = 2 ## Setting Linear Length
# Bz_max = 1 ## Setting RHS BC
# n = 111 ## Number of sample points
# m = Bz_max/Le

# L = 4 ## Length to be tested
# hl = L/n ## Uniform Discretization step-size, needed to calc area

# R = L/Le ## Important Ratio

# h1 = (L-1/R)/((n-1)/2) ## First step-size
# h2 = (1/R)/((n-1)/2) ## Second step-size

# ticker = 0 ## Global k counter

# x_plot = np.zeros(n) ## Holds the discretization values, built up piece-wise
# y_plot = np.copy(x_plot) ## Holds the y values for each x_k

# for k in np.arange(0,(n-1)/2): ## First section
#     k = int(k)
#     # x_k = h1*k ## Left end-point of subinterval
#     x_k2 = h1*(k+1) ## Right end-point of subinterval
    
#     x_plot[k+1] = x_k2
    
#     A_k = m/2*hl**2 * (2*ticker+1) ## Area under the curve
#     a_k = A_k - h1*y_plot[k] ## Triangular area at top
    
#     dy = 2*a_k/h1 ## Change in y
#     y_plot[k+1] = y_plot[k]+dy ## New y max
    
#     ticker += 1

# x_t = x_plot[ticker] ## where it transitions
    
# for k in np.arange(1, (n-1)/2):
#     k = int(k)
    
#     x_k2 = h2*k + x_t
    
#     x_plot[int(ticker)+1] = x_k2 
    
#     A_k = m/2*hl**2 * (2*ticker+1) ## Area under the curve
#     a_k = A_k - h2*y_plot[int(ticker)] ## Triangular area at top
    
#     dy = 2*a_k/h2 ## Change in y
#     y_plot[int(ticker)+1] = y_plot[int(ticker)]+dy ## New y max
    
#     ticker += 1
    
# plt.plot(x_plot[0:n-1], y_plot[0:n-1])

'''
This attempt did not work well. It did not hold the maximum requirement
'''


###############################################################
         ################ Attempt 2 ###################
###############################################################
'''
This will attempt to minimize the difference of area between a CDF of the 
beta distribution and a given area
'''

Le = 1 ## ~Linear length
Bz_max = 1 ## Maximum Bz value

L = 1.5 ## Length being used to generate function

A = 1/2*Le*Bz_max

def beta_pdf(m,s,t):
    n = m*(1-m)/(s**2)
    a = (m*n)
    b = ((1-m)*n)
    
    # print(n,a,b)
    
    num = t**(a-1)*(1-t)**(b-1)
    den = beta(a,b) ## I believe this works well, based on some tests with Desmos
    
    return float(num/den)


def beta_cdf(m,s,x):
    '''
    beta distribution cdf function generator, for a given:
    
    Parameters
    ----------
    m = mean
    s = std
    c = area adjuster
    L = length cdf needs to rise to 1 from 0 (i.e. support of cdf)
    x = where in (0,L) are you?
    '''
    
    if x < 0 or x > L:
        print('!'*40,'\n','Error. x value outside of domain. Aborting','\n','!'*40)
        raise Exception
    elif x == 0:
        return 0
    elif x == L:
        return Bz_max
    
    def f(z):
        return beta_pdf(m,s,z)
    
    return Bz_max*integrate.quad(f,0,x/L)[0]
'''
Use mpmath.betainc(a,b,0,x/L)/mpmath.beta(a,b) instead for more precision(?) 
'''
    
def opt_func(m,s):
    '''
    Optimization function, to find optimal m,s,c that minimzes 
    the difference between A and the integral of beta_cdf(m,s) from 0 to L
    '''
    
    n = m*(1-m)/(s**2)
    a = (m*n)
    b = ((1-m)*n)
    
    # def g(z):
        # return float((betainc(a,b,0,z)/beta(a,b)).real)
    def g(z):
        return beta_cdf(m,s,z)
        
    
    A2 = integrate.quad(g,0,L)[0]
    
    return A2-A
    










































