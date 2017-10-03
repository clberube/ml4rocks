#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:39:18 2017

@author: Charles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib.colors import LogNorm


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

loc = u'/Users/Charles/Documents/CMIC Footprints/2017 Malartic Samples/AllData_updated.csv'

df = pd.read_csv(loc,header=0, na_values=["x", "X", "#VALEUR!"])
df.set_index('Sample_ID', inplace=True)


cmaps = ["Oranges", "Greens", "Blues"]
group_colors = ["orange", "green", "lightblue"]
group_markers = ["o", "^", "s"]
groups = ["Felsic intrusive rocks", "Mafic intrusive rocks", "Sedimentary rocks"]
lithos = [['Porphyry', 'Monzodiorite', 'Granite'], ["Mafic Dyke", "Mafic Intrusion"], ['Siltstone', 'Greywacke', 'Mudstone']]

lithology = lithos[2]

df = df[df["Litho_Revised"].isin(lithology)]

Xprop = "Rock_Den"
Yprop = "Mag_Susc"

Xprop1 = "GSC_Grain_Den_gcc"
Xprop2 = "Poly_Rock_Den_gcc"

Yprop1 = "GSC_Mag_Susc_SI"
Yprop2 = "Poly_Mag_Susc_SI"

df[Xprop] = df[[Xprop1, Xprop2]].mean(axis=1).dropna()
df[Yprop] = df[[Yprop1, Yprop2]].mean(axis=1).dropna()

#den = np.log10(df[Yprop].dropna())

den = df[Xprop].dropna()

den = reject_outliers(den,4)


#plt.figure()
#plt.hist(np.log10(mag))

plt.figure(figsize=(6,4))
plt.hist(den,bins=25,normed=True)

dist_names = ['norm']

import scipy
import scipy.stats
#from scipy.stats import norm

#dist_names = ['gamma', 'lognorm', 'rayleigh', 'norm']
dist_names = ['norm']

size=len(den)
y=np.array(den)
x=np.linspace(2.6,2.9,size*10)
x=np.linspace(min(den),max(den),size*10)

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)

    param = dist.fit(y)
    print param
    pdf_fitted = dist.pdf(x,*param[:-2],loc=param[-2],scale=param[-1])
    plt.plot(x,pdf_fitted, label=dist_name)
    plt.legend()


    