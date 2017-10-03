#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:51:07 2017

@author: Charles
"""

# Load libraries
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib as mpl

from scipy.ndimage.filters import gaussian_filter

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import string
import plot_class_report


lm = LinearRegression()

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

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

# Filter outliers
df_den = df[["Poly_Sat_Den_gcc","GSC_Sat_Bulk_Den_gcc"]].copy()
df_den = reject_outliers(df_den,4)
df_den.dropna(inplace=True)

df_mag = np.log10(df[["Poly_Mag_Susc_SI","GSC_Mag_Susc_SI"]])
df_mag.replace([np.inf, -np.inf], np.nan, inplace=True)
df_mag = reject_outliers(df_mag,4)
df_mag.dropna(inplace=True)

mineral = 'Norm_SED_Pyrite'

litho = ['Siltstone', 'Greywacke', 'Mudstone', 'Porphyry', 'Mafic Dyke', 'Mafic Intrusion', 'Monzodiorite', 'Granite', 'Diabase']

litho_colors = ["aqua", "skyblue", "dodgerblue", "blanchedalmond", "green", "yellow", "coral", "hotpink", "purple"]

litho_marker = ["s", "s", "s", "o", "^", "^", "o", "o", "v"]

lithoDic = {k:(c,m) for k,c,m in zip(litho,litho_colors,litho_marker)}


cmaps = ["Oranges", "Greens", "Blues"]
group_colors = ["orange", "green", "lightblue"]
group_markers = ["o", "^", "s"]
groups = ["Felsic_intrusive", "Mafic_intrusive", "Sedimentary"]
lithos = [['Felsic', 'Monzodiorite', 'Granite'], ["Mafic Dyke", "Mafic Intrusion"], ['Siltstone', 'Greywacke', 'Mudstone']]

df.ix[df.Litho_Revised.isin(lithos[0]), 'Litho_Revised'] = groups[0]
df.ix[df.Litho_Revised.isin(lithos[1]), 'Litho_Revised'] = groups[1]
df.ix[df.Litho_Revised.isin(lithos[2]), 'Litho_Revised'] = groups[2]

XXXprop = 'BtChl-Mg#'
Xprop = "Rock_Den"
Yprop = "Mag_Susc"

Xprop1 = "GSC_Grain_Den_gcc"
Xprop2 = "Poly_Rock_Den_gcc"

Yprop1 = "GSC_Mag_Susc_SI"
Yprop2 = "Poly_Mag_Susc_SI"

XXprop = 'GSC_Resistivity_Ohmm'
#XXXprop = 'GSC_Chargeability_TimeDomain_ms'

Zprop = 'AR_Au_ppm'
#Zprop = 'LECO_S_pct'

#cmap = 'inferno'
cmap = 'viridis'
   
    
df[Xprop] = df[[Xprop1, Xprop2]].mean(axis=1).dropna()
df[Yprop] = np.log10(df[[Yprop1, Yprop2]].mean(axis=1)).dropna()
df[Yprop].replace([np.inf, -np.inf], np.nan, inplace=True)

df[Zprop] = pd.to_numeric(df[Zprop], errors='coerce')
df[Zprop] = np.log10(df[Zprop])

df[XXprop] = pd.to_numeric(df[XXprop], errors='coerce')
df[XXprop] = np.log10(df[XXprop])


#df[Zprop].replace([np.inf, -np.inf], np.nan, inplace=True)
#dataset = df[[Xprop, XXprop, "Litho_Revised"]].dropna()
dataset = df[[XXXprop, Yprop, "Litho_Revised"]].dropna()

#dataset = dataset[dataset["Litho_Revised"]=="Sedimentary"]

dataset = dataset[dataset.index.isin(altered_seds)]
#dataset = dataset[dataset.index.isin(altered_seds)]
#dataset = dataset[dataset.index.isin(altered_fels)]


dataset = dataset.join(df[Zprop]).dropna()

D = dataset.drop(['Litho_Revised'], axis=1)

D = reject_outliers(D,4)

D = D.dropna()

Z = D[Zprop]

D = D.drop([Zprop], axis=1)


scaler = preprocessing.StandardScaler().fit(D)
#scaler = preprocessing.MinMaxScaler().fit(D)


#dataset = dataset1[[Xprop, Yprop, "Litho_Revised"]]


poly = preprocessing.PolynomialFeatures(degree=3)
D_ = poly.fit_transform(scaler.transform(D))
#D_ = scaler.transform(D)


lm.fit(D_, Z)

predictions = lm.predict(D_)


poly_1 = preprocessing.PolynomialFeatures(degree=1)
D_1 = poly_1.fit_transform(scaler.transform(D))
D_1 = scaler.transform(D)
lm_1 = LinearRegression()


lm_1.fit(D_1, Z)

predictions_1 = lm_1.predict(D_1)

#import scipy.stats as stats
#import pylab 
#stats.probplot(pred_train - Y_train, dist="norm", plot=pylab)
#plt.title("")

NRMSE = abs(100*np.mean((Z - predictions)**2)/(Z.max()-Z.min()))
print "Quadratic model normalized RMSE: %.2f%%" %NRMSE

NRMSE_1 = abs(100*np.mean((Z - predictions_1)**2)/(Z.max()-Z.min()))
print "Linar model normalized RMSE: %.2f%%" %NRMSE_1

fig1, ax1 = plt.subplots(1, 2, figsize=(8,2.5))
plt.axes(ax1[0])
plt.scatter(10**Z, 10**predictions_1, s=10,alpha=0.7, vmin=-4, vmax=1, marker='s',label ="Norm. RMSE: %.1f%%" %NRMSE_1)
plt.plot([1e-4,1e1],[1e-4,1e1],'k-')
#plt.colorbar(label="Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1), extend='both')
plt.legend(loc=2,framealpha=1,handletextpad=0)
plt.yscale('log'), plt.xscale('log')
plt.xlabel("Measured Au (ppm)")
plt.ylabel("Predicted Au (ppm)")
plt.xlim([1e-4,1e1])
plt.ylim([1e-4,1e1])
plt.title('Linear model', fontsize=12)

plt.axes(ax1[1])
plt.scatter(10**Z, 10**predictions, s=10,alpha=0.7, vmin=-4, vmax=1, marker='s',label ="Norm. RMSE: %.1f%%" %NRMSE)
plt.plot([1e-4,1e1],[1e-4,1e1],'k-')
#plt.colorbar(label="Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1), extend='both')
plt.legend(loc=2,framealpha=1,handletextpad=0)
plt.yscale('log'), plt.xscale('log')
plt.xlabel("Measured Au (ppm)")
#plt.ylabel("Predicted Au (ppm)")
plt.xlim([1e-4,1e1])
plt.ylim([1e-4,1e1])
plt.title('Quadratic model',fontsize=12)
ax1[1].axes.get_yaxis().set_ticklabels([])
for i in range(len(ax1)):
    ax1[i].axes.set_axisbelow("True")
plt.tight_layout(pad=0., w_pad=0.5, h_pad=1)
fig1.savefig("ModelComparison.pdf", dpi=600, bbox_inches='tight')





X_train, X_test, Y_train, Y_test = model_selection.train_test_split(D, Z, test_size=0.3, random_state = 7)

X_train_, X_test_ = scaler.transform(X_train), scaler.transform(X_test)

X_train_, X_test_ = poly.fit_transform(X_train_), poly.fit_transform(X_test_)


lm.fit(X_train_, Y_train)
lm.fit(X_test_, Y_test)

pred_train = lm.predict(X_train_)
pred_test = lm.predict(X_test_)

fig3, ax3 = plt.subplots(1, 2, figsize=(8,2.5))
plt.axes(ax3[1])
plt.hist(pred_train - Y_train, alpha=0.7,edgecolor='k',label="Training set")
plt.hist(pred_test - Y_test, alpha=0.7,edgecolor='k',label="Test set")
plt.xlim([-2,2])
plt.legend(loc=2,framealpha=1,handletextpad=0,labelspacing=0)
plt.xlabel("$log$ Residuals")
plt.ylabel("Frequency")

plt.title('Quadratic model',fontsize=12)

plt.axes(ax3[0])
plt.scatter(pred_train, pred_train - Y_train, s=10, marker='s', alpha=0.7,label="Training set")
plt.scatter(pred_test, pred_test - Y_test, s=20, marker='^', alpha=0.7, label="Test set")
plt.hlines(y=0, xmin=-4, xmax=1)
plt.legend(loc=2,framealpha=1,handletextpad=0,labelspacing=0)
#plt.xscale('log')
plt.ylim([-3,3])
plt.xlim([-4,1])
plt.xlabel("$log$ Predictions")
#plt.xlabel("Frequency")
plt.ylabel("$log$ Residuals")
plt.title('Quadratic model',fontsize=12)
for i in range(len(ax3)):
    ax1[i].axes.set_axisbelow("True")
plt.tight_layout(pad=0., w_pad=0.5, h_pad=1)
fig3.savefig("ModelValidation.pdf", dpi=600, bbox_inches='tight')


np.random.seed(7)
mag_fake = np.random.normal(df[Yprop].mean(), 1*df[Yprop].std(), 10000)
res_fake = np.random.normal(df[XXprop].mean(), 1*df[XXprop].std(), 10000)
#charg_fake = np.random.normal(df[XXXprop].mean(), 2*df[XXXprop].std(), 10000)
btchl_fake = np.random.normal(df[XXXprop].mean(), 1*df[XXXprop].std(), 10000)
dens_fake = np.random.normal(df[Xprop].mean(), 1*df[Xprop].std(), 10000)

#mag_fake = np.random.uniform(df[Yprop].min(), df[Yprop].max(), 1000)
#res_fake = np.random.uniform(df[XXprop].min(), df[XXprop].max(), 1000)
#btchl_fake = np.random.uniform(df[Xprop].min(), df[Xprop].max(), 1000)

#mag_fake = np.random.uniform(-5, -2, 1000)
#res_fake = np.random.uniform(2, 5, 1000)
#btchl_fake = np.random.uniform(40, 80, 1000)

data_fake = np.vstack((btchl_fake,mag_fake)).T
#data_fake = np.vstack((btchl_fake,res_fake)).T

data_fake_ = scaler.transform(data_fake)

data_fake_ = poly.fit_transform(data_fake_)

lm.fit(D_,Z)
fake_predictions = lm.predict(data_fake_)

gold_colors = dataset["AR_Au_ppm"].loc[D.index]

fig2, ax2 = plt.subplots(1,2,figsize=(8,3))
plt.axes(ax2[0])
yi = np.linspace(btchl_fake.min(), btchl_fake.max(), 100)
xi = np.linspace(mag_fake.min(), mag_fake.max(), 100)
zi = ml.griddata(mag_fake, btchl_fake, fake_predictions, xi, yi, interp='linear')
zi = gaussian_filter(zi, 3, truncate=1)
#normi = mpl.colors.Normalize(vmin=-4, vmax=1);
cf = plt.contourf(xi, yi, zi, np.arange(-3,1.1,0.1), cmap=plt.get_cmap("viridis"), alpha=0.85,extend='both')
cs = plt.contour(xi, yi, zi, np.arange(-3,1,1), colors='k', linestyles='-', linewidths=0.5, alpha=0.7)
#plt.clabel(cs, inline=1, fontsize=10, fmt='%d')
plt.scatter(D["Mag_Susc"], D["BtChl-Mg#"], marker='s', c=gold_colors, cmap='viridis', s=10, vmin=-3, vmax=1)
#cb = plt.colorbar(cf, label="Predicted Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1))
#cb.solids.set(alpha=1)
plt.ylim([40,80])
plt.xlim([-4.5,-2.5])
plt.xticks(np.arange(-4.5,-2,0.5))
#ax2[0].axes.get_xaxis().set_ticklabels([])
plt.grid('off')
plt.ylabel("BtChl Mg# (%)")
plt.xlabel("$\log_{10}$ MS (MS in SI)")
#plt.tight_layout()

#plt.figure(figsize=(5,4))
#xi = np.linspace(res_fake.min(), res_fake.max(), 50)
#yi = np.linspace(mag_fake.min(), mag_fake.max(), 50)
#zi = ml.griddata(res_fake, mag_fake, fake_predictions, xi, yi, interp='linear')
#zi = gaussian_filter(zi, 2, truncate=1)
#cf = plt.contourf(xi, yi, zi, np.arange(-4,1,1), cmap=plt.get_cmap("viridis"), alpha=1.0,extend='both')
#plt.scatter(dataset["GSC_Resistivity_Ohmm"], dataset["Mag_Susc"], c=dataset["AR_Au_ppm"], s=15)
#plt.colorbar(cf, label="Au ($log_{10}$ ppm)")
#plt.xlim([2,5])
#plt.ylim([-5,-1])
#plt.xlabel("Resistivity ($\log_{10}$ $\Omega\cdot m$)")
#plt.ylabel("Magnetic susceptibiltiy ($\log_{10}$ SI)")

#plt.axes(ax2[1])
#xi = np.linspace(res_fake.min(), res_fake.max(), 100)
#yi = np.linspace(btchl_fake.min(), btchl_fake.max(), 100)
#zi = ml.griddata(res_fake, btchl_fake, fake_predictions, xi, yi, interp='linear')
#zi = gaussian_filter(zi, 5, truncate=1)
#cf = plt.contourf(xi, yi, zi, np.arange(-3,1.1,0.1), cmap=plt.get_cmap("viridis"), alpha=0.85,extend='both')
#cs = plt.contour(xi, yi, zi, np.arange(-3,1,2), colors='k', linestyles='-', linewidths=0.5, alpha=0.7)
##plt.clabel(cs, inline=1, fontsize=10, fmt='%d')
#plt.scatter(D["GSC_Resistivity_Ohmm"], D["BtChl-Mg#"], marker='s', c=gold_colors, s=10, cmap='viridis', vmin=-3, vmax=1)
##cb = plt.colorbar(cf, label="Predicted Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1))
##cb.solids.set(alpha=1)
#plt.grid('off')
#plt.xlim([2.5,4.5])
#plt.xticks(np.arange(2.5,5.0,0.5))
#plt.ylim([40,70])
#plt.xlabel("$\log_{10}$ Res (Res in $\Omega\cdot m$)")
#ax2[1].axes.get_yaxis().set_ticklabels([])
##ax2[1].axes.get_xaxis().set_ticklabels([])

#plt.ylabel("BtChl Mg# (%)")


#plt.axes(ax2[1])
#yi = np.linspace(btchl_fake.min(), btchl_fake.max(), 100)
#xi = np.linspace(dens_fake.min(), dens_fake.max(), 100)
#zi = ml.griddata(dens_fake, btchl_fake, fake_predictions, xi, yi, interp='linear')
#zi = gaussian_filter(zi, 3, truncate=1)
#cf = plt.contourf(xi, yi, zi, np.arange(-4,1.1,0.1), cmap=plt.get_cmap("viridis"), alpha=0.85,extend='both')
#cs = plt.contour(xi, yi, zi, np.arange(-4,2,1), colors='k', linestyles='-', linewidths=0.5, alpha=0.7)
##plt.clabel(cs, inline=1, fontsize=10, fmt='%d')
#plt.scatter(D["Rock_Den"], D["BtChl-Mg#"], marker='s', c=gold_colors, s=10, cmap='viridis', vmin=-4, vmax=1)
##cb = plt.colorbar(cf, label="Predicted Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1))
##cb.solids.set(alpha=1)
#plt.grid('off')
#plt.ylim([30,70])
#plt.xticks(np.arange(2.65,2.90,0.05))
#plt.xlim([2.65,2.85])
##plt.xlabel("Mag. susc. ($\log_{10}$ SI)")
#plt.xlabel("Rock density (g/cm$^3$)")
#ax2[1].axes.get_yaxis().set_ticklabels([])


#plt.axes(ax2[1,1])
#xi = np.linspace(res_fake.min(), res_fake.max(), 100)
#yi = np.linspace(dens_fake.min(), dens_fake.max(), 100)
#zi = ml.griddata(res_fake, dens_fake, fake_predictions, xi, yi, interp='linear')
#zi = gaussian_filter(zi, 3, truncate=1)
#cf = plt.contourf(xi, yi, zi, np.arange(-4,1.1,0.1), cmap=plt.get_cmap("viridis"), alpha=0.85,extend='both')
#cs = plt.contour(xi, yi, zi, np.arange(-4,2,1), colors='k', linestyles='-', linewidths=0.5, alpha=0.7)
##plt.clabel(cs, inline=1, fontsize=10, fmt='%d')
#plt.scatter(D["GSC_Resistivity_Ohmm"], D["Rock_Den"], marker='s', c=gold_colors, s=10, cmap='viridis', vmin=-4, vmax=1)
##cb = plt.colorbar(cf, label="Predicted Au ($log_{10}$ ppm)", ticks=np.arange(-4,2,1))
##cb.solids.set(alpha=1)
#plt.grid('off')
#plt.xlim([2.5,4.5])
#plt.xticks(np.arange(2.5,5.0,0.5))
#plt.ylim([2.65,2.85])
#plt.xlabel("Resistivity ($\log_{10}$ $\Omega\cdot m$)")
#ax2[1,1].axes.get_yaxis().set_ticklabels([])

#fig2.subplots_adjust(right=0.8)
#cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig2.colorbar(cf, ax=ax2.ravel().tolist(), label="$log_{10}$ Au (Au in ppm)", ticks=np.arange(-4,2,1))
#cb = fig2.colorbar(cf, ax=ax2, label="$log_{10}$ predicted Au (Au in ppm)", ticks=np.arange(-4,2,1))



#plt.tight_layout(pad=0., w_pad=0.5, h_pad=0)
#fig2.savefig("PredictionModel.pdf", dpi=600, bbox_inches='tight')
