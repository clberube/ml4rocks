#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:51:07 2017

@author: Charles
"""

# Load libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import plot_class_report
import matplotlib.mlab as ml
from sklearn import preprocessing
import tabulate as tb
from matplotlib.colors import LinearSegmentedColormap



SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
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

df.AR_Au_ppm = pd.to_numeric(df.AR_Au_ppm, errors='coerce')
df.LECO_S_pct = pd.to_numeric(df.LECO_S_pct, errors='coerce')

mineral = 'Norm_SED_Pyrite'

litho = ['Siltstone', 'Greywacke', 'Mudstone', 'Porphyry', 'Mafic Dyke', 'Mafic Intrusion', 'Monzodiorite', 'Granite', 'Diabase']

litho_colors = ["aqua", "skyblue", "dodgerblue", "blanchedalmond", "green", "yellow", "coral", "hotpink", "purple"]

litho_marker = ["s", "s", "s", "o", "^", "^", "o", "o", "v"]

lithoDic = {k:(c,m) for k,c,m in zip(litho,litho_colors,litho_marker)}


cmaps = ["Oranges", "Greens", "Blues"]
group_colors = ["orange", "green", "lightblue"]
group_markers = ["o", "^", "s"]
codes = [['SED-UN'], ['SED-MIN', 'SED-ALT'], ['BAS-UN'], ['BAS-MIN', 'BAS-CARB','BAS-ALT', 'BAS-K'],['POR-UN'], ['POR-MIN', 'POR-ALT']]




fig, ax = plt.subplots(2, 3, figsize=(10,5))

cs = []

list_headers = []
list_fmt_data = []

altered_types = [altered_seds, altered_fels, altered_mafs]
cutoffs = [0.1, 1, 3]


for li in range(3):

    df.ix[((df.AR_Au_ppm >= cutoffs[li])), 'Code_Gold'] = "Gold"

    df.ix[((df.AR_Au_ppm < cutoffs[li])), 'Code_Gold'] = "No gold"
    
    #df.ix[((df.LECO_S_pct >= 0.1)&(df.Code_Alteration.isin(codes[5]))), 'Code_Alteration'] = "FEL-AL"
    #
    #df.ix[((df.LECO_S_pct < 0.1)&(df.Code_Alteration.isin(codes[1]))), 'Code_Alteration'] = "SED-UN"
    #
    #df.ix[((df.LECO_S_pct < 0.1)&(df.Code_Alteration.isin(codes[3]))), 'Code_Alteration'] = "MAF-UN"
    #
    #df.ix[((df.LECO_S_pct < 0.1)&(df.Code_Alteration.isin(codes[5]))), 'Code_Alteration'] = "FEL-UN"
    
    
    #df.ix[df.Code_Alteration.isin(codes[1]), 'Code_Alteration'] = "SED-AL"
    
    #df.ix[df.Code_Alteration.isin(codes[3]), 'Code_Alteration'] = "MAF-AL"
    #
    #df.ix[df.Code_Alteration.isin(codes[5]), 'Code_Alteration'] = "FEL-AL"
        
    #df.ix[df.Code_Alteration == "BAS-UN", 'Code_Alteration'] = "MAF-UN"
    #df.ix[df.Code_Alteration == "BAS-K", 'Code_Alteration'] = "MAF-UN"
    #
    #df.ix[df.Code_Alteration == "POR-UN", 'Code_Alteration'] = "FEL-UN"
    #df.ix[df.Code_Alteration == "POR-ALT", 'Code_Alteration'] = "FEL-UN"
    
    codes = ['Gold','No gold']
    
    
    
    
    #lithos = [['Felsic', 'Monzodiorite', 'Granite'], ["Mafic Dyke", "Mafic Intrusion"], ['Siltstone', 'Greywacke', 'Mudstone']]
    
    #df.ix[df.Litho_Revised.isin(lithos[0]), 'Litho_Revised'] = groups[0]
    #df.ix[df.Litho_Revised.isin(lithos[1]), 'Litho_Revised'] = groups[1]
    #df.ix[df.Litho_Revised.isin(lithos[2]), 'Litho_Revised'] = groups[2]
    
    Xprop = 'Bt-Fe#'
    Yprop = "Mag_Susc"
    
    Xprop1 = "GSC_Grain_Den_gcc"
    Xprop2 = "Poly_Rock_Den_gcc"
    
    Yprop1 = "GSC_Mag_Susc_SI"
    Yprop2 = "Poly_Mag_Susc_SI"
    
    #cmap = 'inferno'
    cmap = 'viridis'
       
        
    #df[Xprop] = df[[Xprop1, Xprop2]].mean(axis=1).dropna()
    df[Yprop] = np.log10(df[[Yprop1, Yprop2]].mean(axis=1)).dropna()
    df[Yprop].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df[Yprop] = reject_outliers(df[Yprop],3)
    df[Xprop] = reject_outliers(df[Xprop],3)
    
    dataset1 = df[[Xprop, Yprop, "Code_Gold", 'GSC_Resistivity_Ohmm']].dropna()
    #dataset = dataset[(dataset["Litho_Revised"].isin(groups)) | (dataset["Litho_Revised"].isin(lithos[2]))]
    #dataset1 = dataset1[dataset1["Code_Alteration"].isin(codes)]
#    dataset1 = dataset1[dataset1["Code_Alteration"].isin(codes[2*li:2*li+2])]
    
    dataset1 = dataset1[dataset1.index.isin(altered_seds)]

    
    Au_colors = pd.to_numeric(df["AR_Au_ppm"], errors='coerce')
    
    
    
    dataset1 = dataset1.join(np.log10(Au_colors)).dropna()
    
    dataset = dataset1[[Xprop, Yprop, "Code_Gold"]]
    
    
    
    # Split-out validation dataset
    array = dataset.values
    #X = array[:,0:2]
    
    scaler = preprocessing.StandardScaler().fit(dataset[[Xprop, Yprop]])
    X = scaler.transform(dataset[[Xprop, Yprop]])
    
    Y = array[:,2]
    validation_size = 0.33
    seed = 27
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        
    # Make predictions on validation dataset
    knn = SVC(probability=True, kernel='rbf', class_weight='balanced')
    #knn = KNeighborsClassifier()
    #knn = DecisionTreeClassifier()
    #knn = GaussianNB()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    ## Testing with fake data
    knn.fit(X, Y)
    
    X = scaler.inverse_transform(X)
    
    ymax = -1
    ymin = -5
    xmax = 70
    xmin = 10
    np.random.seed(seed=37)
    mag_fake = np.random.uniform(ymin, ymax, 100000)
    den_fake = np.random.uniform(xmin, xmax, 100000)
    
#    mag_fake = np.arange(xmin, xmax, 0.001)
#    den_fake = np.arange(xmin, xmax, 0.001)

#    mag_fake = np.random.normal(np.median(X[:,1]), 1*np.std(X[:,1]), 10000)
#    den_fake = np.random.normal(np.median(X[:,0]), 1*np.std(X[:,0]), 10000)
    
    data_fake = np.vstack((den_fake,mag_fake)).T
    data_fake = scaler.transform(data_fake)
    
    
    xi = np.linspace(xmin, xmax, 100)
    yi = np.linspace(ymin, ymax, 100)
    #xx, yy = np.meshgrid(xi,yi)
    fake_Z = knn.predict(data_fake)
    
    # Put the result into a color plot
    #fake_Z = fake_Z.reshape(xx.shape)
    
    for i in range(len(fake_Z)):            
            if fake_Z[i] == 'Gold':
                fake_Z[i] = int(1)
                
            if fake_Z[i] == 'No gold':
                fake_Z[i] = int(0)
        
#            if fake_Z[i] == 'MAF-AL':
#                fake_Z[i] = int(2)
#                
#            if fake_Z[i] == 'MAF-UN':
#                fake_Z[i] = int(3)
#        
#            if fake_Z[i] == 'SED-AL':
#                fake_Z[i] = int(4)
#       
#            if fake_Z[i] == 'SED-UN':
#                fake_Z[i] = int(5)         
                
    fake_predictions = knn.predict(data_fake)
    

    
    results_df = pd.DataFrame(dict(x=den_fake, y=mag_fake, label=fake_predictions))
    results_groups = results_df.groupby('label')
    
    
    mark = ["s","s","s","s","s","s"]
    colors = ["#1f77b4", "white", "#1f77b4", "white", "#1f77b4", "white"]
    colors = colors[2*li:2*li+2]
    mark = mark[2*li:2*li+2]

    lab = ["Felsic intrusive probability", "Mafic intrusive probability", "Sedimentary probability"]
    lab = results_groups.groups.keys()
    titles = ["Au cutoff: 0.1 ppm", "Au cutoff: 1 ppm", "Au cutoff: 3 ppm"]
    labels = ["> 0.1", "< 0.1", "> 1", "< 1", "> 3", "< 3"]
    tab_labels = ["Above", "Below", "Above", "Below", "Above", "Below"]
    labels = labels[2*li:2*li+2]
    tab_labels = tab_labels[2*li:2*li+2]

    #for c, (name, group) in enumerate(results_groups):
    #    ax[0,0].plot(group.x, group.y, marker=mark[c], color=colors[c], linestyle='', ms=8, label=name,zorder=0,alpha=0.7)
    #    ax[0,0].legend(fontsize=8, loc=4,labelspacing=0.2,handletextpad=0,framealpha=1)
    
    initial_groups = dataset.groupby("Code_Gold")

    
    which = [0,2,4]
    colormaps = ['Blues','Blues','Blues']
    
#    for i, a in enumerate(ax.flat[1:]):
#        Z = knn.predict_proba(data_fake)[:, which[i]]
#        xi = np.linspace(xmin, xmax, 100)
#        yi = np.linspace(ymin, ymax, 100)
#        zi = ml.griddata(den_fake, mag_fake, Z, xi, yi, interp='linear')
#        cf = a.contourf(xi, yi, zi, np.arange(0, 1.05, .05), cmap=colormaps[i], alpha=.3)
#        cs = a.contour(xi, yi, zi, np.arange(0.1, 1.1, .2), cmap='Greys', linestyles='-', linewidths=1, alpha=1)
#        a.clabel(cs, cs.levels[::1], inline=1, colors='k', fontsize=8, fmt='%.1f')
#    #    a.set_title(titles[i], fontsize=12)
#    #    a.grid(False)
#    
#    for a in ax.flat:
#        a.set_xlim([2.6, 3.1])
#        a.set_ylim([ymin, ymax])
#        a.set_axisbelow(False)
#        a.axes.set_axisbelow("True")
    
    levels = [0.1, 0.1, 0.1]
    label_levels = [2, 1, 1]

    Z = knn.predict_proba(data_fake)[:,0]
    xi = np.linspace(xmin, xmax, 100)
    yi = np.linspace(ymin, ymax, 100)
    zi = ml.griddata(den_fake, mag_fake, Z, xi, yi, interp='linear')
    cf = ax[1,li].contourf(xi, yi, zi, np.arange(0.0, 1.01, .01), cmap='Blues', alpha=0.3)
    cs = ax[1,li].contour(xi, yi, zi, np.arange(0.0, 1.1, levels[li]), cmap='Greys', linestyles='-', linewidths=1, alpha=1)
    ax[1,li].clabel(cs, cs.levels[::label_levels[li]], inline=1, inline_spacing=0, colors='k', fontsize=10, fmt='%.1f')

#    a.set_title(titles[i], fontsize=12)
#    a.grid(False)
    
    
    data_fake = scaler.inverse_transform(data_fake)
    
    cm = LinearSegmentedColormap.from_list('alterations', colors, N=7)
    
    fake_zi = ml.griddata(data_fake[:,0], data_fake[:,1], fake_Z, xi, yi, interp='linear')
    
    fake_zi = np.round(fake_zi)
    
    ax[0,li].contourf(xi, yi, fake_zi, 6, cmap=colormaps[li], alpha=0.3,zorder=0)
    ax[0,li].contour(xi, yi, fake_zi, 6, colors='k', linewidths=0.1, alpha=0.3, zorder=0)
    
    for c, (name, group) in enumerate(initial_groups):
        z_color = dataset1["AR_Au_ppm"][dataset1["Code_Gold"] == codes[c]]
        ax[0,li].scatter(group['Bt-Fe#'], group.Mag_Susc, s=20, color=colors[c], edgecolors="k", linewidth=0.2, alpha=0.7, marker=mark[c], label=labels[c],zorder=2-c)
    #    ax[0,0].set_title("Alterations SVM classification", fontsize=12)
        ax[0,li].legend(fontsize=10, loc=1,labelspacing=0.1,handletextpad=0,framealpha=1,markerscale=2)
    
    ax[0,li].grid(None)
    
#    y2 = np.sqrt((xi-2.55)*30)-7.6
#    ax[0,li].fill_between(xi, ymin, y2, facecolor='white', interpolate=True)
    
    
    
    arrays_table = np.array(precision_recall_fscore_support(Y_validation, predictions, average=None))
    #for i in range(len(arrays_table[-1])):
    arrays_table[-1] = arrays_table[-1].astype(int)
    #arrays_table = np.delete(arrays_table, (-1), axis=0)
    averages = [['']+['Average/Total']+list(precision_recall_fscore_support(Y_validation, predictions, average='weighted'))+[np.sum(precision_recall_fscore_support(Y_validation, predictions, average=None)[-1])]]
    averages[0].pop(-2)
    #averages = np.delete(averages, (-1), axis=1)
    
    try:
        del(tb.LATEX_ESCAPE_RULES[u'$'])
        del(tb.LATEX_ESCAPE_RULES[u'\\'])
    except:
        pass
    
    headers = ["Au cutoff", "", "Precision", "Recall", "F1 score", "Support"]
    
    formatted_data = np.concatenate((np.array([["",""]]).T, np.array([tab_labels]).T,arrays_table.T),axis=1)
    formatted_data = np.concatenate((formatted_data, averages),axis=0)
    
#    formatted_data = np.concatenate((formatted_data, averages),axis=0)

#    list_headers.append(headers)
    list_fmt_data.append(formatted_data)
    

    
print tb.tabulate(np.vstack(tuple(list_fmt_data)), headers, tablefmt="latex", floatfmt="0.2f")    
    
    
#ax[1,0].clabel(cs[0], cs[0].levels[::2], inline=1, colors='k', fontsize=12, fmt='%.1f')
#ax[1,1].clabel(cs[1], cs[1].levels[::1], inline=1, colors='k', fontsize=12, fmt='%.1f')
#ax[1,2].clabel(cs[2], cs[2].levels[::2], inline=1, colors='k', fontsize=12, fmt='%.1f')

    
for a in range(3):
    ax[0,a].axes.get_xaxis().set_ticklabels([])
    
    ax[1,a].set_xlabel("Biotite Fe#")
    
    ax[0,a].set_title(titles[a], fontsize=14)
#    ax[1,a].grid(False)

for a in range(2):
#    ax[a,1].set_ylim([-5, ymax])
    ax[0,1].axes.get_yaxis().set_ticklabels([])
    ax[a,1].axes.get_yaxis().set_ticklabels([])

    ax[0,2].axes.get_yaxis().set_ticklabels([])
    ax[a,2].axes.get_yaxis().set_ticklabels([])
    
    ax[a,0].set_ylabel("$\log_{10}$ MS (MS in SI)")
    ax[a,0].set_ylabel("$\log_{10}$ MS (MS in SI)")
    
    for b in range(3):
        ax[a,b].set_xlim([10,70])
#        ax[a,b].set_xlim([10,70])
        ax[a,b].set_xticks(np.arange(10,80,10))

#    ax[a,1].set_xlim([2.5,2.9])
#    ax[a,1].set_xlim([2.5,2.9])
#    ax[a,2].set_xlim([2.7,3.1])
#    ax[a,2].set_xlim([2.7,3.1])


for n, a in enumerate(ax.flat):
    a.text(-0.1, 1.05, string.ascii_uppercase[n], transform=a.transAxes, 
            size=14, weight='bold')
    
fig.tight_layout(pad=0., w_pad=0.5, h_pad=1.5)

#fig.colorbar(cf, ax=ax.ravel().tolist(), ticks=np.arange(0, 1.1, .1), label="Probability")
#fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig("GoldClassification.pdf", dpi=600, bbox_inches="tight")



