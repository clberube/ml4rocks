#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:09:38 2017

@author: Charles
"""

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




#fig, ax = plt.subplots(2, 3, figsize=(10,5))

cs = []

list_headers = []
list_fmt_data = []

altered_types = [altered_seds, altered_fels, altered_mafs]
cutoffs = [0.01, 0.1, 1]


#for li in range(3):

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

#    Xprop = 'Bt-Fe#'
Xprop = 'Rock_Den'

Yprop = "Mag_Susc"

Xprop1 = "GSC_Grain_Den_gcc"
Xprop2 = "Poly_Rock_Den_gcc"

Yprop1 = "GSC_Mag_Susc_SI"
Yprop2 = "Poly_Mag_Susc_SI"

#cmap = 'inferno'
cmap = 'viridis'
   
    
df[Xprop] = df[Xprop1].fillna(df[Xprop2])
df[Yprop] = df[Yprop1].fillna(df[Yprop2])

df[Yprop] = np.log10(df[Yprop]).dropna()
df[Yprop].replace([np.inf, -np.inf], np.nan, inplace=True)

#    df[Yprop] = reject_outliers(df[Yprop],3)
#    df[Xprop] = reject_outliers(df[Xprop],3)

dataset1 = df[[Xprop, Yprop, "Code_Gold"]].dropna()
#dataset = dataset[(dataset["Litho_Revised"].isin(groups)) | (dataset["Litho_Revised"].isin(lithos[2]))]
#dataset1 = dataset1[dataset1["Code_Alteration"].isin(codes)]
#    dataset1 = dataset1[dataset1["Code_Alteration"].isin(codes[2*li:2*li+2])]

dataset1 = dataset1[dataset1.index.isin(altered_seds)]

dataset1 = dataset1[dataset1["Mag_Susc"] < -3]


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
#        print(msg)
    
mat = np.empty((50,50))
    
for i, G in enumerate(np.logspace(0.1,0.1,50):
    for j, C in enumerate(np.logspace(-4,3,50)):
#        print C
        # Make predictions on validation dataset
        knn = SVC(C=C,probability=True, kernel='rbf', class_weight='balanced',gamma=G)
        #knn = KNeighborsClassifier()
        #knn = DecisionTreeClassifier()
        #knn = GaussianNB()
        knn.fit(X_train, Y_train)
        predictions = knn.predict(X_validation)
#            print G, accuracy_score(Y_validation, predictions)
        mat[i,j] = accuracy_score(Y_validation, predictions)
#        print(confusion_matrix(Y_validation, predictions))
#        print(classification_report(Y_validation, predictions))
        
plt.imshow(mat)
plt.colorbar()