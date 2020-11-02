import os
import pandas as pd
import numpy as np
import random

import time

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.mixture import GaussianMixture as EM
from sklearn.metrics import silhouette_score as sil_score, f1_score, homogeneity_score


from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict



df = pd.read_csv('diabetes.csv') # https://www.openml.org/d/37
numeric = ["preg","plas","pres","skin","insu","mass","pedi","age"]
pos_label = "tested_positive"
df_num = df[numeric]
normalized_df=(df_num-df_num.min())/(df_num.max()-df_num.min()) # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
df = df.drop(numeric, axis=1)
df_diabetes = pd.concat([df, normalized_df], axis=1)

# data_prefix = "Diabetes Dataset - "


df = pd.read_csv('credit.csv') # https://www.openml.org/d/31
pos_label = "good"

cols_qualitative = ["checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker"]
df_1hot = df[cols_qualitative]
df_1hot = pd.get_dummies(df_1hot).astype('category')
df_others = df.drop(cols_qualitative, axis=1)
df = pd.concat([df_others, df_1hot], axis=1)

cols_quantiative = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents']
df_num = df[cols_quantiative]
df_stand =(df_num-df_num.min())/(df_num.max()-df_num.min())
df_credit_categorical = df.drop(cols_quantiative,axis=1)
df_credit = pd.concat([df_credit_categorical,df_stand],axis=1)
df_credit.describe(include='all')

# data_prefix = "Credit Dataset - "



diabetes_X = np.array(df_diabetes.values[:,1:-1],dtype='int64')
diabetes_Y = np.array(df_diabetes.values[:,0],dtype='int64')
creditX = np.array(df_credit.values[:,1:-1],dtype='int64')
creditY = np.array(df_credit.values[:,0],dtype='int64')







def plot_learning_curve(clf, X, y, title="Insert Title"):
    
    n = len(y)
    train_mean = []; train_std = []
    cv_mean = []; cv_std = []
    fit_mean = []; fit_std = []
    pred_mean = []; pred_std = []
    train_sizes=(np.linspace(.05, 1.0, 20)*n).astype('int')  
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, fit_mean, pred_mean
    


def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    plt.figure()
    plt.title(title + "LC")
    plt.xlabel("# Items")
    plt.ylabel("Accuracy")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="g", label="Train Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Test Score")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title(title + " Time")
    plt.xlabel("Num. Items")
    plt.ylabel("Train Time")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="g", label="Train Time")
    plt.legend(loc="best")
    plt.show()
    

def cluster_predictions(Y,clusterLabels):
    print(Y.shape, clusterLabels.shape)
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return pred

# end helpers https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master/Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.py#L152


def run_kmeans(X,y,title):

    kclusters = list(np.arange(2,50,2))
    sil_scores = []; f1_scores = []; homo_scores = []; train_times = []

    for k in kclusters:
        start_time = timeit.default_timer()
        km = KMeans(n_clusters=k, n_init=10,random_state=100,n_jobs=-1).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        sil_scores.append(sil_score(X, km.labels_))
        y_mode_vote = cluster_predictions(y,km.labels_)
        f1_scores.append(f1_score(y, y_mode_vote))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, sil_scores)
    plt.grid(True)
    plt.xlabel('Num. Clusters')
    plt.ylabel('Silhouette')
    plt.title(title + 'k-means Silhouette')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, f1_scores)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores KMeans: '+ title)
    plt.show()
    



run_kmeans(diabetes_X,diabetes_Y, 'Diabetes Data')
km = KMeans(n_clusters=7, n_jobs=-1)


run_kmeans(creditX,creditY, 'Credit Data')
km = KMeans(n_clusters=20, n_jobs=-1)




def run_EM(X,y,title):

    kdist = list(np.arange(2,100,5))
    sil_scores = []; f1_scores = []; homo_scores = []; train_times = []; aic_scores = []; bic_scores = []
    
    for k in kdist:
        start_time = timeit.default_timer()
        em = EM(n_components=k,covariance_type='diag',n_init=1,warm_start=True,random_state=100).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        
        labels = em.predict(X)
        sil_scores.append(sil_score(X, labels))
        y_mode_vote = cluster_predictions(y,labels)
        f1_scores.append(f1_score(y, y_mode_vote))
        homo_scores.append(homogeneity_score(y, labels))
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.xlabel('# Clusters')
    plt.ylabel('Silhouette')
    plt.title(title + ' Exp Max Silhouette')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.xlabel('# Clusters')
    plt.ylabel('Silhouette')
    plt.title(title + ' Exp Max Silhouette')
    plt.show()
   

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, f1_scores)
    plt.grid(True)
    plt.xlabel('# Clusters')
    plt.ylabel('F1 Score')
    plt.title(title + 'Exp Max F1')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores,label='BIC')
    plt.grid(True)
    plt.xlabel('# Clusters')
    plt.ylabel('Model Complexity Score')
    plt.title(title + 'Exp Max Model Complexity')
    plt.legend(loc="best")
    plt.show()


run_EM(diabetes_X,diabetes_Y,'Diabetes Data')
em = EM(n_components=24, covariance_type='diag', warm_start=True, random_state=100)


X_train, X_test, y_train, y_test = train_test_split(np.array(creditX),np.array(creditY), test_size=0.25)
run_EM(X_train,y_train,'Credit Data')
em = EM(n_components=41, covariance_type='diag', warm_start=True, random_state=100)



def run_PCA(X,y,title):
    
    pca = PCA(random_state=5).fit(X) #for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax1.set_ylabel('Eigenvalues', color='g')
    ax1.tick_params('y', colors='g')
    plt.grid(False)

    plt.title(title + "PCA Eigenvalues")
    fig.tight_layout()
    plt.show()
    
def run_ICA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title(title + "ICA Kurtosis")
    plt.xlabel("ICA")
    plt.ylabel("Avg Kurtosis")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.show()

def run_RCA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i,dim in product(range(5),dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = 0
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()


    fig, ax1 = plt.subplots()
    ax1.plot(dims,mean_recon, 'b-')
    ax1.set_xlabel('RCA')
    ax1.set_ylabel('Mean RC', color='g')
    ax1.tick_params('y', colors='g')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims,std_recon, 'm-')
    ax2.set_ylabel('STD RC', color='r')
    ax2.tick_params('y', colors='r')
    plt.grid(False)

    plt.title(title + "RCA")
    fig.tight_layout()
    plt.show()
    
def run_RFC(X,y,df_original):
    rfc = RFC(n_estimators=500,min_samples_leaf=round(len(X)*.01),random_state=5,n_jobs=-1)
    imp = rfc.fit(X,y).feature_importances_ 
    imp = pd.DataFrame(imp,columns=['Feature Importance'],index=df_original.columns[2::])
    imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum']<=0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols


run_PCA(diabetes_X,diabetes_Y,"Diabetes Data")
run_ICA(diabetes_X,diabetes_Y,"Diabetes Data")
run_RCA(diabetes_X,diabetes_Y,"Diabetes Data")


X_train, X_test, y_train, y_test = train_test_split(np.array(creditX),np.array(creditY), test_size=0.2)
run_PCA(X_train,creditY,"Credit Data")
run_ICA(X_train,creditY,"Credit Data")
run_RCA(X_train,creditY,"Credit Data")


imp_diabetes, topcols_diabetes = run_RFC(diabetes_X,diabetes_Y,df_diabetes)
pca_diabetes = PCA(n_components=3,random_state=5).fit_transform(diabetes_X)
ica_diabetes = ICA(n_components=5,random_state=5).fit_transform(diabetes_X)
rca_diabetes = ICA(n_components=6,random_state=5).fit_transform(diabetes_X)
rfc_diabetes = df_diabetes[topcols_diabetes]
rfc_diabetes = np.array(rfc_diabetes.values,dtype='int64')



run_kmeans(pca_diabetes,diabetes_Y,'PCA Diabetes Data')
run_kmeans(ica_diabetes,diabetes_Y,'ICA Diabetes Data')
run_kmeans(rca_diabetes,diabetes_Y,'RCA Diabetes Data')
run_kmeans(rfc_diabetes,diabetes_Y,'RFC Diabetes Data')


run_EM(pca_diabetes,diabetes_Y,'PCA Diabetes Data')
run_EM(ica_diabetes,diabetes_Y,'ICA Diabetes Data')
run_EM(rca_diabetes,diabetes_Y,'RCA Diabetes Data')
run_EM(rfc_diabetes,diabetes_Y,'RFC Diabetes Data')





pca_credit = PCA(n_components=3,random_state=5).fit_transform(X_train)
ica_credit = ICA(n_components=6,random_state=5).fit_transform(X_train)
rca_credit = ICA(n_components=9,random_state=5).fit_transform(X_train)
rfc_credit = df_credit[topcols_credit]
rfc_credit = np.array(rfc_credit.values,dtype='int64')


print(len(pca_credit), len(ica_credit), len(rca_credit), len(rfc_credit), len(y_train), "**!@#*!@&$!@&$!@")
run_kmeans(pca_credit,y_train,'PCA Credit Data')
run_kmeans(ica_credit,y_train,'ICA Credit Data')
run_kmeans(rca_credit,y_train,'RCA Credit Data')
run_kmeans(rfc_credit[0:800],y_train,'RFC Credit Data')


run_EM(pca_credit,creditY[0:800],'PCA Credit Data')
run_EM(ica_credit,creditY[0:800],'ICA Credit Data')
run_EM(rca_credit,creditY[0:800],'RCA Credit Data')
run_EM(rfc_credit[0:800],creditY[0:800],'RFC Credit Data')







for data in [diabetes_X, pca_diabetes, ica_diabetes, rca_diabetes, rfc_diabetes]:
    X_train, X_test, y_train, y_test = train_test_split(np.array(data),np.array(diabetes_Y), test_size=0.20)
    est = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                            learning_rate_init=0.05, random_state=100)
    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(est, X_train, y_train,title="Diabetes NN Cluster Normal")




n = train_samp_full
plt.figure()
plt.title("Diabetes Dataset Learn Time")
plt.xlabel("# Items")
plt.ylabel("F1 Score")
plt.plot(n, NN_train_score_full, '-', color="c", label="Normal")
plt.plot(n, NN_train_score_pca, '-', color="g", label="PCA")
plt.plot(n, NN_train_score_ica, '-', color="y", label="ICA")
plt.plot(n, NN_train_score_rca, '-', color="r", label="RCA")
plt.plot(n, NN_train_score_rfc, '-', color="m", label="RFC")
plt.legend(loc="best")
plt.show() 


df = pd.read_csv('diabetes.csv') # https://www.openml.org/d/37
numeric = ["preg","plas","pres","skin","insu","mass","pedi","age"]
pos_label = "tested_positive"
df_num = df[numeric]
normalized_df=(df_num-df_num.min())/(df_num.max()-df_num.min()) # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
df = df.drop(numeric, axis=1)
df_diabetes = pd.concat([df, normalized_df], axis=1)



df = pd.read_csv('diabetes.csv') # https://www.openml.org/d/37
numeric = ["preg","plas","pres","skin","insu","mass","pedi","age"]
pos_label = "tested_positive"
df_num = df[numeric]
normalized_df=(df_num-df_num.min())/(df_num.max()-df_num.min()) # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
df = df.drop(numeric, axis=1)
df_diabetes = pd.concat([df, normalized_df], axis=1)


km = KMeans(n_clusters=9,n_init=10,random_state=100,n_jobs=-1).fit(diabetes_X)
km_labels = km.labels_
em = EM(n_components=24,covariance_type='diag',n_init=1,warm_start=True,random_state=100).fit(diabetes_X)
em_labels = em.predict(diabetes_X)

for data in [diabetes_X, pca_diabetes, ica_diabetes, rca_diabetes, rfc_diabetes]:
    X_train, X_test, y_train, y_test = train_test_split(np.array(data),np.array(diabetes_Y), test_size=0.20)
    est = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                            learning_rate_init=0.05, random_state=100)
    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(est, X_train, y_train,title="Diabetes NN Cluster Normal")

n = train_samp_full
plt.figure()
plt.title("Diabetes Dataset Learn Time")
plt.xlabel("# Items")
plt.ylabel("F1 Score")
plt.plot(n, NN_train_score_full, '-', color="c", label="Normal")
plt.plot(n, NN_train_score_pca, '-', color="g", label="PCA")
plt.plot(n, NN_train_score_ica, '-', color="y", label="ICA")
plt.plot(n, NN_train_score_rca, '-', color="r", label="RCA")
plt.plot(n, NN_train_score_rfc, '-', color="m", label="RFC")
plt.legend(loc="best")
plt.show() 