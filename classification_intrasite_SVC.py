
# coding: utf-8

# In[1]:


import nilearn
from nilearn import input_data
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.decomposition import DictLearning
from nilearn.regions import RegionExtractor

import sklearn
from sklearn import svm
from sklearn import model_selection

import scipy
import scipy.cluster

import bct

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import os

#from pathlib import Path

from unipath import Path
from scipy import io as sio
from pygsp import graphs


# Computation the differents statistical measures of the time series for each ROI


from nilearn.connectome import ConnectivityMeasure
from scipy.stats import moment
datadir1 = Path('Time_serie_per_site')

filelists1 = os.listdir(datadir1)
ts_eachsites=[]
ts_eachsites_TC=[]
ts_eachsites_TC_all=[]
ts_eachsites_ASD=[]
ts_eachsites_ASD_all=[]
ts_allsites=[]

allstd_TC=[]
allvar_TC=[]
allmean_TC=[]
allkurtosis_TC=[]

allstd_ASD=[]
allvar_ASD=[]
allmean_ASD=[]
allkurtosis_ASD=[]

eig_cen=[]


ts_std_eachsites=[]
ts_std_allsites=[]
connectivity_biomarkers= []

for k,curfile in enumerate(filelists1):
    sitename=curfile[5:].replace("time_series.npz", "")
    print(sitename)
    LABELS = np.load(os.path.join(datadir1,curfile))['labels']
    print(LABELS.shape)
    
    ts_eachsites=np.load(os.path.join(datadir1,curfile))['abide_ts']
    ts_eachsites_TC=np.load(os.path.join(datadir1,curfile))['abide_ts_normal']
    ts_eachsites_ASD=np.load(os.path.join(datadir1,curfile))['abide_ts_autism']
    
    
    for nor in  ts_eachsites_TC:
        allstd_TC.append(np.std(nor,axis=0))
        allvar_TC.append(np.var(nor,axis=0))
        allmean_TC.append(np.mean(nor,axis=0))
        allkurtosis_TC.append(moment((nor),axis=0, moment=4))
        ts_eachsites_TC_all.append(nor)
        
    abide_ts_std_TC_all = np.stack(allstd_TC)
    abide_ts_var_TC_all=np.stack(allvar_TC)
    abide_ts_mean_TC_all=np.stack(allmean_TC)
    abide_ts_kurtosis_TC_all=np.stack(allkurtosis_TC)
    
    
    for aut in ts_eachsites_ASD:
        allstd_ASD.append(np.std(aut,axis=0))
        allvar_ASD.append(np.var(aut,axis=0))
        allmean_ASD.append(np.mean(aut,axis=0))
        allkurtosis_ASD.append(moment((aut),axis=0, moment=4))
        ts_eachsites_ASD_all.append(aut)
    
    abide_ts_std_ASD_all = np.stack(allstd_ASD)
    abide_ts_var_ASD_all=np.stack(allvar_ASD)
    abide_ts_mean_ASD_all=np.stack(allmean_ASD)
    abide_ts_kurtosis_ASD_all=np.stack(allkurtosis_ASD)
    
    

abide_ts_std=np.concatenate((abide_ts_std_TC_all,abide_ts_std_ASD_all))   
abide_ts_var=np.concatenate((abide_ts_var_TC_all,abide_ts_var_ASD_all)) 
abide_ts_mean=np.concatenate((abide_ts_mean_TC_all,abide_ts_mean_ASD_all))
abide_ts_kurtosis=np.concatenate((abide_ts_kurtosis_TC_all,abide_ts_kurtosis_ASD_all))

print(abide_ts_std.shape)
print(abide_ts_var.shape)
print(abide_ts_mean.shape)
print(abide_ts_kurtosis.shape)





LABELS_ALL=np.concatenate((np.zeros(len(abide_ts_std_TC_all)),np.ones(len(abide_ts_std_ASD_all))))
LABELS_ALL.shape


# ### Computation of functionnel connectivity for all

# In[4]:


ts_allsites=np.concatenate((ts_eachsites_TC_all,ts_eachsites_ASD_all))
from nilearn.connectome import ConnectivityMeasure
connectivity_biomarkers = {}
kinds = ['correlation', 'partial correlation', 'tangent']
for kind in kinds:
    #conn_measure = ConnectivityMeasure(kind=kind,vectorize=True)
    conn_measure = ConnectivityMeasure(kind=kind)
    connectivity_biomarkers[kind] = conn_measure.fit_transform(ts_allsites)

# For each kind, all individual coefficients are stacked in a unique 2D matrix.
print('{0} correlation biomarkers for each subject.'.format(
    connectivity_biomarkers['tangent'].shape[1]))


#### Complex graph metrics
eig_cens=[]
clusterings=[]
Node_strengths=[]

for i in range(len(ts_allsites)):
    Node_strength=bct.strengths_und(connectivity_biomarkers['tangent'][i])
    Node_strengths.append(Node_strength)
    eig_cen = bct.centrality.eigenvector_centrality_und(connectivity_biomarkers['tangent'][i])
    eig_cens.append(eig_cen)
    clustering=bct.clustering_coef_wd(connectivity_biomarkers['tangent'][i])
    clusterings.append(clustering)
    
Node_strengths = np.stack(Node_strengths)
eig_cens=np.stack(eig_cens)
clusterings=np.stack(clusterings)


from nilearn.connectome import sym_matrix_to_vec
mat_connectivity= []


matrix=connectivity_biomarkers['tangent']

    
for mat in matrix:
    mat_connectivity.append(sym_matrix_to_vec(mat,discard_diagonal=True))
mat_connectivity = np.stack(mat_connectivity)

print(mat_connectivity.shape)


# ### Projection of statistical features

# In[6]:


## structural connectivity

from pathlib import Path
from scipy import io as sio
from pygsp import graphs


#connectivity = sio.loadmat('SC.mat')['SC']
connectivity = sio.loadmat('SC_avg56.mat')['SC_avg56']
print(connectivity.shape)
coordinates = sio.loadmat('Glasser360_2mm_codebook.mat')['codeBook']
G = graphs.Graph(connectivity,gtype='HCP subject',lap_type='normalized',coords=coordinates)
G.set_coordinates(kind='spring')
G.plot()





## Projection of  signals on structural graph

G.compute_fourier_basis()
signals_fourier_std=[]
signals_fourier_var=[]
signals_fourier_mean=[]
signals_fourier_eig_cens=[]
signals_fourier_clusterings=[]
signals_fourier_Node_strengths=[]
signals_fourier_kurtosis=[]
for i in range(len(abide_ts_std)):
    signal_fourier_std = G.gft(abide_ts_std[i])
    signal_fourier_mean = G.gft(abide_ts_mean[i])
    signal_fourier_var=G.gft(abide_ts_var[i])
    signal_fourier_eig_cens = G.gft(eig_cens[i])
    signal_fourier_clusterings = G.gft(clusterings[i])
    signal_fourier_Node_strengths = G.gft(Node_strengths[i])
    signal_fourier_kurtosis = G.gft(abide_ts_kurtosis[i])
    
    signals_fourier_std.append(signal_fourier_std)
    signals_fourier_mean.append(signal_fourier_mean)
    signals_fourier_var.append(signal_fourier_var)
    signals_fourier_eig_cens.append(signal_fourier_eig_cens )
    signals_fourier_clusterings.append(signal_fourier_clusterings)
    signals_fourier_Node_strengths.append(signal_fourier_Node_strengths)
    signals_fourier_kurtosis.append(signal_fourier_kurtosis)
    
signals_fourier_std=np.stack(signals_fourier_std)
signals_fourier_mean=np.stack(signals_fourier_mean)
signals_fourier_var=np.stack(signals_fourier_var)
signals_fourier_eig_cens=np.stack(signals_fourier_eig_cens)
signals_fourier_clusterings=np.stack(signals_fourier_clusterings)
signals_fourier_Node_strengths=np.stack(signals_fourier_Node_strengths)
signals_fourier_kurtosis=np.stack(signals_fourier_kurtosis)

print(signals_fourier_std.shape)
print(signals_fourier_mean.shape)
print(signals_fourier_var.shape)
print(signals_fourier_eig_cens.shape)
print(signals_fourier_clusterings.shape)
print(signals_fourier_Node_strengths.shape)
print(signals_fourier_kurtosis.shape)

# In[9]:


def classification_metrics_vis_v1(nperm,data1,data2,data3,labels,step_min,step_max,step):
    """
     Computes the accuracy,sensitivity, specificity and pvalue of permutation test using
     kbest as feature selection method for tow datasets.
    ---
    Params:
    nperm (int): number of permutation of the labels.
    data = the data that we want to classify
    step_min:mimimal number of features to select
    step_max: maximal number of features to select
    step: step of the selection of features
    ---
    Returns:
    The classification metrics, its STD and its log p value after permutation of labels
    ---
    
    """
    import tqdm as tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score,permutation_test_score,cross_val_predict
    from scipy.stats import ttest_ind
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import make_scorer
    from imblearn.metrics import  specificity_score,sensitivity_score
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit

    loo=StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    
    
    svc = LinearSVC(random_state=0,max_iter=100000,C=0.01)
 
   
    
    accuracies1 = []
    std_accuracies1= []
    pvals_permut_acc1 = []
    sensitivities1 = []
    specificities1= []
    std_sensitivities1= []
    std_specificities1= []
    pvals_permut_sen1 = []
    pvals_permut_spe1 = []
    
    
    accuracies2 = []
    std_accuracies2= []
    pvals_permut_acc2 = []
    sensitivities2 = []
    specificities2= []
    std_sensitivities2= []
    std_specificities2= []
    pvals_permut_sen2 = []
    pvals_permut_spe2 = []
    
    accuracies3 = []
    std_accuracies3= []
    pvals_permut_acc3 = []
    sensitivities3 = []
    specificities3= []
    std_sensitivities3= []
    std_specificities3= []
    pvals_permut_sen3 = []
    pvals_permut_spe3 = []
    
   ## pvalue between data1 and data2 
    pvals_acc1=[]
    tvals_acc1=[]
    pvals_sen1=[]
    tvals_sen1=[]
    pvals_spe1=[]
    tvals_spe1=[]
    
    ## pvalue between data1 and data3 
    pvals_acc2=[]
    tvals_acc2=[]
    pvals_sen2=[]
    tvals_sen2=[]
    pvals_spe2=[]
    tvals_spe2=[]

    for i in tqdm.tqdm(range(step_min,step_max,step)):
        #pipe_clf= make_pipeline(StandardScaler(),SelectKBest(k=i), PCA(n_components=i), svc)
        pipe_clf= make_pipeline(StandardScaler(),SelectKBest(k=i), svc)
        
        accuracy1 = cross_val_score(pipe_clf, data1, y=labels, cv=loo,scoring='accuracy')
        accuracies1.append(accuracy1.mean())
        std_accuracies1.append(accuracy1.std()/np.sqrt(data1.shape[0]))
        score1, permutation_scores1,pval1 = permutation_test_score(pipe_clf, data1, n_permutations=nperm,y=labels, cv=loo, scoring='accuracy')
        print("Classification score %s (pvalue : %s) for k=%s" % (score1, pval1,i))   
        # View histogram of permutation scores
        #plt.hist(permutation_scores1, 20, label='Permutation scores',edgecolor='black')
        #ylim = plt.ylim()
        #plt.plot(2 * [score1], ylim, '--g', linewidth=3,label='Classification Score' ' (pvalue %.4f)' % pval1)
        #plt.plot(2 * [1. / np.unique(labels).size], ylim, '--k', linewidth=3, label='Luck')
        #plt.ylim(ylim)
        #plt.legend()
        #plt.xlabel('Score')
        #plt.show()
        pvals_permut_acc1.append(pval1)
        
        sensitivity1=model_selection.cross_val_score(pipe_clf,data1, y=labels, cv=loo, scoring='recall')
        sensitivities1.append(sensitivity1.mean())
        std_sensitivities1.append(sensitivity1.std()/np.sqrt(data1.shape[0]))  
        _,_,pval11 = permutation_test_score(pipe_clf, data1, n_permutations=nperm,y=labels, cv=loo,scoring='recall')
        pvals_permut_sen1.append(pval11)
        
        specificity1=model_selection.cross_val_score(pipe_clf,data1, y=labels, cv=loo, scoring=make_scorer(specificity_score))
        specificities1.append(specificity1.mean())
        std_specificities1.append(specificity1.std()/np.sqrt(data1.shape[0]))
        _,_,pval111 = permutation_test_score(pipe_clf, data1, n_permutations=nperm,y=labels, cv=loo,scoring=make_scorer(specificity_score))
        pvals_permut_spe1.append(pval111)
     
        accuracy2 = cross_val_score(pipe_clf, data2, y=labels, cv=loo, scoring='accuracy')
        accuracies2.append(accuracy2.mean())
        std_accuracies2.append(accuracy2.std()/np.sqrt(data2.shape[0]))
        score2, permutation_scores2,pval2 = permutation_test_score(pipe_clf, data2, n_permutations=nperm,y=labels, cv=loo, scoring='accuracy')
        print("Classification score %s (pvalue : %s) for k=%s" % (score2, pval2,i))   
        # View histogram of permutation scores
        #plt.hist(permutation_scores2, 20, label='Permutation scores',edgecolor='black')
        #ylim = plt.ylim()
        #plt.plot(2 * [score2], ylim, '--g', linewidth=3,label='Classification Score' ' (pvalue %.4f)' % pval2)
        #plt.plot(2 * [1. / np.unique(labels).size], ylim, '--k', linewidth=3, label='Luck')
        #plt.ylim(ylim)
        #plt.legend()
        #plt.xlabel('Score')
        #plt.show()
        pvals_permut_acc2.append(pval2)
        
        sensitivity2=model_selection.cross_val_score(pipe_clf,data2, y=labels, cv=loo, scoring='recall')
        sensitivities2.append(sensitivity2.mean())
        std_sensitivities2.append(sensitivity2.std()/np.sqrt(data2.shape[0]))  
        _,_,pval22 = permutation_test_score(pipe_clf, data2, n_permutations=nperm,y=labels, cv=loo,scoring='recall')
        pvals_permut_sen2.append(pval22)
        
        specificity2=model_selection.cross_val_score(pipe_clf,data2, y=labels, cv=loo,scoring=make_scorer(specificity_score))
        specificities2.append(specificity2.mean())
        std_specificities2.append(specificity2.std()/np.sqrt(data2.shape[0]))
        _,_,pval222 = permutation_test_score(pipe_clf, data2, n_permutations=nperm,y=labels, cv=loo,scoring=make_scorer(specificity_score))
        pvals_permut_spe2.append(pval222)
        
        accuracy3 = cross_val_score(pipe_clf, data3, y=labels, cv=loo, scoring='accuracy')
        accuracies3.append(accuracy3.mean())
        std_accuracies3.append(accuracy3.std()/np.sqrt(data3.shape[0]))
        score3, permutation_scores3,pval3 = permutation_test_score(pipe_clf, data3, n_permutations=nperm,y=labels, cv=loo, scoring='accuracy')
        print("Classification score %s (pvalue : %s) for k=%s" % (score3, pval3,i))   
        # View histogram of permutation scores
        #plt.hist(permutation_scores3, 20, label='Permutation scores',edgecolor='black')
        #ylim = plt.ylim()
        #plt.plot(2 * [score3], ylim, '--g', linewidth=3,label='Classification Score' ' (pvalue %.4f)' % pval3)
        #plt.plot(2 * [1. / np.unique(labels).size], ylim, '--k', linewidth=3, label='Luck')
        #plt.ylim(ylim)
        #plt.legend()
        #plt.xlabel('Score')
        #plt.show()
        pvals_permut_acc3.append(pval3)
        
        sensitivity3=model_selection.cross_val_score(pipe_clf,data3, y=labels, cv=loo, scoring='recall')
        sensitivities3.append(sensitivity3.mean())
        std_sensitivities3.append(sensitivity3.std()/np.sqrt(data3.shape[0]))  
        _,_,pval33 = permutation_test_score(pipe_clf, data3, n_permutations=nperm,y=labels, cv=loo,scoring='recall')
        pvals_permut_sen3.append(pval33)
        
        specificity3=model_selection.cross_val_score(pipe_clf,data3, y=labels, cv=loo, scoring=make_scorer(specificity_score))
        specificities3.append(specificity3.mean())
        std_specificities3.append(specificity3.std()/np.sqrt(data3.shape[0]))
        _,_,pval333 = permutation_test_score(pipe_clf, data3, n_permutations=nperm,y=labels, cv=loo,scoring=make_scorer(specificity_score))
        pvals_permut_spe3.append(pval333)
        
        
        
        t0,p0 = ttest_ind(accuracies1,accuracies2)
        pvals_acc1.append(p0)
        tvals_acc1.append(t0)
        
        t1,p1 = ttest_ind(sensitivities1,sensitivities2)
        pvals_sen1.append(p1)
        tvals_sen1.append(t1)
        
        t2,p2 = ttest_ind(specificities1,specificities2)
        pvals_spe1.append(p2)
        tvals_spe1.append(t2)
        
        
        t00,p00 = ttest_ind(accuracies1,accuracies3)
        pvals_acc2.append(p00)
        tvals_acc2.append(t00)
        
        t11,p11 = ttest_ind(sensitivities1,sensitivities3)
        pvals_sen2.append(p11)
        tvals_sen2.append(t11)
        
        t22,p22 = ttest_ind(specificities1,specificities3)
        pvals_spe2.append(p22)
        tvals_spe2.append(t22)
    
    

    
    return (accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3, pvals_permut_spe3)





print('Classification: STD+SG, STD, FC')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,abide_ts_std,mat_connectivity,LABELS_ALL,10,360,10)


print('save classification metrics')


np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_STD_FC_0.01.npz",
                        labels=LABELS_ALL,
                        abide_ts_std=abide_ts_std,
                        signals_fourier_std=signals_fourier_std,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )

print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for STD approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for FC approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for STD approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for FC approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for STD approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for FC approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))


print('Classification: STD+SG, Mean, Mean+SG')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,abide_ts_mean,signals_fourier_mean,LABELS_ALL,10,360,10)

np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_Mean_Mean+SG_0.01.npz",
                        labels=LABELS_ALL,
                        abide_ts_mean=abide_ts_mean,
                        signals_fourier_std=signals_fourier_std,
                        signals_fourier_mean=signals_fourier_mean,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )
print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for Mean approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for Mean+SG approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for Mean approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for Mean+SG  approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for Mean approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for Mean+SG  approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))

print('Classification: STD+SG, EC, NS')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,eig_cens,Node_strengths,LABELS_ALL,10,360,10)

np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_EC_NS_0.01.npz",
                        labels=LABELS_ALL,
                        eig_cens=eig_cens,
                        Node_strengths=Node_strengths,
                        signals_fourier_std=signals_fourier_std,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )
print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for EC approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for NS approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for EC approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for NS  approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for EC approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for NS  approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))

print('Classification: STD+SG, CC, var')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,clusterings,abide_ts_var,LABELS_ALL,10,360,10)

np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_CC_var_0.01.npz",
                        labels=LABELS_ALL,
                        clusterings=clusterings,
                        abide_ts_var=abide_ts_var,
                        signals_fourier_std=signals_fourier_std,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )
print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for CC approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for var approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for CC approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for var  approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for CC approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for var  approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))

print('Classification:STD+SG, kurtosis, kurtosis+SG')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,abide_ts_kurtosis,signals_fourier_kurtosis,LABELS_ALL,10,360,10)


print('save classification metrics')



np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_kurtosis_kurtosis+SG_0.01.npz",
                        labels=LABELS_ALL,
                        abide_ts_std=abide_ts_std,
                        abide_ts_kurtosis=abide_ts_kurtosis,
                        signals_fourier_std=signals_fourier_std,
                        signals_fourier_kurtosis=signals_fourier_kurtosis,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )

print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for kurtosis approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for kurtosis+SG approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for kurtosis approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for kurtosis+SG approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for kurtosis approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for kurtosis+SG approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))

print('Classification:STD+SG, VAR, VAR+SG')


accuracies1,std_accuracies1,pvals_acc1,pvals_permut_acc1,sensitivities1,std_sensitivities1,pvals_sen1,pvals_permut_sen1,specificities1,std_specificities1,pvals_spe1,pvals_permut_spe1,accuracies2,std_accuracies2,pvals_acc2,pvals_permut_acc2,sensitivities2,std_sensitivities2,pvals_sen2,pvals_permut_sen2,specificities2,std_specificities2,pvals_spe2,pvals_permut_spe2, accuracies3,std_accuracies3,pvals_permut_acc3,sensitivities3,std_sensitivities3,pvals_permut_sen3,specificities3,std_specificities3,pvals_permut_spe3=classification_metrics_vis_v1(100,signals_fourier_std,abide_ts_var,signals_fourier_var,LABELS_ALL,10,360,10)


print('save classification metrics')



np.savez_compressed("all_site_"+"intra"+"filtering_GlobalSR_k10_QC_svc_std+SG_var_var+SG_0.01.npz",
                        labels=LABELS_ALL,
                        abide_ts_std=abide_ts_std,
                        abide_ts_var=abide_ts_var,
                        signals_fourier_std=signals_fourier_std,
                        signals_fourier_var=signals_fourier_var,
                        mat_connectivity=mat_connectivity,
                        accuracies1=accuracies1,
                        std_accuracies1=std_accuracies1,
                        pvals_acc1=pvals_acc1,
                        pvals_permut_acc1=pvals_permut_acc1,
                        sensitivities1=sensitivities1,
                        std_sensitivities1=std_sensitivities1,
                        pvals_sen1=pvals_sen1,
                        pvals_permut_sen1=pvals_permut_sen1,
                        specificities1=specificities1,
                        std_specificities1=std_specificities1,
                        pvals_spe1=pvals_spe1,
                        pvals_permut_spe1=pvals_permut_spe1,
                        accuracies2=accuracies2,
                        std_accuracies2=std_accuracies2,
                        pvals_acc2=pvals_acc2,
                        pvals_permut_acc2=pvals_permut_acc2,
                        sensitivities2=sensitivities2,
                        std_sensitivities2=std_sensitivities2,
                        pvals_sen2=pvals_sen2,
                        pvals_permut_sen2=pvals_permut_sen2,
                        specificities2=specificities2,
                        std_specificities2=std_specificities2,
                        pvals_spe2=pvals_spe2,
                        pvals_permut_spe2=pvals_permut_spe2,
                        accuracies3=accuracies3,
                        std_accuracies3=std_accuracies3,
                        pvals_permut_acc3=pvals_permut_acc3,
                        sensitivities3=sensitivities3,
                        std_sensitivities3=std_sensitivities3,
                        pvals_permut_sen3=pvals_permut_sen3,
                        specificities3=specificities3,
                        std_specificities3=std_specificities3,
                        pvals_permut_spe3=pvals_permut_spe3,
                        )

print('####################Accuracy##########################')

print('Maximal value of accuracy for STD+SG approach: %s +/- %s'% (max(accuracies1),std_accuracies1[np.argmax(accuracies1)]))
print('Maximal value of accuracy for var approach: %s +/- %s'% (max(accuracies2),std_accuracies2[np.argmax(accuracies2)]))
print('Maximal value of accuracy for var+SG approach: %s +/- %s'% (max(accuracies3),std_accuracies3[np.argmax(accuracies3)]))

print('####################Sensitivity##########################')

print('Maximal value of sensitivity for STD+SG approach: %s +/- %s'%(max(sensitivities1),std_sensitivities1[np.argmax(sensitivities1)]))
print('Maximal value of sensitivity for var approach:  %s +/- %s'% (max(sensitivities2),std_sensitivities2[np.argmax(sensitivities2)]))
print('Maximal value of sensitivity for var+SG approach:  %s +/- %s'% (max(sensitivities3),std_sensitivities3[np.argmax(sensitivities3)]))

print('####################Specificity##########################')

print('Maximal value of specificity for STD+SG approach: %s +/- %s'% (max(specificities1),std_specificities1[np.argmax(specificities1)]))
print('Maximal value of specificity for var approach: %s +/- %s'% (max(specificities2),std_specificities2[np.argmax(specificities2)]))
print('Maximal value of specificity for var+SG approach:%s +/- %s'%  (max(specificities3),std_specificities3[np.argmax(specificities3)]))





