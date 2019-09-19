
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

from pathlib import Path
from scipy import io as sio
from pygsp import graphs
import shutil


# In[2]:


from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_icbm152_2009

mnitemp = fetch_icbm152_2009()

glassermasker = NiftiLabelsMasker(labels_img='Glasser_masker.nii.gz',mask_img=mnitemp['mask'])
glassermasker.fit()


# In[3]:


import numpy as np 
coords= np.load('coords_with_order.npz')['coordinates']

ind_reord= np.load('coords_with_order.npz')['order']


# In[4]:


from nilearn.datasets import fetch_abide_pcp
import pandas
import os
df = pandas.read_csv('Phenotypic_V1_0b_preprocessed1.csv', sep='\t')
site_id=np.unique(df['SITE_ID'])
print(site_id)

for S in (site_id): 
    
    data_dir='test'
    #DX_GROUP: 1 is autism, 2 is control
    abidedata_func_normal=fetch_abide_pcp(data_dir=data_dir, n_subjects= None, pipeline='cpac', band_pass_filtering=True, global_signal_regression=True, derivatives=['func_preproc'], quality_checked=True,DX_GROUP=2,SITE_ID=S)
    abidedata_func_autism=fetch_abide_pcp(data_dir=data_dir, n_subjects= None, pipeline='cpac', band_pass_filtering=True, global_signal_regression=True, derivatives=['func_preproc'], quality_checked=True,DX_GROUP=1,SITE_ID=S)

    

    print(len(abidedata_func_normal['func_preproc']))
    print(len(abidedata_func_autism['func_preproc']))

    ## Extaction of time series for normal
    abide_ts=[]
    abide_ts_normal=[]
    for i in range(len(abidedata_func_normal['func_preproc'])):
        ts_nor = glassermasker.transform(imgs=abidedata_func_normal['func_preproc'][i])
        ts_normal = ts_nor[:,ind_reord]
        abide_ts_normal.append(ts_normal)
        abide_ts.append(ts_normal)
   

    ## Extaction of time series for autism
   
    abide_ts_autism=[]
    for i in range(len(abidedata_func_autism['func_preproc'])):
        ts_aut = glassermasker.transform(imgs=abidedata_func_autism['func_preproc'][i])
        ts_autism = ts_aut[:,ind_reord]
        abide_ts_autism.append(ts_autism)
        abide_ts.append(ts_autism)
        
    print(len(abide_ts))
    
    labels=np.concatenate((np.zeros(len(abide_ts_normal)),np.ones(len(abide_ts_autism))))
    print(labels.shape)

   
    np.savez_compressed("site_"+str(S)+"time_series.npz",
                        labels=labels,
                        abide_ts_normal=abide_ts_normal,
                        abide_ts_autism=abide_ts_autism,
                        abide_ts=abide_ts
                        )
    shutil.rmtree('test/ABIDE_pcp')




