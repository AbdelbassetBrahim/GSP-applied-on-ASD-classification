# GSP-applied-on-ASD-classification
Graph Signal Processing applied on Autism Spectrum Disorder classification

### Contributors

Abdelbasset Brahim- IMT Atlantique - France 
abdelbasset.brahim@imt-atlantique.fr

Nicolas Farrugia - IMT Atlantique - France 
nicolas.farrugia@imt-atlantique.fr 

September 2019
==============================

This repository hosts scripts necessary for running predictions on ABIDE database using Grapĥ Signal Processing.


Data we provide
---------------

Timeseries signals
~~~~~~~~~~~~~~~~~~~

we provide the timeseries signals which are extracted on all the datasets of this database using the Glasser atlas [1].

We provide the timeseries in a zip folder titled 'Time_serie_per_site'. Please run Script titled 'ABIDE_dataset_time_series.py'to know how to download it automatically.

Phenotypes 
---------------

In this repository, we also uploaded the phenotypic information in csv files for ABIDE  as it is accessible for downloading.

- ABIDE, a csv file called as "Phenotypic_V1_0b_preprocessed1.csv" is downloaded manually from https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv and uploaded here.

What software we used
----------------------
- The sotware used for timeseries extraction, a Python library Nilearn (http://nilearn.github.io/) [2]. 

- Installing Nilearn and its dependencies are essential. To install Nilearn and its dependencies (http://nilearn.github.io/introduction.html#installing-nilearn) 

- We also used Pandas (http://pandas.pydata.org/) in the scripts to read csv files.

- For plotting results: matplotlib (https://matplotlib.org/)

- For graph construction, GFT, etc., we use pygsp package v0.5.1.

Scripts we provide: 
-------------------------------------------------------------------------------

-The scripts ``classification_intrasite_lr.py`` and ``classification_intrasite_svc.py`` can be used for launching predictions on 866 subjects classifying individuals between autism 403 and healthy subjects 468 using logistic regression and support vector machines, respectively.

Notebooks we provide: 
-------------------------------------------------------------------------------
- The Jupyter notebooks ``Classification_vis_ABIDE_LR_UP.ipynb`` and ``Classification_vis_ABIDE_SVM_UP.ipynb`` can be used to generate the classification figures using two different classifiers.

-The Jupyter notebook ``GFT_visualisations_ABIDE_SVM.ipynb`` can be used for the visualization of features using several approaches, such as , CC, STD, STD+SG, Var, Var+SG.


References
^^^^^^^^^^
[1] M. F. Glasser, T. S. Coalson, E. C. Robinson, C. D. Hacker, J. Harwell, E. Yacoub, K. Ugurbil, J. Andersson, C. F. Beckmann, S. S. M. Jenkinson, M., D. C. Van Essen, A multi-modal parcellation of human cerebral cortex, Nature vol. 536,7615 (2016): 171-178. 536 (7615) (2016) 171–178.
[2] Abraham, A., et al. 2014. Machine learning for neuroimaging with scikit-learn. Frontiers in neuroinformatics 8.
