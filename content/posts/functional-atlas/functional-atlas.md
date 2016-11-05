Title: Functional brain atlas with Nilearn
Date: 2016-04-02 14:54
Category: Neuroscience
Tags: functional atlas, fmri, resting state
Slug: functional-atlas
Author: João Loula
publications_src: content/posts/functional-atlas/references.bib
Summary: This post is based on the Nilearn tutorial given by myself and Alex Abraham at the 2016 Brainhack Vienna: in it, we'll give a brief introduction to Nilearn and its functionalities, and we'll present a usecase of extracting a functional brain atlas from the ABIDE resting state dataset.

This post is based on the Nilearn tutorial given by myself and Alex Abraham at the 2016 Brainhack Vienna: in it, we'll give a brief introduction to Nilearn and its functionalities, and we'll present a usecase of extracting a functional brain atlas from the ABIDE resting state dataset.

## Nilearn

Nilearn is a python module for statistical and machine learning analysis on brain data: it leverages python's simplicity and versatility into an easy-to-use integrated pipeline. Having analysis run on single, simple scripts allows for better reproducibility than, say, clicking on things in a GUI.

This is how a typical Nilearn analysis goes:

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/functional-atlas/DMN.jpg" alt='missing' align='middle' />
	<figcaption> <sup>source: [@@nilearn-poster]</sup>
</figure>

One of the main objects in the module is the Masker: it allows for easy conversion from a 4D brain scan time-series to a numpy array that's ready to be treated by scikit-learn algorithms and vice-versa. Accompanying it are a wide range of image processing functions, allowing for flexible data manipulation.

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/functional-atlas/masking.jpg" alt='missing' align='middle' />
	<figcaption> <sup>source: [@@nilearn-poster]</sup>
</figure>

Next, we'll take a look at a use case to see how the module works in action, on the ABIDE autism resting-state data [@@abide].

## Brain atlas

Before analyzing functional connectivity, we need to reduce the dimensionality of the problem. To do that, we estimate an atlas directly on our data. We'll start by importing some classic libraries:

	:::python
	# This line allows plotting directly in the notebook
	%matplotlib inline

	# Python scientific package
	import numpy as np

### Loading the data

Nilearn provides a bunch of automatic downloaders to ease reproducibility of the analysis. With nilearn, an analysis is run in a single script and can be shared easily. The nilearn fetchers can be found in the module nilearn.datasets.

```python
from nilearn.datasets import fetch_abide_pcp


# We specify the site and number of subjects we want to download
abide = fetch_abide_pcp(derivatives=['func_preproc'], 
                        SITE_ID=['NYU'], 
                        n_subjects=3)

# We look at the available data in this dataset
print(abide.keys())
```

We can print a description of the dataset:

```python
print(abide.description)
```

Retrieving the functional dataset is also straightforward:

```python
# To get the functional dataset, we have to retrieve the variable 'func_preproc'
func = abide.func_preproc

# We can also look at where the data is loaded
print(func[1])
```

### Computing a brain atlas

Several reference atlases are available in nilearn. We also provide functions to compute a brain atlas directly from the data. In this example, we'll do this using a group ICA implementation called Canonical ICA.

```python
from nilearn import decomposition

# CanICA is nilearn's approach of group ICA. It directly embeds a masker.
canica = decomposition.CanICA(n_components=20, mask_strategy='background')
canica.fit(func)
```


```python
# Retrieve the components
components = canica.components_

# Use CanICA's masker to project the components back into 3D space
components_img = canica.masker_.inverse_transform(components)

# We visualize the generated atlas
from nilearn import plotting, image

plotting.plot_stat_map(image.index_img(components_img, 9), title='DMN')
plotting.show()
```

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/functional-atlas/DMN.png"/>
</p>


### Extracting subject specific timeseries signals from brain parcellations

Computing mask from the data, filtering, extracting data from the in-mask voxels can be processed easily by using nilearn classes such as NiftiMasker, NiftiMapsMasker, NiftiLabelsMasker which can be imported from nilearn.input_data module.
The advantage of using such tools from this module is that we can restrict our analysis to mask specific voxels timeseries data. For instance, class NiftiMasker can be used to compute mask over the data and apply preprocessing steps such as filtering, smoothing, standardizing and detrending on voxels timeseries signals. This type of processing is very much necessary, particularly during resting state fMRI data analysis. Additional to NiftiMasker, classes NiftiMapsMasker and NiftiLabelsMasker, can be used to extract subject specific timeseries signals on each subject data provided with the atlas maps (3D or 4D) comprising of specific brain regions. NiftiMapsMasker operated on 4D atlas maps, can be used to extract signals from each 4th dimensional map using least squares regression. Whereas, NiftiLabelsMasker operated on 3D maps denoted as labels image, can be used to extract averaged timeseries from group of voxels that correponds to each label in the image.

```python
# Import and initialize `NiftiMapsMasker` object and call `fit_transform` to
# extract timeseries signals from computed atlas.
from nilearn.input_data import NiftiMapsMasker

# The parameters used are maps_img as parcellations, resampling to maps image,
# smoothing of 6mm, detrending, standardizing and filtering (TR in sec). These later
# parameters are applied automatically when extracting timeseries data.
masker = NiftiMapsMasker(components_img, smoothing_fwhm=6, 
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1, 
                         high_pass=0.01)
```

### Extracting time series for each subject

```python
# We loop over the subjects to extract the time series
subjects_timeseries = []
for subject_func in func:
    subjects_timeseries.append(masker.fit_transform(subject_func))
```


```python
# Visualizing extracted timeseries signals. We import matplotlib.pyplot
import matplotlib.pyplot as plt


# We loop over the subjects to extract the time series
# We show them for a single subject
timeseries = subjects_timeseries[0]
print(timeseries.shape) # (number of scans/time points, number of brain regions/parcellations)
plt.plot(timeseries)
plt.title('Timeseries for single subject shown for 20 brain regions')
plt.xlabel('Number of regions')
plt.ylabel('Normalized signal')
plt.show()
```

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/functional-atlas/time-series.png"/>
</p>

### Extracting regions from computed atlas

ICA requires post-preprocessing. Here we use the RegionExtractor that thresholds the maps and extract brain regions.

```python
from nilearn.regions import RegionExtractor


extractor = RegionExtractor(components_img, threshold=2.,
                            thresholding_strategy=
                            'ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, 
                            min_region_size=1350)

# Just call fit() to process for regions extraction
extractor.fit()

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_

# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img, 
                         view_type='filled_contours',
                         title=title, threshold=0.008)
```


### Connectomes Estimation

Connectivity is typically estimated using correlation between time series. Recent studies has shown that partial correlation could give better results. Different estimators can also be used to apply some regularization on the matrix coefficients. Nilearn's ConnectivityMeasure object (in the nilearn.connectome module) provides three types of connectivity matrix: correlation, partial_correlation, and tangent (a method developped in our laboratory). ConnectivityMeasure can also use any covariance estimator shipped by scikit-learn (ShrunkCovariance, GraphLasso). In a first time, we estimate the connectivity using default parameters. We check that we have one matrix per subject.

from nilearn.connectome import ConnectivityMeasure


```python
conn_est = ConnectivityMeasure(kind='partial correlation')
conn_matrices = conn_est.fit_transform(abide.rois_cc200)
```


### Plotting connectivity matrix

We visualize the connectivity matrix of the first subject. This code is directly taken from a nilearn example.

```python
plt.imshow(conn_matrices[0], vmax=.20, vmin=-.20, cmap='RdBu_r')
plt.colorbar()
plt.title('Connectivity matrix of subject 0')
```

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/functional-atlas/connectivity-matrix.png"/>
</p>


### Extracting useful coefficients

Connecitivity matrices are symmetric. As such, half of the coefficients are redundant. They can even impact the results of some predictors. In order to "extract" these coefficients, we want to use a mask. numpy.tril function can help us with this task. However, using masking is hazardous without a good knowledge of numpy. Fortunately, nilearn provides a function to do this automatically and efficiently: nilearn.connectome.sym_to_vec.

```python
from nilearn.connectome import sym_to_vec


X = sym_to_vec(conn_matrices)
X.shape
```
### Setting up cross-validation

Getting reliable prediction results require to predict on unseen data. Cross-validation consists in leaving out a part of the dataset (testing set) to validate the model learnt on the remaining of the dataset (training set). Scikit-learn has all the utils necessary to do automatic cross-validation. In the case of ABIDE, we have a very heterogenous dataset and we want the sets to be balanced in term of acquisition sites and condition. We use a stratified cross-validation method for that.

```python
from sklearn.cross_validation import StratifiedShuffleSplit


ids = []
for site_id, dx in abide.phenotypic[['SITE_ID', 'DX_GROUP']]:
    ids.append(str(site_id) + str(dx))
cv = StratifiedShuffleSplit(ids, n_iter=10, test_size=.2)
```

### Prediction using Support Vector Classifier

Now that we have shown how to estimate a connectome and extract the interesting coefficients, we will see how to use them to diagnose ASD vs healthy individuals. For that purpose, we use a Support Vector Machine. This is one of the most simple classifiers. We use the default parameters in a first time and look at classification scores.


```python
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
import numpy as np


# DX_GROUP are the labels of the ABIDE dataset. 1=ASD, 2=Healthy
y = abide.phenotypic['DX_GROUP']
predictor = LinearSVC(C=0.01)
np.mean(cross_val_score(predictor, X, y, cv=cv))
```

### Exploring other methods and parameters

So far, we built a basic prediction procedure without tuning the parameters. Now we use for loops to explore several options. Note that the imbrication of the steps allow us to re-use connectivity matrix computed in the first loop for the different predictors. The same result can be achieved using nilearn's caching capacities.

```python
from sklearn.linear_model import RidgeClassifier


measures = ['correlation', 'partial correlation', 'tangent']
predictors = [
    ('svc_l2', LinearSVC(C=1)),
    ('svc_l1', LinearSVC(C=1, penalty='l1', dual=False)),
    ('ridge_classifier', RidgeClassifier()),
]

for measure in measures:
    conn_est = ConnectivityMeasure(kind=measure)
    conn_matrices = conn_est.fit_transform(abide.rois_cc200)
    X = sym_to_vec(conn_matrices)
    for name, predictor in predictors:
        print(measure, name, np.mean(cross_val_score(predictor, X, y, cv=cv)))
```

This should show the Ridge Classifier and the SVM classifier with L1 penalty as the highest scoring options.
