Title: Brain surface plotting with Nilearn
Date: 2016-04-02 14:54
Category: Neuroscience
Tags: surface plot, atlas, statistical map
Slug: surface-plotting
Author: João Loula
publications_src: content/posts/surf-plotting/references.bib
Summary: In this post we'll explore Nilearn's recently acquired surface plotting capabilities through an example using seed-based resting state connectivity analysis.

We start by importing the libraries we're gonna need:

```python
from nilearn import plotting
from nilearn import datasets
from scipy import stats
import nibabel as nb
import numpy as np
%matplotlib inline
```

In this example we'll use the Rockland NKI enhanced resting state dataset [@@nki], a dataset containing 100 subjects with ages ranging from 6 to 85 years that aims at characterizing brain development, maturation and aging.

```python
# Retrieve the data
nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)

# NKI resting state data set of one subject left hemisphere in fsaverage5 space
resting_state = nki_dataset['func_left'][0]
```

We'll want to define regions of interest for our analysis: for this, we'll need a brain parcellation. For this purpose, we'll use the sulcal-depth based Destrieux cortical atlas [@@destrieux]: 

```python
# Destrieux parcellation left hemisphere in fsaverage5 space
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = nb.freesurfer.read_annot(destrieux_atlas['annot_left'])
```

The Destrieux parcellation is based on the Fsaverage5 [@@fsaverage5] surface data, so we'll go ahead and fetch that as well so as to be able to plot our atlas.

```python
# Retrieve fsaverage data
fsaverage = datasets.fetch_surf_fsaverage5()

# Fsaverage5 left hemisphere surface mesh files
fsaverage5_pial = fsaverage['pial_left'][0]
fsaverage5_inflated = fsaverage['infl_left'][0]
sulcal_depth_map = fsaverage['sulc_left'][0]
```

Last steps needed for our analysis: we'll pick a region as seed (we'll choose the dorsal posterior cingulate gyrus) and extract the time-series correspondent to it. Next, we want to calculate statistical correlation between the seed time-series and time-series of other cortical regions. For our measure of correlation, we'll use the Pearson product-moment correlation coefficient, given by $ \rho_{X, Y} \frac{\text{cov}\left( X, Y \right)}{ \sigma_X \sigma_Y } $.

```python
# Load resting state time series and parcellation
timeseries = plotting.surf_plotting.check_surf_data(resting_state)

# Extract seed region: dorsal posterior cingulate gyrus
region = 'G_cingul-Post-dorsal'
labels = np.where(parcellation[0] == parcellation[2].index(region))[0]

# Extract time series from seed region
seed_timeseries = np.mean(timeseries[labels], axis=0)

# Calculate Pearson product-moment correlation coefficient between seed
# time series and timeseries of all cortical nodes of the hemisphere
stat_map = np.zeros(timeseries.shape[0])
for i in range(timeseries.shape[0]):
    stat_map[i] = stats.pearsonr(seed_timeseries, timeseries[i])[0]

# Re-mask previously masked nodes (medial wall)
stat_map[np.where(np.mean(timeseries, axis=1) == 0)] = 0
```

Now for the actual plotting: we start by plotting the seed:

```python
# Display ROI on surface
plotting.plot_surf_roi(fsaverage5_pial, roi_map=labels, hemi='left',
                       view='medial', bg_map=sulcal_depth_map, bg_on_data=True)
plotting.show()
```

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/surf-plotting/roi.png"/>
</p>

Next, we'll plot the correlation statistical map in both the lateral and medial views:

```python
# Display unthresholded stat map in lateral and medial view
# dimmed background
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            darkness=.5)
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            view='medial', bg_map=sulcal_depth_map,
                            bg_on_data=True, darkness=.5)
plotting.show()
```
<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/surf-plotting/lateral.png"/>
</p>

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/surf-plotting/medial.png"/>
</p>

Finally, to show off Nilearn's plotting capabilities, we'll play a little with colormaps and transparency:

```python
# Displaying a thresholded stat map with a different colormap and transparency
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            cmap='Spectral', threshold=.6, alpha=.5)

plotting.show()
```

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/surf-plotting/alpha.png"/>
</p>
