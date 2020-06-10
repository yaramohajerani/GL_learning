
# Automated Delineation of Grounding Lines with a Convolutional Neural Network
## By Yara Mohajerani
---

This repository contains the pipeline used for the automatic delineation of glacier grounding lines from interferograms.

The step-by-step procedure is described below.

### 1. Pre-process geocoded data (geotiff files) into numpy arrays to be used in the training of the neural network:

`python geocoded_preprocess.py --DIR=<data directory> --N_TEST=<# of tiles set aside for testing>`

In addition, there are extra optional flags `AUGMENT` which provides additional augmentation by flipping each tile horizontally, verticall, and both horizontally and vertically, as well as `DILATE` which increases the width of the labelled grounding lines for traning. These options are set to `False` by detault. But by just listing them (i.e. `--AUGMENT` and `DILATE`) you can activate these additional operations.

### 2A. Train the neural network:
This step is done in a Jupyter Notebook so that it can be run on Google Colab with GPU access for faster computation. The full notebook with executable cells and documentation is `GL_delineation_geocoded_complex_customLoss.ipynb`.
**or**
### 2B. Run pre-trained neural network:
If you want to start from a pre-trained network, you can just use the `test_model.py` script. 

`python test_model.py`
For now you can change the setting inside the script. This script produces output for both the train and test data. 

### 3. Optional Post-Processing Notebook for exploration and outputting multi-plot PNGs of the pipeline
This is again in a Jupyter Notebook. Note that the next step does not depend on the output of this step so it is optional. The notebook with executable cells and documentation is `GL_postprocessing_geocoded_complex.ipynb`.

### 4. Post-Processing: Stitching tiles together
Here we stitch the 512x512 tiles back together before postprocessing.

`python stitch_tile.py --DIR=<subdirectory with outputs> --NX=<width (512)> --NY=<height (512)> --KERNEL=<guassian width>`

The Guassian kernel refers to the variance of the Gaussian averaging kernel, givn by `kernel_weight = np.exp(-(gxx**2+gyy**2)/sigma_kernel)` where `gxx` and `gyy` are the grids.

If you want to use a uniform averaging kernel instead of Gaussian kernel to average the overlapping tiles, `--noFLAG` to the command line arguments. However, it is preferable to use Gaussian averaging so that the center of each tile counts more than the edges, in order to avoid edge effects.

### 5. Post-Processing: Vectorizing Results and Converting to Shapefiles
Use the combined tiles from the previous step to convert the raster output of the neural network to vectorized LineStrings and save as Shapefiles.
`python convert_shapefile.py --DIR=<subdirectory with outputs> --FILTER=<minimum line threshold in meters>`
The `FILTER` input refers to the minimum threshold used to clean up the output. Every line segment shorter than this threshold is disregarded. In addition, note that you can use the `--CLOBBER` in commandline arguments to overwrite existing files in the directory. The default is to not overwrite existing files.

### 6. Error Analysis
Now we can use the vectorized output from the previous step to assess the uncertainty. 

`python mean_difference.py --DIR=<subdirectory with outputs> --FILTER=<minimum line threshold in meters>`
The arguments are the same as the vectorization step above. In addition, since the previous step categorizes and labels each geometric object, we can do the error analysis with or without pinning points. The default settings do NOT include pinning points. If you want to include them, add the `--PINNING` flag to the commandline arguments.
