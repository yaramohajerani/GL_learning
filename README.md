
# Automatic Delineation of Glacier Grounding Lines in Differential Interferometric Synthetic-Aperture Radar Data using Deep Learning

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/yaramohajerani/GL_learning/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/237542709.svg)](https://zenodo.org/badge/latestdoi/237542709)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yaramohajerani/GL_learning/graphs/commit-activity)
[![Language](https://img.shields.io/badge/python-v3.7-green.svg)](https://www.python.org/)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](mailto:ymohajer@uw.edu)

This repository contains the pipeline used for the automatic delineation of glacier grounding lines from Differential InSAR interferograms, as described in

> Mohajerani, Y., Jeong, S., Scheuchl, B. et al. Automatic delineation of glacier grounding lines in differential interferometric synthetic-aperture radar data using deep learning. Sci Rep 11, 4992 (2021). https://doi.org/10.1038/s41598-021-84309-3

The step-by-step procedure is described below.

For questions, contact **Yara Mohajerani** at [ymohajer@uci.edu](mailto:ymohajer@uci.edu).

For a list of Python dependencies or creating a Conda environment, please refer to  `gl_env.yml`.  In addition, you also need to install `label_centerlines` from [this repo](https://github.com/ungarj/label_centerlines).

---

### 1. Pre-process geocoded data (geotiff files) into numpy arrays to be used in the training of the neural network:

`python geocoded_preprocess.py --DIR=<data directory> --N_TEST=<# of tiles set aside for testing>`

In addition, there are extra optional flags `AUGMENT` which provides additional augmentation by flipping each tile horizontally, verticall, and both horizontally and vertically, as well as `DILATE` which increases the width of the labelled grounding lines for traning. These options are set to `False` by detault. But by just listing them (i.e. `--AUGMENT` and `DILATE`) you can activate these additional operations.

### 2. Training (A) or Testing (B) neural network

**A.** Train the neural network:
This step is done in a Jupyter Notebook so that it can be run on Google Colab with GPU access for faster computation. The full notebook with executable cells and documentation is `GL_delineation_geocoded_complex_customLoss.ipynb`.

**or**

**B.** Run pre-trained neural network on any data directory:

`python run_prediction.py --DIR=< > --MODEL_DIR=< >`.
Where `DIR` is the data directory,`MODEL_DIR` is the code directory where the model is located (the model code and `.h5` file are assumed to be in the same location). If any other run configurations are different from the default settings specified in the code, they can also be changed as inline commandline arguments as above.

Note that if you want to run massive amounts of data in parallel on a computing node with Slurm commands, you can run

`python make_slurm.py --DIR=< > --MODEL_DIR=< > --SLURM_DIR=< > --USER=< >`
Where `SLURM_DIR` is the directory path for the slurm jobs to be made by `make_slurm` which can be run by the user using the master "job_list" file. `USER` is the username used for the Slurm job.


In addition, if you want to run the network specifically on the train/test data, you can just use the `test_model.py` script. 

`python test_model.py`

Again with the run configurations specified as inline commandline arguments.

### 3. Post-Processing: Stitching tiles together
Here we stitch the 512x512 tiles back together before postprocessing (although the code is generic for any tile dimension as long as they're all the same):

`python stitch_tile.py --DIR=<subdirectory with outputs> --KERNEL=<guassian width>`

The Guassian kernel refers to the variance of the Gaussian averaging kernel, givn by `kernel_weight = np.exp(-(gxx**2+gyy**2)/sigma_kernel)` where `gxx` and `gyy` are the grids.

If you want to use a uniform averaging kernel instead of Gaussian kernel to average the overlapping tiles, `--noFLAG` to the command line arguments. However, it is preferable to use Gaussian averaging so that the center of each tile counts more than the edges, in order to avoid edge effects.

### 4. Post-Processing: Vectorizing Results and Converting to Shapefiles
Use the combined tiles from the previous step to convert the raster output of the neural network to vectorized LineStrings and save as Shapefiles.

`python polygonize.py --DIR=<subdirectory with outputs> --FILTER=<minimum line threshold in meters> --OUT_BASE=<base directory for slurm outputs> --IN_BASE=<base directory for input (parent directory of "--DIR" --noMASK`

The `FILTER` input refers to the minimum threshold used to clean up the output. Every line segment shorter than this threshold is disregarded. In addition, note that you can use the `--noMASK` in commandline arguments to not output training vs test masks. If not specified, masks will also be outputted. This is only useful if the input scenes are a combinatino of training and testing tiles (which is the case for the original training and testing data on the Getz Ice Shelf).

### 5. Post-Processing: Drawing centerlines
Use the slurm files produced by `polygonize.py` in the previous step to processes centerlines in parallel. Note this is currently designed for the partitions on UC Irvine's [GreenPlanet](https://ps.uci.edu/greenplanet/Partitions-new-names), but can be adjusted accordingly for other users. This is done by calling the `run_centerline.py` script.

`python run_centerline.py <input_file1> <input_file2>`

Where you can list an unlimited number of input files to be run in serial. But the Slurm script runs these in parallel to significantly cut down on processing time.

Note that in order to run the files in parallel with Slurm, all the user has to do is the run the master job list, e.g.:

`sh total_job_list_8.0km.sh` 


Note that currently `polygonize.py` is set up such that the centerline is done in parallel for each line segment of each DInSAR scene. If you want to combine the slurm jobs such that each scene is run as one job, run the following script:

`python combine_centerline_slurms.py <input_file_list>`

where the input is the list of all files produced by `polygonize.py` (e.g. `total_job_list_8.0km.sh`). Make sure you provide the full path. This program will output a similar output with the suffix `combined` (e.g. `total_job_list_8.0km_combined.sh`) which runs each scene as a single job to reduce the total number of submitted jobs. Note that the output of running 

`sh total_job_list_8.0km_combined.sh`

is still individual shapefiles for each line segment. Note that only the lines that are not classified as noise are run through the centerline routine.

### 6. Combining centerlines
To combine the individual centerlines produced in the previous step, run

`python combine_shapefiles.py --DIR=<complete path to shapefile directory> --FILTER=<minimum line threshold in meters in exisiting files>`

Note that `--DIR` is the full path to the directory where the shapefiles to be combined are. `--FILTER` is the same filter size previously (in meters) that is the suffix of the files to be combined.

### 7. Error Analysis
Now we can use the vectorized output from the previous step to assess the uncertainty. 

`python mean_difference.py --DIR=<subdirectory with outputs> --FILTER=<minimum line threshold in meters>`

The arguments are the same as the vectorization step above. In addition, since the previous step categorizes and labels each geometric object, we can do the error analysis with or without pinning points. The default settings do NOT include pinning points. If you want to include them, add the `--PINNING` flag to the commandline arguments.

## Grounding Zone Analysis
In order to calculate the grounding zone purely from the velocity direction, use

`python calc_gz.py --GL_FILE=<GL file path> --BASIN_FILE=<Shapefile with all the basins> --VEL_FILE=<velocity file> --REGION=<name of region to analyze (default Getz)>`

You can also use the `--DIST` option to specify the length of the transects to be drawn across the grounding zone (default 10km) and `--NUM` for the number of transects to be drawn (default 500).

Note that in addition to the velocity-based transects, the program also outputs the estimates from a geometric approach to finding the perpendicular transect to the grounding zone. 

Also note that the program is currently set up to choose the shortest transect at each point among 20 purturblation to make sure the shortest path is taken. However, this could have the side effect of finding the tangent (instead of the perpendicular transect) at some points.

A better option is to use a hybrid approach where velocity-based transects are used in areas of fast flow, and a pure geometric centerline-based approach is used in areas of slow flow. 

The centerline-based approach is currently done manually in QGIS using the [Geometric Attributes](https://plugins.qgis.org/plugins/geometric_attributes/) plugin, and the widths transects are saved to a Shapefile. 

Then, the following script can be used to retrieve the hybrid width estimates:

`python calc_gz_hybrid.py --WIDTH_FILE=<widths from QGIS> --POINT_FILE=<shapefile for any prescribed coordinates to get GZ (optional)> --THRESHOLD=<threshold for geometric vs velocity based transects (m/yr)>`
The rest of the commandline input arguments are the same as `calc_gz.py` above.

Also note this script does NOT use the ensemble approach for finding the shortest path (used in `calc_gz.py`), which achieves better results.

Furthermore, you can calculate the GZ widths from hydrostatic equilibrium at the same coordinates, and compare with the delineated GZ widths:

`python hydrostatic_gz.py`

`python calc_gz_stats.py`

Note that the input file paths are specified inside the scripts if they need to be changed. 

