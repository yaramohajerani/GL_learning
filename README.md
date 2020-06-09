# GL_learning
Automated Grounding Line Delineation

## By Yara Mohajerani

This repository contains the pipeline used for the automatic delineation of glacier grounding lines from interferograms.

The step-by-step procedure is described below.

### 1. Pre-process geocoded data (geotiff files) into numpy arrays to be used in the training of the neural network:

`python geocoded_preprocess.py --DIR=<data directory> --N_TEST=<# of tiles set aside for testing>`

In addition, there are extra optional flags `AUGMENT` which provides additional augmentation by flipping each tile horizontally, verticall, and both horizontally and vertically, as well as `DILATE` which increases the width of the labelled grounding lines for traning. These options are set to `False` by detault. But by just listing them (i.e. `--AUGMENT` and `DILATE`) you can activate these additional operations.

### 2. Train the neural network:
