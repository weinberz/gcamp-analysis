# GCaMP Analysis

This code enables segmentation of images of PC12 cells expressing the calcium sensor GCaMP6f and enables extracting information from them.

A rough walkthrough is provided in:
- `train_gcamp.ipynb`: Generating a model for segmenting labeled PC12 cells from microscopy images. Based on [StarDist](https://github.com/mpicbg-csbd/stardist)
- `gcamp_predict.ipynb`: Use the above trained model predict segmentation for images. This was fine-tuned for eventualy analysis.
- `gcamp_analysis.ipynb`: This combines the above two notebooks with filtering cells based on responsivity to final KCl stimulation into a functional analysis for calcium signaling.

Use `gcamp_analysis.py` to run analysis on new data.
