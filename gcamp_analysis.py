"""
gcamp_analysis

Analyzes 40X confocal images of whole-cell biosensors.

Parameters:
 (--indir): dir containing images or multiple subdirs with images to analyze

General workflow:
1. Find all subfolders with tiffs
2. For each tiff:
    1. Segment images
    2. Collect fluorescence data in 1 or 2 channels for each segmented cell
3. For each folder, summarize data into max responses
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D

from skimage import measure
from skimage import draw
from centroidtracker import CentroidTracker
from csbdeep.io import save_tiff_imagej_compatible

import os
from pathlib import Path

import argparse

np.random.seed(6)

# Create arguments. 'indir' is a positional argument
parser = argparse.ArgumentParser()
parser.add_argument("indir")
parser.add_argument("--summarize_only", action="store_true")
args = parser.parse_args()

def summarize_folder(folder, min_frames = 240, min_gcamp = 130):
    """Collects analyzed dataframes within a folder and returns
    a simple summary of maximum responses across all cells given
    a set of filtering criteria"""

    csvs = []
    for file in os.listdir(folder):
        if 'csv' in file:
            csvs.append(file)

    data = {}

    for csv in csvs:
        csv = Path(csv)
        dat = pd.read_csv(folder / csv, index_col=0)
        name = (csv.stem
                .replace('analysis','')
                .replace('Gcamp6f','')
                .replace('PC12','')
                .replace('_',' ')
                .replace('  ',' ')
                .strip())
        date = str(csv.parent.name).split(" ")[0]
        dat['date'] = date
        data[name] = dat

    data = pd.concat(data, names=['condition']).reset_index(level=0).reset_index(drop=True)

    cells_in_conditions = np.unique(data[['condition','cell']].to_records(index=False))

    for condition, cell in cells_in_conditions:
        num_frames = len(data.loc[(data['condition'] == condition) &
                              (data['cell'] == cell), 'frame'])
        mean_gcamp = data.loc[(data['condition'] == condition) &
                              (data['cell'] == cell), 'primary_mean'].mean()
        if (num_frames <= min_frames) or (mean_gcamp < min_gcamp):
            data = data.drop(data[(data['condition'] == condition) &
                                  (data['cell'] == cell)].index)

    for normalization in ['primary_mean','primary_intden','secondary_mean', 'secondary_intden']:
        data[normalization + "_normalized"] = 0
        for condition, cell in cells_in_conditions:
            baseline = data.loc[(data['condition'] == condition) &
                              (data['cell'] == cell) &
                              (data['frame'] < 100), normalization]
            baseline = np.mean(baseline)
            normalized = data.loc[(data['condition'] == condition) &
                                  (data['cell'] == cell), normalization].values / baseline
            data.loc[(data['condition'] == condition) &
                     (data['cell'] == cell), normalization + "_normalized"] = normalized

    output = {}

    output['intden_by_time'] = data.pivot_table(values='primary_intden_normalized', index='frame', columns=['condition','cell'])
    output['intden_by_time'].columns = output['intden_by_time'].columns.droplevel(1)

    output['mean_by_time'] = data.pivot_table(values='primary_mean_normalized', index='frame', columns=['condition','cell'])
    output['mean_by_time'].columns = output['mean_by_time'].columns.droplevel(1)

    output['max_drug_mean'] = (data[(data['frame'] >100) & (data['frame'] < 200)]
     .pivot_table(values='primary_mean_normalized', index='cell', columns='condition', aggfunc=max))
    output['max_drug_intden'] = (data[(data['frame'] >100) & (data['frame'] < 200)]
     .pivot_table(values='primary_intden_normalized', index='cell', columns='condition', aggfunc=max))

    output['max_intden_kcl_spike'] = (data[(data['frame'] >300) & (data['frame'] < 400)]
     .pivot_table(values='primary_intden_normalized', index='cell', columns='condition', aggfunc=max))
    output['max_mean_kcl_spike']= (data[(data['frame'] >300) & (data['frame'] < 400)]
     .pivot_table(values='primary_mean_normalized', index='cell', columns='condition', aggfunc=max))

    output['max_intden_kcl_plateau'] = (data[(data['frame'] >430)]
     .pivot_table(values='primary_intden_normalized', index='cell', columns='condition', aggfunc=max))
    output['max_mean_kcl_plateau'] = (data[(data['frame'] >430)]
     .pivot_table(values='primary_mean_normalized', index='cell', columns='condition', aggfunc=max))

    return output

def get_movie_dims(movie):
    """Handler for checking tiff dimensions"""
    movie_dims = len(movie.shape)
    if movie_dims <= 2:
        raise TypeError("Movie appears to not have time data - is it multiple frames?")
    elif movie_dims == 3:
        return (movie.shape[0], 1, movie.shape[1], movie.shape[2])
    elif movie_dims == 4:
        return tuple(movie.shape)
    else:
        raise NotImplementedError("Don't know what to do with a movie with this many dimensions ¯\_(ツ)_/¯")

def analyze_gcamp(movie, model, num_frames, num_ch, dim_y, dim_x, threshold=0.4):
    """Segments and collects fluorescence data for a single movie.

    Parameters:
    movie: the movie to be analyzed
    model: the pretrained stardist model used for segementation
    num_frames: dimension in time
    num_ch: number of fluorescence channels to analyze
    dim_y: number of rows in image
    dim_x: number of columns in image
    threshold: confidence threshold for including an object detected by stardist

    Returns:
    mask: an 8-bit tiff mask, with each cell labeled, for every frame
    data_list: a Pandas DataFrame containing fluorescence values for each
     cell in each frame
    """

    # Assumes GCaMP is second channel if there are two channels
    if num_ch == 1:
        primary_ch = 0
    else:
        primary_ch = 1
        secondary_ch = 0

    # create an empty mask image with size equal to the movie
    mask_img = np.zeros((num_frames, dim_y, dim_x))
    # initiates tracker for object tracking between frames
    tracker = CentroidTracker(maxDisappeared=num_frames)
    data_list = []

    for frame in range(num_frames):
        # prepare GCaMP frame for stardist
        if 'secondary_ch' in locals():
            img = normalize(movie[frame,primary_ch,:,:],1,99.8,axis=(0,1))
        else:
            img = normalize(movie[frame,:,:],1,99.8,axis=(0,1))

        # predict labels using stardist
        labels, details = model.predict_instances(img, prob_thresh=threshold)
        # get connected components in each frame
        label_props = measure.regionprops(labels)
        # find center position for each region
        centroids = [props.centroid for props in label_props]
        # convert to centroid: property pair for each label
        label_props = {centroid: props for centroid, props in zip(centroids, label_props)}
        # track the objects!
        objects = tracker.update(centroids)

        # collect data for each object
        for obj, centroid in objects.items():
            # this try-except statement handles objects that
            # are missing in some frames
            try:
                cur_label = label_props[centroid].label
                label_mask = labels==cur_label
                mask_img[frame,label_mask] = obj+1

                area = np.sum(label_mask)

                if 'secondary_ch' in locals():
                    primary_mean = np.mean(movie[frame,primary_ch,label_mask])
                    primary_intden = np.sum(movie[frame,primary_ch,label_mask])
                    secondary_mean = np.mean(movie[frame,secondary_ch,label_mask])
                    secondary_intden = np.sum(movie[frame,secondary_ch,label_mask])

                    data = pd.Series({'frame': frame,
                                      'cell' : obj+1,
                                      'area': area,
                                      'primary_mean' : primary_mean,
                                      'primary_intden' : primary_intden,
                                      'secondary_mean' : secondary_mean,
                                      'secondary_intden' : secondary_intden})
                else:
                    primary_mean = np.mean(movie[frame,label_mask])
                    primary_intden = np.sum(movie[frame,label_mask])
                    data = pd.Series({'frame': frame,
                                      'cell' : obj+1,
                                      'area': area,
                                      'primary_mean' : primary_mean,
                                      'primary_intden' : primary_intden})

                data_list.append(data)
            except KeyError:
                pass

    return mask_img, pd.DataFrame(data_list)

def main():
    tifs = []
    folders = []
    for root, dirs, files in os.walk(args.indir):
        for file in files:
            if file.endswith('tif') & ('mask' not in file):
                tifs.append(Path(root) / file)
                if Path(root) not in folders:
                    folders.append(Path(root))

    if not args.summarize_only:
        model = StarDist2D(None, name='gcamp-stardist', basedir='models')

        for tif in tifs:
            print(("Analyzing %s..." % str(tif.stem)), end='', flush=True)
            movie = imread(str(tif))
            num_frames, num_ch, dim_y, dim_x = get_movie_dims(movie)
            labels, df = analyze_gcamp(movie, model, num_frames, num_ch, dim_y, dim_x)
            savedir = tif.parent
            mask_file = savedir / (tif.stem + '_mask.tif')
            data_file = savedir / (tif.stem + '_analysis.csv')
            save_tiff_imagej_compatible(mask_file, labels.astype("uint8"), axes="TYX")
            df.to_csv(data_file)
            print("done!")

    for folder in folders:
        print(("Summarizing %s...") % str(folder), end='', flush=True)
        summary_dfs = summarize_folder(folder)
        savedir = folder.parent
        for summary, df in summary_dfs.items():
            df.to_csv(savedir / (folder.stem + '_' + summary + '.csv'))
        print('done!')

    print("Mischief managed :)")

if __name__ == '__main__':
    main()
