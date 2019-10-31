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
lbl_cmap = random_label_cmap()

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("indir")
args = parser.parse_args()

def get_movie_dims(movie):
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
    if num_ch == 1:
        primary_ch = 0
    else:
        primary_ch = 1
        secondary_ch = 0

    mask_img = np.zeros((num_frames, dim_y, dim_x))
    tracker = CentroidTracker(maxDisappeared=num_frames)
    data_list = []

    for frame in range(num_frames):
        if 'secondary_ch' in locals():
            img = normalize(movie[frame,primary_ch,:,:],1,99.8,axis=(0,1))
        else:
            img = normalize(movie[frame,:,:],1,99.8,axis=(0,1))
        
        labels, details = model.predict_instances(img, prob_thresh=threshold)
        label_props = measure.regionprops(labels)
        centroids = [props.centroid for props in label_props]
        label_props = {centroid: props for centroid, props in zip(centroids, label_props)}
        objects = tracker.update(centroids)

        for obj, centroid in objects.items():
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
    model = StarDist2D(None, name='gcamp-stardist', basedir='models')
    
    tifs = []
    for root, dirs, files in os.walk(args.indir):
        for file in files:
            if file.endswith('tif') & ('mask' not in file):
                tifs.append(Path(root) / file)
                
    data_list = []
    for tif in tifs:
        movie = imread(str(tif))
        print(("Analyzing %s..." % str(tif.stem)), end='')
        num_frames, num_ch, dim_y, dim_x = get_movie_dims(movie)
        labels, df = analyze_gcamp(movie, model, num_frames, num_ch, dim_y, dim_x)
        savedir = tif.parent
        mask_file = savedir / (tif.stem + '_mask.tif')
        data_file = savedir / (tif.stem + '_analysis.csv')
        save_tiff_imagej_compatible(mask_file, labels.astype("uint8"), axes="TYX")
        data_list.append(df)
        df.to_csv(data_file)
        print("done!")
        
if __name__ == '__main__':
    main()