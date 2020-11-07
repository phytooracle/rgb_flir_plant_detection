#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-11-06
Purpose: FLIR plant detection
"""

import argparse
from multiprocessing import process
import os
import sys
from detecto import core, utils, visualize
import glob
import cv2
from detecto.core import Model
import numpy as np
import tifffile as tifi
from osgeo import gdal
import pyproj
import utm
import json
import pandas as pd
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='FLIR plant detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img_list',
                        nargs='+',
                        metavar='img_list',
                        help='Images (ex. flir_images/*.tif)')

    parser.add_argument('-m',
                        '--model',
                        help='A .pth model file',
                        metavar='model',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='detect_out')

    parser.add_argument('-g',
                        '--geojson',
                        help='GeoJSON containing plot boundaries',
                        metavar='str',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-c',
                        '--detect_class',
                        nargs='+',
                        help='Classes to detect',
                        metavar='class',
                        type=list,
                        default=['lettuce'])

    parser.add_argument('-d',
                        '--date',
                        help='Scan date',
                        metavar='date',
                        type=str,
                        default=None,
                        required=True)

    return parser.parse_args()

    # args = parser.parse_args()

    # if '/' not in args.dir[-1]:
    #     args.dir = args.dir + '/'

    # return args


# --------------------------------------------------
def get_min_max(box):
    min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    return min_x, min_y, max_x, max_y


# --------------------------------------------------
def get_trt_zones():
    trt_zone_1 = []
    trt_zone_2 = []
    trt_zone_3 = []

    for i in range(3, 19):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_1.append(str(plot))

    for i in range(20, 36):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_2.append(str(plot))

    for i in range(37, 53):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_3.append(str(plot))

    return trt_zone_1, trt_zone_2, trt_zone_3


# --------------------------------------------------
def find_trt_zone(plot_name):
    trt_zone_1, trt_zone_2, trt_zone_3 = get_trt_zones()

    if plot_name in trt_zone_1:
        trt = 'treatment 1'

    elif plot_name in trt_zone_2:
        trt = 'treatment 2'

    elif plot_name in trt_zone_3:
        trt = 'treatment 3'

    else:
        trt = 'border'

    return trt


# --------------------------------------------------
def get_genotype(plot, geojson):
    with open(geojson) as f:
        data = json.load(f)

    for feat in data['features']:
        if feat.get('properties')['ID']==plot:
            genotype = feat.get('properties').get('genotype')

    return genotype


# --------------------------------------------------
def pixel2geocoord(one_img, x_pix, y_pix):
    ds = gdal.Open(one_img)
    c, a, b, f, d, e = ds.GetGeoTransform()
    lon = a * int(x_pix) + b * int(y_pix) + a * 0.5 + b * 0.5 + c
    lat = d * int(x_pix) + e * int(y_pix) + d * 0.5 + e * 0.5 + f

    return (lat, lon)


# --------------------------------------------------
def open_image(img_path):
    a_img = tifi.imread(img_path)
    a_img = cv2.cvtColor(a_img, cv2.COLOR_GRAY2BGR)
    a_img = cv2.normalize(a_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return a_img


# --------------------------------------------------
def process_image(img):
    args = get_args()
    cont_cnt = 0
    lett_dict = {}

    model = core.Model.load(args.model, args.detect_class)

    plot = img.split('/')[-1].replace('_ortho.tif', '')
    trt_zone = find_trt_zone(plot)
    plot_name = plot.replace('_', ' ')
    print(f'Image: {plot_name}')
    genotype = get_genotype(plot_name, args.geojson)
    a_img = open_image(img)

    predictions = model.predict(a_img)
    labels, boxes, scores = predictions
    copy = a_img.copy()

    for i, box in enumerate(boxes):
        if scores[i] >= 0.1:
            cont_cnt += 1

            min_x, min_y, max_x, max_y = get_min_max(box)
            center_x, center_y = ((max_x+min_x)/2, (max_y+min_y)/2)
            nw_lat, nw_lon = pixel2geocoord(img, min_x, max_y)
            se_lat, se_lon = pixel2geocoord(img, max_x, min_y)

            nw_e, nw_n, _, _ = utm.from_latlon(nw_lat, nw_lon, 12, 'N')
            se_e, se_n, _, _ = utm.from_latlon(se_lat, se_lon, 12, 'N')

            area_sq = (se_e - nw_e) * (se_n - nw_n)
            lat, lon = pixel2geocoord(img, center_x, center_y)
            lett_dict[cont_cnt] = {
                'date': args.date,
                'treatment': trt_zone,
                'plot': plot,
                'genotype': genotype,
                'lon': lon,
                'lat': lat,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'nw_lat': nw_lat,
                'nw_lon': nw_lon,
                'se_lat': se_lat,
                'se_lon': se_lon,
                'bounding_area_m2': area_sq
            }

    df = pd.DataFrame.from_dict(lett_dict, orient='index', columns=['date',
                                                                'treatment',
                                                                'plot',
                                                                'genotype',
                                                                'lon',
                                                                'lat',
                                                                'min_x',
                                                                'max_x',
                                                                'min_y',
                                                                'max_y',
                                                                'nw_lat',
                                                                'nw_lon',
                                                                'se_lat',
                                                                'se_lon',
                                                                'bounding_area_m2']).set_index('date')

    return df


# --------------------------------------------------
def main():
    """Detect plants in image"""

    args = get_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    #img_list = glob.glob(f'{args.dir}*.tif')
    major_df = pd.DataFrame()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        #df = p.map(process_image, img_list)
        df = p.map(process_image, args.img_list)
        major_df = major_df.append(df)

    out_path = os.path.join(args.outdir, f'{args.date}_detection.csv')
    major_df.to_csv(out_path)

    print(f'Done, see outputs in ./{args.outdir}.')


# --------------------------------------------------
if __name__ == '__main__':
    main()