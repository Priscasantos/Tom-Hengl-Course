#!/usr/bin/env python3

"""
project:    CTU@GeoHarmonizer
date:       2021-07-19
author:     lukas.brodsky@fsv.cvut.cz
purpose:    manage geographic data input / output operations with GDAL/OGR
"""

import os
import glob
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr
from osgeo.gdalconst import *

import numpy as np



def get_image_attribute(image_filename):
    """ Use GDAL to open image and return attributes
    Args:
        image_filename (str): image filename
    Returns:
        tuple: nrow (int), ncol (int), nband (int), NumPy datatype (type)
    """
    try:
        image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except Exception as e:
        # logger.error('Could not open image dataset ({f}): {e}'
                     # .format(f=image_filename, e=str(e)))
        print('Could not open image dataset.')
        raise

    nrow = image_ds.RasterYSize
    ncol = image_ds.RasterXSize
    nband = image_ds.RasterCount
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        image_ds.GetRasterBand(1).DataType)

    return (nrow, ncol, nband, dtype)


def get_image_geo_attributes(image_filename):
    """ Use GDAL to open image and return geo-attributes
    Args:
        image_filename (str): image filename
    Returns:
        tuple: projection (str), geo_transform (tuple)
    """
    try:
        if os.path.isfile((image_filename)):
            image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except Exception as e:
        # logger.error('Could not open image dataset ({f}): {e}'
                     # .format(f=image_filename, e=str(e)))
        print('Could not open image dataset.')
        raise

    projection = image_ds.GetProjection()
    geo_transform = image_ds.GetGeoTransform()

    return projection, geo_transform


def read_image(image_filename, bands):
    """ Use GDAL to open image and return geo-attributes
    Args:
        image_filename (str): image filename
        bands (list): list of bands to read from image
    Returns:
        ndarray: img_arr [bnads, y, x]
    """
    try:
        if os.path.isfile((image_filename)):
            image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except Exception as e:
        # logger.error('Could not open image dataset ({f}): {e}'
                     # .format(f=image_filename, e=str(e)))
        print('Could not open image dataset.')

        raise

    # assert(image_ds)
    rb = image_ds.GetRasterBand(1)
    b1 = src_array = rb.ReadAsArray()

    img_arr = np.zeros((len(bands), b1.shape[0], b1.shape[1]), dtype=np.float)
    # img_arr[img_arr == 0] = np.nan

    b = 0
    for band in bands:
        data_b = image_ds.GetRasterBand(band)
        img_arr[b, :, :] = data_b.ReadAsArray()
        b += 1

    return img_arr


# vector_file_name = in_vec
def read_vector(vector_filename):
    """
    :param vector_file_name:
    :return:
    """
    print(vector_filename)
    try:
        vds = ogr.Open(vector_filename, gdal.OF_VECTOR)

    except Exception as e:
        # logger.error('Could not open vector dataset ({f}): {e}'
                     # .format(f=image_filename, e=str(e)))
        print('Could not open vector dataset.')
        raise

    vlyr = vds.GetLayer()

    return vlyr


def create_vector(overlay_data, file_name, ogr_format, epsg):
    """Create a vector file from overlay data
    """

    drv = ogr.GetDriverByName(ogr_format)
    vector_ds = drv.CreateDataSource(file_name)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # create the layer
    layer_name = os.path.basename(file_name).split('.')[0]
    layer = vector_ds.CreateLayer(layer_name, srs, ogr.wkbPoint)

    # Add the attribute fields
    attributes = list(overlay_data[0].keys())
    if 'id' in attributes:
        field_name = ogr.FieldDefn("id", ogr.OFTInteger)
        layer.CreateField(field_name)
    if 'x' in attributes:
        field_x = ogr.FieldDefn("x", ogr.OFTReal)
        layer.CreateField(field_x)
    if 'y' in attributes:
        field_y = ogr.FieldDefn("y", ogr.OFTReal)
        layer.CreateField(field_y)
    if 'reference' in attributes:
        field_reference = ogr.FieldDefn("reference", ogr.OFTInteger)
        layer.CreateField(field_reference)
    if 'map' in attributes:
        field_map = ogr.FieldDefn("map", ogr.OFTInteger)
        layer.CreateField(field_map)
    if 'status' in attributes:
        field_status = ogr.FieldDefn("status", ogr.OFTInteger)
        layer.CreateField(field_status)

    # Process the attributes and features to the vector
    # item = overlay_data[1]
    for item in overlay_data:
        # print(item)
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes
        if item['status'] is not None:
            if item['id'] is not None:
                feature.SetField('id', item['id'])
            if item['x'] is not None:
                feature.SetField('x', item['x'])
            if item['y'] is not None:
                feature.SetField('y', item['y'])
            if item['reference'] is not None:
                feature.SetField('reference', int(item['reference']))
            if item['map'] is not None:
                feature.SetField('map', int(item['map']))
            if item['status'] is not None:
                if item['status']:
                    feature.SetField('status', 1)
                else:
                    feature.SetField('status', 0)

            # create the WKT for the feature using Python string formatting
            wkt = "POINT(%f %f)" % (float(item['x']), float(item['y']))
            # Create the point from the Well Known Txt
            point = ogr.CreateGeometryFromWkt(wkt)
            # Set the feature geometry using the point
            feature.SetGeometry(point)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Dereference the feature
            feature = None

    vector_ds = None


    return 0



def create_template_vector(vector_filename, ogr_format, epsg, example_vector, new_att):
    """
    :param vector_fn:
    :return:
    """
    if os.path.isfile(vector_filename):
        # os.remove(vector_filename)
        for fn in glob.glob(vector_filename.split('.')[0] + '*'):
            os.remove(fn)

    # create
    driver = ogr.GetDriverByName(ogr_format)
    if driver is None:
        print('%s driver not available.\n' % ogr_format)

    dataset_out = driver.CreateDataSource(vector_filename)
    if dataset_out is None:
        print('Creation of output file failed.\n')

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    layer_name = os.path.basename(vector_filename).split('.')[0]
    outlayer = dataset_out.CreateLayer(layer_name, geom_type=ogr.wkbPoint, srs=sr)

    # setup attributes
    example_ds = driver.Open(example_vector, 0)
    if example_ds is None:
        print('Could not open %s' % (example_vector))
    else:
        example_layer = example_ds.GetLayer()
        layer_definition = example_layer.GetLayerDefn()
        for i in range(layer_definition.GetFieldCount()):
            field_def = layer_definition.GetFieldDefn(i)
            if outlayer.CreateField(field_def) != 0:
                print('Creating %s field.\n' % field_def.GetNameRef())

    # Add a field for map value TODO: integer to 16b.?
    if new_att is not None:
        new_field = ogr.FieldDefn(new_att, ogr.OFTInteger)
        outlayer.CreateField(new_field)

    # TODO: better closing
    outlayer = None
    example_layer = None

    return 0


def check_read_raster(image_filename):
    """ Use GDAL to check open raster image
    Args:
        image_filename (str): image filename
    Returns:
        bool: valid (boolean)
    """
    valid = False
    image_ds = None
    try:
        image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
        if image_ds is not None:
            valid = True
    except RuntimeError as e:
        print('Could not open image dataset.')
        raise
    image_ds = None

    return valid


def check_read_vector(vector_filename):
    """ Use GDAL to check open raster image
    Args:
        image_filename (str): image filename
    Returns:
        bool: valid (boolean)
    """
    valid = False
    vector_ds = None

    if os.path.basename(vector_filename).split('.')[-1] == 'shp':
        ogr_format = 'ESRI Shapefile'
    elif os.path.basename(vector_filename).split('.')[-1] == 'gpkg':
        ogr_format = 'GPKG'

    driver = ogr.GetDriverByName(ogr_format)
    if driver is None:
        print('%s driver not available.\n' % ogr_format)

    try:
        vector_ds = driver.Open(vector_filename, 0)
        if vector_ds is not None:
            valid = True
    except RuntimeError as e:
        print('Could not open image dataset.')
        raise

    return valid



