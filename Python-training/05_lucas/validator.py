#!/usr/bin/env python3

"""
project:    CTU@GeoHarmonizer
date:       2021-07-19
author:     lukas.brodsky@fsv.cvut.cz
purpose:    discrete classes land cover validation
"""

import os
import sys
import datetime
import yaml
import pickle

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from gio import *



class Validator:
    """Land product validation
    """
    def __init__(self, config):
        print('\n')
        print('Validation project initialized!')
        # read configuration
        if type(config) is dict:
            self.cfg = config
        elif os.path.isfile(config):
            with open(config) as file:
                self.cfg = yaml.load(file, Loader=yaml.FullLoader)
                # print('Config ready.')        
        else:
            print('Cannot read configuration.')

        # init validation project
        self.val_time = self._generated_at(day=True, time=False)
        self.report_dir = os.path.join(self.cfg['report']['path'],
                                       self.cfg['report']['dir_name'] + '_' + self.val_time)
        if not os.path.isdir(self.report_dir):
            print('Creating report dir at {}'.format(self.report_dir))
            os.makedirs(self.report_dir)

        # inputs
        input = self.cfg['input']
        self.in_ras = os.path.join(input['path'], input['in_ras'])
        self.ndv = input['ndv']
        self.no_data_value = [self.ndv, 'None', None, '']
        self.in_vec = os.path.join(input['path'], input['in_vec'])
        self.ref_att = input['ref_att']
        self.agg = agg = {}
        self.ogr_format = self._get_ogr_format()
        self.epsg = self._get_epsg()
        print('Inputs: ')
        print(os.path.basename(self.in_ras))
        print(os.path.basename(self.in_vec))
        print('\n')

    def _get_epsg(self):
        """identify epsg code from raster
        """
        d = gdal.Open(self.in_ras)
        proj = osr.SpatialReference(wkt=d.GetProjection())
        return int(proj.GetAttrValue('AUTHORITY', 1))

    def _get_ogr_format(self):
        """identify ogr format from input vector file
        """
        if os.path.basename(self.in_vec).split('.')[-1] == 'shp':
            ogr_format = 'ESRI Shapefile'
        elif os.path.basename(self.in_vec).split('.')[-1] == 'gpkg':
            ogr_format = 'GPKG'

        return ogr_format
    
    def check_inputs(self):
        """Check input geodata: raster an vector
           before running validation task
        """
        self.inputs_valid = False
        if os.path.isfile(self.in_ras) and os.path.isfile(self.in_vec):
            ras_valid = check_read_raster(self.in_ras)
            vec_valid = check_read_vector(self.in_vec)
            if ras_valid and vec_valid:
                self.inputs_valid = True
            else:
                print('Raster map is {}'.format(ras_valid))
                print('Vector reference is {}'.format(vec_valid))
        else:
            print('Input files not found.')

        return self.inputs_valid


    def overlay(self, aggregation=None):
        """ run validation process
            currently raster map and vector reference overlay
        """
        print('\n')
        if aggregation is not None:
            self.agg = aggregation
        self._vec_ras_overlay()
        print('| {} reference points.'.format(len(self.overlay_data)))

        return 0

    def print_validation_config(self):
        """..."""
        print(self.cfg)

    def _get_y_true_pred(self):
        """..."""
        y_true = []
        y_pred = []

        for i in range(len(self.overlay_data)):
            if self.overlay_data[i]['map'] not in self.no_data_value and \
                    self.overlay_data[i]['reference'] not in self.no_data_value:
                y_true.append(self.overlay_data[i]['reference'])
                y_pred.append(self.overlay_data[i]['map'])

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        return y_true_arr, y_pred_arr

    def report(self):
        """..."""
        self.sklearn_report = {}
        self.classical_report = {}
        # labels
        y_true_arr, y_pred_arr = self._get_y_true_pred()
        self.legend_labels = np.unique(y_true_arr)
        # print(self.legend_labels)

        cls_report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
        print('\n')
        print('Machine learning validation indicators (per class): ')
        print('---')
        print(classification_report(y_true_arr, y_pred_arr, labels=self.legend_labels, zero_division=0))
        # prepare report
        self.sklearn_report_ = classification_report(y_true_arr, y_pred_arr, labels=self.legend_labels, zero_division=0)
        self.sklearn_report = classification_report(y_true_arr, y_pred_arr, labels=self.legend_labels,
                                                    output_dict=True, zero_division=0)
        c_matrix = confusion_matrix(y_true_arr, y_pred_arr)
        # print(conf_matrix)
        print('Classical LC validation indicators: ')
        print('---')
        overall_accuracy_ = (np.sum(c_matrix.diagonal()) / np.sum(c_matrix))
        producers_accuracy_ = (self.sklearn_report['weighted avg']['precision'])
        users_accuracy_ = (self.sklearn_report['weighted avg']['recall'])
        kappa_ = (cohen_kappa_score(y_true_arr, y_pred_arr, labels=None, weights=None))
        self.classical_report['overall_accuracy'] = "{0:.4f}".format(overall_accuracy_)
        self.classical_report['producers_accuracy'] = "{0:.4f}".format(producers_accuracy_)
        self.classical_report['users_accuracy'] = "{0:.4f}".format(users_accuracy_)
        self.classical_report['kappa'] = "{0:.4f}".format(kappa_)
        for k in self.classical_report.keys():
            print(k, ': ', self.classical_report[k])
        print('\n')

        return 0

    def save_report(self):
        """... """
        print('Saving validation report:')
        print('---')
        project = self.cfg['project']['abbrev']
        report_fn = os.path.join(self.report_dir,
                                  project + '_validation_report.txt')
        text = open(report_fn, "w+")

        # report content
        text.write('VALIDATION REPORT' + '\n')
        text.write('---' + '\n')
        text.write('Project: ' + project + '\n')
        text.write('Generated at: ' + self.val_time + '\n\n')

        text.write('\n')
        text.write('Inputs: ' + '\n')
        text.write('---' + '\n')
        text.write('Data path: ' + os.path.dirname(self.in_ras) + '\n')
        text.write('Map: ' + os.path.basename(self.in_ras) + '\n')
        text.write('Reference: ' + os.path.basename(self.in_vec) + '\n')
        text.write('\n\n')
        text.write('Land product classes: ' + '\n')
        text.write('---' + '\n')
        text.write(str(self.legend_labels) + '\n')
        text.write('\n\n')

        text.write('Machine learning QI: ' + '\n')
        text.write('---' + '\n')
        text.write(self.sklearn_report_)
        # for k, v in self.sklearn_report.items():
            # print(k, v)
            # text.write(str(k) + ': ' + str(v) + '\n')
        text.write('\n\n')

        text.write('Classical land cover QI: ' + '\n')
        text.write('---' + '\n')
        for k, v in self.classical_report.items():
            # print(k, v)
            text.write(str(k) + ': ' + str(v) + '\n')
        text.write('\n\n')
        text.close()
        print(os.path.basename(report_fn))
        print('\n')

        return 0

    def _get_cache_fn(self, name_string):
        """... """
        project = self.cfg['project']['abbrev']
        cache_fn = os.path.join(self.report_dir,
                                 project + '_validation_overlay_data_' + name_string + '.pkl')

        return cache_fn

    def save_validation_data_to_cache(self, name_string):
        """..."""

        cache_fn = self._get_cache_fn(name_string)

        # serialize results
        try:
            with open(cache_fn, 'wb') as handle:
                pickle.dump(self.overlay_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Validation overlay data saved to pickle.')
        except:
            print('Cannot save the validation overlay data to pickle')

        return 0


    def load_validation_data_from_cache(self, name_string):
        """..."""
        cache_fn = self._get_cache_fn(name_string)
        print(cache_fn)

        # load serialized data
        try:
            with open(cache_fn, 'rb') as handle:
                self.overlay_data = pickle.load(handle)
            print('Validation overlay data loaded from pickle.')

        except:
            print('Cannot load the validation overlay data from cache')

        return 0


    def save_vec(self):
        """save overlay data to vector file
        """

        print('Saving validation data:')
        print('---')
        if self.cfg['validation_points']['ogr_format'] == 'ESRI Shapefile':
            suffix = 'shp'
        elif self.cfg['validation_points']['ogr_format'] == 'GPKG':
            suffix = 'gpkg'

        self.validation_fn = os.path.join(self.report_dir,
                                          self.cfg['validation_points']['file_name'] + '.' + suffix)
        try:
            create_vector(self.overlay_data, self.validation_fn, self.cfg['validation_points']['ogr_format'], self.epsg)
            print('Vector data: {} created from overlay data'.format(os.path.basename(self.validation_fn)))
        except:
            print('Cannot create vector file for overlay data')

        return 0

    def show_confusion_matrix(self):
        """..."""
        # self.overlay_data
        y_true_arr, y_pred_arr = self._get_y_true_pred()
        # lp_classes = np.unique(y_true_arr)
        classes = unique_labels(y_true_arr, y_pred_arr)
        self.plot_confusion_matrix(y_true_arr, y_pred_arr, classes)
        plt.show()

        return 0

    def save_confusion_matrix(self):
        """..."""
        project = self.cfg['project']['abbrev']
        figure_fn = os.path.join(self.report_dir, project + '_confusion_matrix.png')
        y_true_arr, y_pred_arr = self._get_y_true_pred()
        # lp_classes = np.unique(y_true_arr)
        classes = unique_labels(y_true_arr, y_pred_arr)
        self.plot_confusion_matrix(y_true_arr, y_pred_arr, classes)
        plt.savefig(figure_fn, format='png', dpi=300)

        return 0

    def show_normalized_confusion_matrix(self):
        """..."""
        # self.overlay_data
        y_true_arr, y_pred_arr = self._get_y_true_pred()
        # lp_classes = np.unique(y_true_arr)
        classes = unique_labels(y_true_arr, y_pred_arr)
        self.plot_confusion_matrix(y_true_arr, y_pred_arr, classes, normalize=True)
        plt.show()

        return 0

    def save_normalized_confusion_matrix(self):
        """..."""
        project = self.cfg['project']['abbrev']
        figure_fn = os.path.join(self.report_dir, project + '_normalized_confusion_matrix.png')
        y_true_arr, y_pred_arr = self._get_y_true_pred()
        # lp_classes = np.unique(y_true_arr)
        classes = unique_labels(y_true_arr, y_pred_arr)
        # print('Classes: {}'.format(classes))
        self.plot_confusion_matrix(y_true_arr, y_pred_arr, classes, normalize=True)
        plt.savefig(figure_fn, format = 'png', dpi = 300)

        return 0


    def _generated_at(self, day=True, time=False):
        """Formated time of validation processing
        """
        if day & time:
            format = "%Y%m%d_%H%M"
        elif day:
            format = "%Y%m%d"
        elif time:
            print('Incorrect format definition!')

        now = datetime.datetime.now()
        generated_at_string = now.strftime(format)

        return generated_at_string


    def _bbox_to_pixel_offsets(self, gt, bbox):
        originX = gt[0]
        originY = gt[3]
        pixel_width = gt[1]
        pixel_height = gt[5]
        x1 = int((bbox[0] - originX) / pixel_width)
        x2 = int((bbox[1] - originX) / pixel_width) + 1

        y1 = int((bbox[3] - originY) / pixel_height)
        y2 = int((bbox[2] - originY) / pixel_height) + 1

        xsize = x2 - x1
        ysize = y2 - y1

        return (x1, y1, xsize, ysize)


    def _vec_ras_overlay(self):
        """Retrieve vector-raster overaly dataset
           for validation report
        """
        # sys.stdout.write("\rOverlay process started!")
        # sys.stdout.flush()

        rds = gdal.Open(self.in_ras, gdal.GA_ReadOnly)
        rb = rds.GetRasterBand(1)

        rgt = get_image_geo_attributes(self.in_ras)[-1]
        vds = ogr.Open(self.in_vec, gdal.OF_VECTOR)
        vlyr = vds.GetLayer(0)

        # Loop through points
        self.overlay_data = []
        self.passed = []
        self.failed = []
        feat = vlyr.GetNextFeature()
        poc = 0
        while feat is not None:
            poc += 1
            point_data = {}
            status = None

            id = feat.GetField('point_id')
            # sys.stdout.write("\rProcessing point id: %s   " % (id))
            # sys.stdout.flush()
            att = feat.GetField(self.ref_att)
            src_offset = self._bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)
            x_coord = feat.geometry().GetX()
            y_coord = feat.geometry().GetY()
            # geom = (feat.geometry().GetEnvelope())
            map_code = src_array[0][0] 
            # agg = {2: [6, 11]} 

            # print('orig: ', att, map_code, status)           
            for k, vs in self.agg.items():
                if att in vs or map_code in vs:
                    att = map_code = k
	           
            # compare map and reference
            if att == map_code:
                status = True
#            else:            
#                for k, vs in agg.items():
#                    for v in vs:
#                        if (att == k and map_code == k) or (att == k and map_code == v) or (att == v and map_code == k): 
#                            status = True
#                            att = map_code = k
                            
            if status is True:
                self.passed.append(id)
                
            if status is None and map_code not in self.no_data_value and att not in self.no_data_value:
                self.failed.append(id)
                status = False
            # print(att, map_code, status) 

            point_data['id'] = id
            point_data['x'] = x_coord
            point_data['y'] = y_coord
            point_data['reference'] = att
            point_data['map'] = map_code
            point_data['status'] = status

            # print('Point: {} | reference: {} | map: {} | {}'.format(id, att, src_array[0][0], status))
            if point_data['reference'] not in self.no_data_value or point_data['map'] not in self.no_data_value:
                self.overlay_data.append(point_data)
            feat = vlyr.GetNextFeature()

        # print('{} points evaluated'.format(len(self.overlay_data)))
        vds = None
        rds = None
        sys.stdout.write("\rProcessed: 100% ")
        sys.stdout.flush()

        return 0

    def short_report(self):
        """print short report
        """
        print('Validation indicators: ')
        print('---')
        print('Overall map accuracy is {} %'.format(round((len(self.passed) /
                                            (len(self.passed) + len(self.failed))) * 100.0, 2)))
        print('No. passed: {}'.format(len(self.passed)))
        print('No. failed: {}'.format(len(self.failed)))
        print('No. points used in validation: {}'.format(len(self.failed) + len(self.passed)))

        return 0


    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This sklearn function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + .001)
            # print("Normalized confusion matrix")

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # label size
        if len(classes) < 10:
            plt.rcParams.update({'font.size': 10})
        if len(classes) > 10 and len(classes) < 20:
            plt.rcParams.update({'font.size': 8})
        if len(classes) > 20:
            plt.rcParams.update({'font.size': 5})

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.1f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        return ax

