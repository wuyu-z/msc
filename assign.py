# Imports
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Own libs.
from models.clustering.leiden_representations import assign_additional_only


def assign_cluster(resolution,meta_field,folds_pickle,h5_complete_path,h5_additional_path,rep_key='z_latent',main_path=None):
    # Folder permissions for cluster.
    os.umask(0o002)
    # H5 File bug over network file system.
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    if main_path is None:
        main_path = os.path.dirname(os.path.realpath(__file__))

    # Default resolutions.
    if resolution is None:
        resolutions = [0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        # resolutions.extend([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    else:
        resolutions = [resolution]

    # Run leiden clustering.
    assign_additional_only(meta_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions)


