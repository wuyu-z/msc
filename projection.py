# Imports.
from models.evaluation.features import *
from data_manipulation.data import Data
import tensorflow as tf
import argparse
import os
def vectorise(real_hdf5,img_size=224,img_ch=3,z_dim=128,dataset='sampledataset',marker='he',batch_size=64,model='BarlowTwins_3',main_path=None,dbs_path=None,save_img=False):
	path_str = './datasets/%s/he/patches_h224_w224/' % (dataset)
	if not os.path.exists(path_str):
		os.makedirs(path_str)
	shutil.copy(real_hdf5, os.path.join(path_str,'hdf5_%s_he_train.h5'% (dataset)))
	# Folder permissions for cluster.
	os.umask(0o002)
	# H5 File bug over network file system.
	os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
	if main_path is None:
		main_path = os.path.dirname(os.path.realpath(__file__))
	if dbs_path is None:
		dbs_path = os.path.dirname(os.path.realpath(__file__))
	image_height=img_size
	image_width=img_size
	image_channels=img_ch
	checkpoint='./checkpoints/BarlowTwins_3.ckt'
	# Directory handling.
	name_run = 'h%s_w%s_n%s_zdim%s' % (image_height, image_width, image_channels, z_dim)
	data_out_path = os.path.join(main_path, 'data_model_output')
	data_out_path = os.path.join(data_out_path, model)
	data_out_path = os.path.join(data_out_path, dataset)
	data_out_path = os.path.join(data_out_path, name_run)

	# Hyperparameters for training.
	regularizer_scale = 1e-4
	learning_rate_e = 5e-4
	beta_1 = 0.5

	# Model Architecture param.
	layers_map = {512: 7, 448: 6, 256: 6, 224: 5, 128: 5, 112: 4, 56: 3, 28: 2}
	layers = layers_map[image_height]
	spectral = True
	attention = 56
	init = 'xavier'
	# init       = 'orthogonal'

	# Handling of different models.
	if 'BYOL' in model:
		z_dim = 256
		from models.selfsupervised.BYOL import RepresentationsPathology
	elif 'SimCLR' in model:
		from models.selfsupervised.SimCLR import RepresentationsPathology
	elif 'SwAV' in model:
		learning_rate_e = 1e-5
		from models.selfsupervised.SwAV import RepresentationsPathology
	elif 'SimSiam' in model:
		from models.selfsupervised.SimSiam import RepresentationsPathology
	elif 'Relational' in model:
		from models.selfsupervised.RealReas import RepresentationsPathology
	elif 'BarlowTwins' in model:
		from models.selfsupervised.BarlowTwins import RepresentationsPathology

	# Collect dataset.
	data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels,
				batch_size=batch_size, project_path=dbs_path)
	print(type(data.training))
	# Run PathologyContrastive Encoder.
	with tf.Graph().as_default():
		# Instantiate Model.
		contrast_pathology = RepresentationsPathology(data=data, z_dim=z_dim, layers=layers, beta_1=beta_1, init=init,
													  regularizer_scale=regularizer_scale, spectral=spectral,
													  attention=attention, learning_rate_e=learning_rate_e,
													  model_name=model)
		# Run projections into H5.
		real_encode_contrastive_from_checkpoint(model=contrast_pathology, data=data, data_out_path=main_path,
												checkpoint=checkpoint, real_hdf5=real_hdf5, batches=batch_size,
												save_img=save_img)
	result_path='./results/%s/%s/h224_w224_n3_zdim128/%s'% (model,dataset,real_hdf5.split('/')[1])
	if os.path.isfile(result_path):
		return result_path
	else:
		return None

