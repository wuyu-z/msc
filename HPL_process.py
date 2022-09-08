import os
import shutil
import warnings
warnings.filterwarnings("ignore")
import projection
import utilities.h5_handling.create_metadata_h5 as meta
import assign
import h5py
import pandas as pd
import numpy as np
import wsi


def luad(row):
    if'Adenocarcinoma' in row['primary_diagnosis']:
        return 1
    else:
        return 0

def create_meta_csv(file):
    slide_id = ""
    with h5py.File(file, 'r+') as vector:
        # ength=len(x['combined_slides'])
        if 'slides' not in list(vector.keys()):
            vector['slides'] = vector['combined_slides']

        slides = vector['slides'][:].astype(str)
        if 'samples' not in list(vector.keys()):
            vector['samples'] = ['-'.join(slide.split('-')[:3]) for slide in slides]

        slide_id = vector['slides'][0].astype(str)
    csv_path = './testSample/clinical.tsv'
    clinical = pd.read_csv(csv_path, sep='\t', na_values='\'--')
    df_data = clinical.drop_duplicates(subset='case_id')

    df_data['months_to_death'] = df_data['days_to_death'] / 365 * 12
    df_data['months_to_last_follow_up'] = df_data['days_to_last_follow_up'] / 365 * 12

    df_data['slides'] = [slide_id] * df_data.shape[0]
    filter_alive = df_data['vital_status'] == 'Alive'
    df_data['event_ind'] = [1] * df_data.shape[0]
    df_data['event_ind'] = df_data['event_ind'].mask(filter_alive, 0)

    df_data['luad'] = df_data.apply(luad, axis=1)

    df_data['event_data'] = df_data['months_to_death']
    df_data.loc[df_data['event_data'].isnull(), 'event_data'] = \
    df_data[df_data['event_data'].isnull()]['months_to_last_follow_up'].values
    df_data.loc[df_data['event_data'].isnull(), 'event_data'] = np.max(df_data['event_data'])

    retain_fields = ['slides', 'participants', 'luad', 'os_event_ind', 'os_event_data']
    frame_final = df_data.rename(
        columns={'case_submitter_id': 'participants', 'event_ind': 'os_event_ind', 'event_data': 'os_event_data'})
    frame_final = frame_final[retain_fields]
    if not os.path.exists("./temp"):
        os.makedirs('./temp')
    output_path='./temp/clinical.csv'
    frame_final.to_csv(output_path, encoding='utf-8', index=False)
    return output_path


'''
h5='testsample'
h5_path='./%s.h5' % h5

result_file=projection.vectorise(real_hdf5=h5_path,dataset='sampledataset')
if result_file is None:
    print("File Error")
path,file=os.path.split(result_file)
new_file=os.path.join(path, '%s_vector.h5' % (h5_path.split('/')[1].split('.')[0]))
os.rename(result_file,new_file)
shutil.copy(new_file,'./output')
shutil.copy(h5_path,'./output')
_,file=os.path.split(new_file)
new_file=os.path.join('./output',file)




meta_csv=create_meta_csv(new_file)
#new_file='./output/mytest_vector.h5'
#meta_csv='./temp/clinical.csv'


meta_h5_file=meta.include_metadata(meta_file=meta_csv,meta_name='lungsubtype_survival',list_meta_field=['luad', 'os_event_ind', 'os_event_data'],matching_field='slides',h5_file=new_file,override=True)
meta_path,meta_file=os.path.split(meta_h5_file)
meta_file='hdf5_'+meta_file
os.rename(meta_h5_file,os.path.join(meta_path,meta_file))
#---------------add hdf5 file name


h5_compelte_path='./results/BarlowTwins_3/sampledataset/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5'
h5_additional_path='./output/hdf5_%svector_lungsubtype_survival.h5' % h5
lung_subtype=['lung_subtypes_nn250','./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl']
overall_survive=['luad_overall_survival_nn250','./utilities/files/LUAD/overall_survival_TCGA_folds.pkl']
list_args=[lung_subtype,overall_survive]
for i in list_args:
    assign.assign_cluster(resolution=2.0,meta_field=i[0],folds_pickle=i[1],h5_complete_path=h5_compelte_path,h5_additional_path=h5_additional_path)
'''
h5='testsample'
h5_path='./%s.h5' % h5
cluster_path=[['./results/BarlowTwins_3/sampledataset/h224_w224_n3_zdim128/lung_subtypes_nn250/adatas/%s_vector_lungsubtype_survival_leiden_2p0__fold0.csv' % h5,'lungsubtype'],
              ['./results/BarlowTwins_3/sampledataset/h224_w224_n3_zdim128/luad_overall_survival_nn250/adatas/%s_vector_lungsubtype_survival_leiden_2p0__fold4.csv' % h5,'survival']]

for i in cluster_path:
    wsi.plot_wsi(i[0],h5_path,i[1])



