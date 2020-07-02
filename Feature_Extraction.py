import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import cv2
import os
import json
import sys
import lmdb
from collections import defaultdict
import random
from utils import *
from datetime import datetime
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
# GPU_ID
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)

#############
# Visual Feature Extraction
# Columbia University
#############

# Specify data paths
corpus_path = '/root/LDC/'
working_path = '/root/shared/'
model_path = '/root/models/'
TEST_MODE = True
if TEST_MODE:
    print("TEST_MODE is open")

# Version Setting
# Set evaluation version as the prefix folder
version_folder = '' # 'E/' could be ignored if there is no version management


# Input: LDC unpacked data, CU visual grounding and instance matching moodels, UIUC text mention results, CU object detection results
# Input Paths
# Source corpus data paths
print('Check Point: Raw Data corpus_path change',corpus_path)
parent_child_tab = corpus_path + 'docs/parent_children.tab'
kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
jpg_path = corpus_path + 'data/jpg/jpg/'


#UIUC text mention result paths
video_asr_path = working_path + 'uiuc_asr_files/' + version_folder +'ltf_asr/'
video_map_path = working_path + 'uiuc_asr_files/' + version_folder +'map_asr/'
print('Check Point: text mentions path change',video_asr_path)


# CU object detection result paths
det_results_path_img = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34a.pkl' # jpg images
det_results_path_kfrm = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34b.pkl' # key frames
print('Check Point: path change:','\n',det_results_path_img,'\n', det_results_path_kfrm,'\n')


# Model Paths
# CU visual grounding and instance matching moodel paths
grounding_model_path = model_path + 'model_ELMo_PNASNET_VOA_norm'
matching_model_path = model_path + 'model_universal_no_recons_ins_only'


# Output: CU visual grounding and instance matching features
# Output Paths
# CU visual grounding feature paths
out_path_jpg = working_path + 'cu_grounding_matching_features/' + 'semantic_features_jpg.lmdb'
out_path_kfrm = working_path + 'cu_grounding_matching_features/' + 'semantic_features_keyframe.lmdb'

# CU instance matching feature paths
out_path_jpg = working_path + 'cu_grounding_matching_features/' + 'instance_features_jpg.lmdb'
out_path_kfrm = working_path + 'cu_grounding_matching_features/' + 'instance_features_keyframe.lmdb'


#loading grounding pretrained model
print('Loading grounding pretrained model...')
sess, graph = load_model(grounding_model_path,config)
input_img = graph.get_tensor_by_name("input_img:0")
mode = graph.get_tensor_by_name("mode:0")
v = graph.get_tensor_by_name("image_local_features:0")
v_bar = graph.get_tensor_by_name("image_global_features:0")
print('Loading done.')



#preparing dicts
parent_dict, child_dict = create_dict(parent_child_tab)
id2dir_dict_kfrm = create_dict_kfrm(kfrm_path, kfrm_msb, video_asr_path, video_map_path)
#jpg
path_dict = create_path_dict(jpg_path)
#mp4
path_dict.update(create_path_dict_kfrm(id2dir_dict_kfrm))
# print('HC000TJCP' in id2dir_dict_kfrm.keys())
# print(id2dir_dict_kfrm.keys())

#loading object detection results
with open(det_results_path_img, 'rb') as f:
    dict_obj_img = pickle.load(f)
    
with open(det_results_path_kfrm, 'rb') as f:
    dict_obj_kfrm = pickle.load(f)
print(datetime.now())
# print(child_dict)
# Semantic Features
# about 8 hours in total for Instance Features


#opening lmdb environment
lmdb_env_jpg = lmdb.open(out_path_jpg, map_size=int(1e11), lock=False)
lmdb_env_kfrm = lmdb.open(out_path_kfrm, map_size=int(1e11), lock=False)

#about 1.5 hour
print(datetime.now())
missed_children_jpg = []
for i, key in enumerate(dict_obj_img):
    imgs,_ = fetch_img(key+'.jpg.ldcc', parent_dict, child_dict, path_dict, level = 'Child')
    if len(imgs)==0:
        missed_children_jpg.append(key)
        continue
    img_batch, bb_ids, bboxes_norm = batch_of_bbox(imgs[0], dict_obj_img, key,\
                                        score_thr=0, filter_out=False)
    if len(bb_ids)>0:
        feed_dict = {input_img: img_batch, mode: 'test'}
        v_pred = sess.run([v], feed_dict)[0]
        for j,bb_id in enumerate(bb_ids):
            mask = mask_fm_bbox(feature_map_size=(19,19),bbox_norm=bboxes_norm[j,:],order='xyxy')
            if np.sum(mask)==0:
                continue
            img_vec = np.average(v_pred[j,:], weights = np.reshape(mask,[361]), axis=0)
            save_key = key+'/'+str(bb_id)
            with lmdb_env_jpg.begin(write=True) as lmdb_txn:
                lmdb_txn.put(save_key.encode(), img_vec)
    if TEST_MODE:
        # [break] only for dockerization testing
        break  
    sys.stderr.write("Stored for image {} / {} \r".format(i, len(dict_obj_img)))
print(datetime.now())
 
#about 4-6 hours
print(datetime.now())
missed_children_kfrm = []
for i, key in enumerate(dict_obj_kfrm):
    # key+'.mp4.ldcc'
#     print('path from obj detecton for kfrm:',key+'.mp4.ldcc')
    imgs,_ = fetch_img(key+'.mp4.ldcc', parent_dict, child_dict, path_dict, level = 'Child')
    if len(imgs)==0:
        missed_children_kfrm.append(key)
        continue
    img_batch, bb_ids, bboxes_norm = batch_of_bbox(imgs[0], dict_obj_kfrm, key,\
                                      score_thr=0, filter_out=False)
    if len(bb_ids)>0:
        feed_dict = {input_img: img_batch, mode: 'test'}
        v_pred = sess.run([v], feed_dict)[0]
        for j,bb_id in enumerate(bb_ids):
            mask = mask_fm_bbox(feature_map_size=(19,19),bbox_norm=bboxes_norm[j,:],order='xyxy')
            if np.sum(mask)==0:
                continue
            img_vec = np.average(v_pred[j,:], weights = np.reshape(mask,[361]), axis=0)
            save_key = key+'/'+str(bb_id)
            with lmdb_env_kfrm.begin(write=True) as lmdb_txn:
                lmdb_txn.put(save_key.encode(), img_vec)
    if TEST_MODE:
        # [break] only for dockerization testing
        break   
    sys.stderr.write("Stored for keyframe {} / {} \r".format(i, len(dict_obj_kfrm)))
print(datetime.now())
len(missed_children_jpg)
len(missed_children_kfrm)

# Instance Features
# about 3 hours in total for Instance Features

#opening lmdb environment
lmdb_env_jpg = lmdb.open(out_path_jpg, map_size=int(1e11), lock=False)
lmdb_env_kfrm = lmdb.open(out_path_kfrm, map_size=int(1e11), lock=False)

#loading instance matching pretrained model
sess, graph = load_model(matching_model_path, config)
input_img = graph.get_tensor_by_name("input_img:0")
mode = graph.get_tensor_by_name("mode:0")
img_vec = graph.get_tensor_by_name("img_vec:0")

#about 0.5 hour
print(datetime.now())
missed_children_jpg = []
for i, key in enumerate(dict_obj_img):
    # Todo test
#     if 'HC0005KMS' not in key: #or 'HC0001H01' in key:
#         continue
    print(i,key)

    imgs,_ = fetch_img(key+'.jpg.ldcc', parent_dict, child_dict, path_dict, level = 'Child')
    if len(imgs)==0:
        missed_children_jpg.append(key)
        continue
    img_batch, bb_ids, bboxes_norm = batch_of_bbox(imgs[0], dict_obj_img, key,\
                                        score_thr=0, filter_out=False,img_size=(224,224))
    
    if len(bb_ids)>0:
        # Test for Corpping bug
        feed_dict = {input_img: img_batch, mode: 'test'}
        img_vec_pred = sess.run([img_vec], feed_dict)[0]
#     print('img_batch',img_batch)
#         print('img_batch len:',len(img_batch),np.shape(img_batch))
#         print('img_batch vec:',img_batch)
#         print(np.shape(img_vec_pred))
#         print('img_vec_pred',type(img_vec_pred),img_vec_pred)
        for j,bb_id in enumerate(bb_ids):
            save_key = key+'/'+str(bb_id)
            with lmdb_env_jpg.begin(write=True) as lmdb_txn:
                lmdb_txn.put(save_key.encode(), img_vec_pred[j,:])
#                 print(sum(img_vec_pred[j,:]))
    if TEST_MODE:
        # [break] only for dockerization testing
        break    
    sys.stderr.write("Stored for image {} / {} \r".format(i, len(dict_obj_img)))
print(datetime.now())

#about 3 hours
missed_children_kfrm = []
for i, key in enumerate(dict_obj_kfrm):
    imgs,_ = fetch_img(key+'.mp4.ldcc', parent_dict, child_dict, path_dict, level = 'Child')
    if len(imgs)==0:
        missed_children_kfrm.append(key)
        continue
    img_batch, bb_ids, bboxes_norm = batch_of_bbox(imgs[0], dict_obj_kfrm, key,\
                                      score_thr=0, filter_out=False,img_size=(224,224))
    if len(bb_ids)>0:
        feed_dict = {input_img: img_batch, mode: 'test'}
        img_vec_pred = sess.run([img_vec], feed_dict)[0]
        for j,bb_id in enumerate(bb_ids):
            save_key = key+'/'+str(bb_id)
            with lmdb_env_kfrm.begin(write=True) as lmdb_txn:
                lmdb_txn.put(save_key.encode(), img_vec_pred[j,:])
    if TEST_MODE:
        # [break] only for dockerization testing
        break  
    sys.stderr.write("Stored for keyframe {} / {} \r".format(i, len(dict_obj_kfrm)))
print(datetime.now())

print('Visual Feature Extraction Finished.')

