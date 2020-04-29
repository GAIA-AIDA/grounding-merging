print("begin smoke test")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # GPU_ID
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)
working_path = '/root/data/'
corpus_path = '/root/dryrun/'
kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
print('Loading grounding pretrained model...')
model_path = working_path+ 'models/model_ELMo_PNASNET_VOA_norm'
sess, graph = load_model(model_path,config)
input_img = graph.get_tensor_by_name("input_img:0")
mode = graph.get_tensor_by_name("mode:0")
v = graph.get_tensor_by_name("image_local_features:0")
v_bar = graph.get_tensor_by_name("image_global_features:0")
print('Loading done.')
parent_dict, child_dict = create_dict(parent_child_tab)
print("smoke test complete")
