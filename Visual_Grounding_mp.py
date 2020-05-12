import tensorflow as tf
import numpy as np
import pickle
import os
import json
import sys
import re
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from textwrap import wrap
import utils
from rdflib import Graph
import copy
import cv2
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)

#############
# Visual Grounding and Instance Matching
# Columbia University
#############

# Specify data paths
corpus_path = '/root/LDC/'
working_path = '/root/shared/'
model_path = '/root/models/'


# Version Setting
# Set evaluation version as the prefix folder
version_folder = 'E/'
# Set run version as prefix and uiuc_run_folder
p_f_run = 'E1' # E5
uiuc_run_folder = 'RPI_TA1_E1/'


# Input: LDC2019E42 unpacked data, CU visual grounding and instance matching moodels, UIUC text mention results, CU object detection results
# Input Paths
# Source corpus data paths
print('Check Point: Raw Data corpus_path change',corpus_path)
parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
jpg_path = corpus_path + 'data/jpg/jpg/'
ltf_path = corpus_path + 'data/ltf/ltf/'


#UIUC text mention result paths
txt_mention_ttl_path = working_path + 'uiuc_ttl_results/' + version_folder + uiuc_run_folder # 1/7th May
pronouns_path = working_path + 'uiuc_asr_files/' + 'pronouns.txt'
video_asr_path = working_path + 'uiuc_asr_files/' + version_folder +'ltf_asr/'
video_map_path = working_path + 'uiuc_asr_files/' + version_folder +'map_asr/'
print('Check Point: text mentions path change',video_asr_path)


# CU object detection result paths
det_results_path_img = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34a.pkl' # jpg images
det_results_path_kfrm = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34b.pkl' # key frames
print('Check Point: Alireza path change:','\n',det_results_path_img,'\n', det_results_path_kfrm,'\n')


# Model Paths
# CU visual grounding and instance matching moodel paths
grounding_model_path = model_path + 'model_ELMo_PNASNET_VOA_norm'
matching_model_path = model_path + 'model_universal_no_recons_ins_only'

# Output: CU Visual grounding dict files for USC, CU visual grounding results
# Output Paths
# CU Visual grounding dict files for USC
entity2mention_dict_path = working_path + 'cu_grounding_dict_files/' + version_folder + 'entity2mention_dict_'+p_f_run+'.pickle'
id2mentions_dict_path = working_path + 'cu_grounding_dict_files/' + version_folder + 'id2mentions_dict_'+p_f_run+'.pickle'

# CU Visual grounding result paths
grounding_dict_path = working_path + 'cu_grounding_results/' + version_folder + 'grounding_dict_'+p_f_run+'.pickle'
grounding_log_path = working_path + 'cu_grounding_results/' + version_folder + 'log_grounding_'+p_f_run+'.txt'



def attn(e,v,e_bar,v_bar):
    ## Inputs: local and global cap and img features ##
    ## Output: Heatmap for each word, Global Heatmap, Attnded Vis features, Corr-vals
    #e: ?xTxD, v: ?xNxD, e_bar: ?xD, v_bar: ?xN2xD
    with tf.variable_scope('word_level_attn'):
        #Eq.8
        s = tf.einsum('bij,bkj->bik',e,v) #pair-wise ev^T: ?xTxN
        
        #s_bar = tf.nn.softmax(s, axis=1) #softmax on words

        #Eq.9
        #alpha = tf.nn.softmax(gamma_1*s_bar, axis=2) #softmax on regions
        alpha = s
        c = tf.einsum('bij,bjk->bik',alpha,v) #?xTxD attnded visual reps for each of T words
        #Eq.10
        c_norm = tf.nn.l2_normalize(c,axis=2)
        e_norm = tf.nn.l2_normalize(e,axis=2)
        R_i = tf.einsum('bik,bik->bi',c_norm,e_norm) #cosine for T (words,img_reps) for all pairs
        R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_2*R_i),axis=1),1/gamma_2)) #? corrs
        N0=int(np.sqrt(alpha.get_shape().as_list()[-1]))
        heatmap_w = tf.reshape(alpha,[tf.shape(alpha)[0],tf.shape(alpha)[1],N0,N0])
    
    with tf.variable_scope('sen_level_attn'):
        #Eq.8
        s_s = tf.einsum('bj,bkj->bk',e_bar,v_bar) #pair-wise e_bar*v_bar^T: ?xN2
        
        #Eq.9
        #alpha_s = tf.nn.softmax(gamma_1_s*s_s, axis=1) #softmax on regions
        alpha_s = s_s
        c_s = tf.einsum('bj,bjk->bk',alpha_s,v_bar) #?xD attnded visual reps for sen.
        #Eq.10
        c_s_norm = tf.nn.l2_normalize(c_s,axis=1)
        e_bar_norm = tf.nn.l2_normalize(e_bar,axis=1)
        R_s = tf.einsum('bk,bk->b',c_s_norm,e_bar_norm) #cosine for (sen,img_reps)
        N0_g=int(np.sqrt(alpha_s.get_shape().as_list()[-1]))
        heatmap_s = tf.reshape(alpha_s,[-1,N0_g,N0_g])
        
    return heatmap_w, heatmap_s, c_norm, c_s_norm, R_i, R, R_s
def top_dict(kv_dict, num = 2):
    k_list = list(kv_dict.keys())[:num]
    subdict =  {k: kv_dict[k] for k in k_list}
    print('\ndict len:', len(kv_dict))
    return subdict

print('Gathering keyframes info...')
#preparing dicts
id2dir_dict_kfrm = utils.create_dict_kfrm(kfrm_path, kfrm_msb, video_asr_path, video_map_path)
id2time_dict_kfrm = utils.id2time(kfrm_msb)

print('Creating parent-child dictionaries...')
#generate parent-child dictionaries
parent_dict, child_dict = utils.create_dict(parent_child_tab)

# for i in child_dict.keys():
#     if 'mp4' in i:
#         print(i,child_dict[i])
# print(top_dict(child_dict,5))

print('Creating path dictionary...')
#generate global id2path dictionary
#ltf
path_dict = utils.create_path_dict(ltf_path)
#jpg
path_dict.update(utils.create_path_dict(jpg_path))
#mp4
path_dict.update(utils.create_path_dict_kfrm(id2dir_dict_kfrm))


#loading a list of pronouns
with open(pronouns_path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    pronouns = set(lines)
    
print('Loading object detection results...')
#loading object detection results
with open(det_results_path_img, 'rb') as f:
    dict_obj = pickle.load(f)
    
with open(det_results_path_kfrm, 'rb') as f:
    dict_obj.update(pickle.load(f))
print('Loading done.')

print('Loading AIF crawl dictionaries...')
with open(entity2mention_dict_path, 'rb') as f:
    en2men = pickle.load(f)

with open(id2mentions_dict_path, 'rb') as f:
    id2men = pickle.load(f)

# Check Point: en2men and id2men maybe None
print('en2men len:', len(en2men.keys()))
print('id2men len:', len(id2men.keys()))
print(datetime.now())

# maybe changed: turtle_files comes from UIUC, .ttl maybe changed 
# about 1 hour 15 min
print('Creating id-sentence-entity dictionary...')
print(datetime.now())
en2men = {}
turtle_files = os.listdir(txt_mention_ttl_path)
import ltf_util as uiuc_ltf_util
ltf_util = uiuc_ltf_util.LTF_util(ltf_path)
for i,file in enumerate(turtle_files):
    # [break] only for dockerization testing
    if i>15:
        break
    if ".ttl" not in file:
        continue
#     print(i,file) # HC000VULD.ttl
#     # Done: text mention filter checking
#     if (file not in ['HC000VULD.ttl','HC000VULD.ttl']):
#         continue
#     else:
#         print('file',file)
    turtle_path = os.path.join(txt_mention_ttl_path, file)
    #loading turtle content
    turtle_content = open(turtle_path).read()
    g = Graph().parse(data=turtle_content, format='n3')
    # Check Point: en2men maybe changed
    #find 'aida:Entity', then find 'aida:justifiedBy'
#     print('g',len(en2men.keys()))
    en2men.update(utils.get_entity2mention(g,ltf_util)) 
    sys.stdout.write('File {}/{} \r'.format(i+1,len(turtle_files)))                
    sys.stdout.flush()
    
## mention type: mention, nominal_mention, pronominal_mention, normalized_mention
## Named entity types: PER PRG GPE LOC ORG WEA VEH
## Filler types: BAL COM CRM LAW MON RES SID TTL VAL

#short_to_long_map = {
#    'BAL': 'Ballot',
#    'COM': 'Commodity',
#    'CRM': 'Crime',
#    'FAC': 'Facility',
#    'GPE': 'GeopoliticalEntity',
#    'LAW': 'Law',
#    'LOC': 'Location',
#    'MON': 'Money',
#    'ORG': 'Organization',
#    'PER': 'Person',
#    'RES': 'Result',
#    'SID': 'Sides',
#    'TTL': 'Title',
#    'VAL': 'Value',
#    'VEH': 'Vehicle',
#    'WEA': 'Weapon',
# }
# filter_out=['pronominal_mention','GeopoliticalEntity','Organization',
#             'Location','Money','NumericalValue','Time','URL','Sides',
#             'Sentence','Results','Law','Crime','Ballot','Age']
print('Creating dictionary and filtering out for grounding...')
filter_out=[ 'pronominal_mention', 'GPE', 'ORG',
       'LOC','MON', 'NumericalValue', 'VAL', 'Time', 'URL', 'COM', 'SID',
        'Sentence', 'RES', 'LAW', 'CRM', 'BAL', 'Age', 'TTL'] 

id2men = utils.create_entity_dict(en2men, path_dict, caption_alignment_path=[], filter_out=filter_out)
# uncomment
print('Processing graphs done.')
open(grounding_log_path, 'w').write('Processing graphs done.\n')
with open(entity2mention_dict_path, 'wb') as f:
    pickle.dump(en2men,f, protocol=pickle.HIGHEST_PROTOCOL)

with open(id2mentions_dict_path, 'wb') as f:
    pickle.dump(id2men,f, protocol=pickle.HIGHEST_PROTOCOL)
print(datetime.now())

#loading grounding pretrained model
print('Loading grounding pretrained model...')
gamma_1 = 10.0
gamma_1_s = 10.0
gamma_2 = 5.0
gamma_3 = 10.0
#model_path = '../model_CNN_avg'
sess, graph = utils.load_model(grounding_model_path,config)
input_img = graph.get_tensor_by_name("input_img:0")
text_batch = graph.get_tensor_by_name("text_input:0")
mode = graph.get_tensor_by_name("mode:0")
v = graph.get_tensor_by_name("image_local_features:0")
v_bar = graph.get_tensor_by_name("image_global_features:0")
w_embedding = graph.get_tensor_by_name("w_embedding:0")
sen_embedding = graph.get_tensor_by_name("sen_embedding:0")
heatmap_w, heatmap_s, c, c_s, R_i, R, R_s = attn(w_embedding,v,sen_embedding,v_bar)
print('Loading done.')
print(datetime.now())
##Grounding

# about 6-8 hours in total for Grounding
#grounding, entity level
print(datetime.now())
# Todo: adjust parameters
en_score_thr = .5 #.9
sen_score_thr = .6 #.6
suffix_tmp = '_' + p_f_run + '_5-6'

en_to_img_dict = {}
img_to_feat_dict = {}
img_cnt_dict = {}
for k,key in enumerate(id2men):
    # [break] only for dockerization testing
    if k > 200:
        break
    
    sys.stdout.write('Key {}/{} \r'.format(k,len(id2men)))                
    sys.stdout.flush()
    open(grounding_log_path, 'w').write('Key {}/{} \r'.format(k,len(id2men)))
    #get all sens of the given leaf doc, and all images of its root doc
    imgs,ids = utils.fetch_img(key, parent_dict, child_dict, path_dict, level = 'Parent')
    sens = list(id2men[key].keys())
    if len(sens)==0:
        continue
    #generate all img-sen pairs and their corresponding grounding params
    text_flag = len(imgs)==0
    img_batch, sen_batch, img_info_batch = utils.img_cap_batch_gen(imgs,sens,ids,key,path_dict,id2time_dict_kfrm)
    placeholders = [input_img, text_batch, mode]
    tensor_list = [w_embedding, c, heatmap_w, R_i, R, R_s]
    inputs = [img_batch,sen_batch]
    EN_embd, IMG_embd, EN_heat, EN_score, avg_EN_score, sen_score = utils.batch_split_run(sess,tensor_list,placeholders,inputs,text_flag,b_size_thr=30)
    for i,sen in enumerate(sen_batch):
        #if sen_score[i] < sen_score_thr:
        #    continue
        for entity in id2men[key][sen]:
            en_name = id2men[key][sen][entity]['name']
            if en_name.lower() in pronouns:
                continue
            for mention in id2men[key][sen][entity]['mentions']:
                men_dict = id2men[key][sen][entity]['mentions'][mention]
                if men_dict['idx']==[]:
                    continue
            
                #if np.mean(EN_score[i,men_dict['idx']]) < en_score_thr:
                #    continue
                men_name = men_dict['name']
                if men_name.lower() in pronouns:
                    continue
                en_type = id2men[key][sen][entity]['type_rdf']
                source_type = id2men[key][sen][entity]['source_type']
                language = id2men[key][sen][entity]['language']
                             
                #men_embd = np.average(EN_embd[i,men_dict['idx'],:], weights = EN_score[i,men_dict['idx']], axis=0)
                men_embd = np.mean(EN_embd[i,men_dict['idx'],:], axis=0) #for now, just averaging word embdngs
                orig_img_id = img_info_batch[i][0]
                
                if text_flag:
                    grnd = {}
                elif sen_score[i] < sen_score_thr or np.mean(EN_score[i,men_dict['idx']]) < en_score_thr:
                    grnd = {}
                else:
                    heatmap = np.average(EN_heat[i,men_dict['idx'],:], weights = EN_score[i,men_dict['idx']], axis=0)
                    img_embd = np.average(IMG_embd[i,men_dict['idx'],:], weights = EN_score[i,men_dict['idx']], axis=0)
                    orig_img_shape = img_info_batch[i][1]
                    bbox_dict = utils.heat2bbox(heatmap,orig_img_shape)
                    bbox, bbox_norm, score = utils.filter_bbox(bbox_dict=bbox_dict, order='xyxy')
                    grnd = {'bbox': bbox, 'bbox_norm': bbox_norm, 'bbox_score': score,
                            'heatmap': heatmap, 'sen-img-score': sen_score[i],
                            'men-img-score': EN_score[i,men_dict['idx']],
                            'grounding_features': img_embd}
            
                men_dict = {mention: {'grounding': {orig_img_id: grnd},
                                      'textual_features': men_embd,
                                      'name': men_name,
                                      'sentence': sen}}
                en_dict = {'textual_features': np.zeros((men_embd.shape),dtype='float32'),
                           'name': en_name,
                           'type_rdf': en_type,
                           'mentions': men_dict,
                           'source_type': source_type,
                           'language': language}
                if entity not in en_to_img_dict:
                    en_to_img_dict[entity] = en_dict
                elif mention not in en_to_img_dict[entity]['mentions']:
                    en_to_img_dict[entity]['mentions'].update(men_dict)
                elif orig_img_id not in en_to_img_dict[entity]['mentions'][mention]['grounding']:
                    en_to_img_dict[entity]['mentions'][mention]['grounding'].update(men_dict[mention]['grounding'])
sess.close()


print('Grounding done.\nAveraging textual features to shape entity features...')
#averaging mention features to shape entity features
for entity in en_to_img_dict:
    for mention in en_to_img_dict[entity]['mentions']:
        en_to_img_dict[entity]['textual_features'] += en_to_img_dict[entity]['mentions'][mention]['textual_features']
    en_to_img_dict[entity]['textual_features'] /= len(en_to_img_dict[entity]['mentions'])

print('Averaging done.\nClustering mention bboxes to shape entity bbox...')
#clustering mention bboxes for each entity
for k,entity in enumerate(en_to_img_dict):
    men_dict = copy.deepcopy(en_to_img_dict[entity]['mentions'])
    # Done for USC: print dict of en_to_img_dict, and output
    en_to_img_dict[entity]['grounding'] = utils.men2en_grnd(men_dict,dict_obj)
    sys.stdout.write('Key {}/{} \r'.format(k,len(en_to_img_dict)))                
    sys.stdout.flush() 

# saving as a temporal file
grounding_dict_path_tmp = 'grounding_dict_path_tmp' + '.' + suffix_tmp +'.tmp'
with open(grounding_dict_path_tmp, 'wb') as f:
    pickle.dump(en_to_img_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
print('Clustering done.', 'grounding_dict.tmp len:',len(en_to_img_dict))
open(grounding_log_path, 'w').write('\nGrounding Stage done.')
print(datetime.now())
##Instance Features

#loading instance matching pretrained model
sess_ins, graph_ins = utils.load_model(matching_model_path,config)
input_img_ins = graph_ins.get_tensor_by_name("input_img:0")
mode_ins = graph_ins.get_tensor_by_name("mode:0")
img_vec_ins = graph_ins.get_tensor_by_name("img_vec:0")
def batch_of_bbox(img, bbox_norm):
    img_batch = np.empty((len(bbox_norm),224,224,3), dtype='float32')
    for i,bbox in enumerate(bbox_norm):
        roi = utils.crop_resize_im(img, bbox, (224,224), order='xyxy')
        img_batch[i,:,:,:] = roi
    return img_batch

# about 2 hours
# saving as a temporal file
# suffix = suffix_tmp #'_8-6'
with open(grounding_dict_path_tmp, 'rb') as f:
    en_to_img_dict = pickle.load(f)
print(len(en_to_img_dict))

missed_children_ins = []
maxbatch = 0
for i,en in enumerate(en_to_img_dict):
    
    try:
#         if i in [11996,12085]:
#             continue
        for img_id in en_to_img_dict[en]['grounding']:
            imgs,_ = utils.fetch_img(img_id, parent_dict, child_dict, path_dict, level = 'Child')
            if len(imgs)==0:
                missed_children_ins.append(key)
                continue
            bbox_norm = en_to_img_dict[en]['grounding'][img_id]['bbox_norm']
            img_batch = batch_of_bbox(imgs[0], bbox_norm)
            feed_dict = {input_img_ins: img_batch, mode_ins: 'test'}
            img_vec_pred = sess_ins.run([img_vec_ins], feed_dict)[0]
            en_to_img_dict[en]['grounding'][img_id]['instance_features']=[]
            for j in range(len(bbox_norm)):
                en_to_img_dict[en]['grounding'][img_id]['instance_features'].append(img_vec_pred[j,:])
        sys.stderr.write("Stored for image {} / {} \r".format(i, len(en_to_img_dict)))
        sys.stdout.flush() 
        # [break] only for dockerization testing
        break
    except ValueError:
        print("Oops!",i,ValueError) 
sess_ins.close()

with open(grounding_dict_path, 'wb') as f:
    pickle.dump(en_to_img_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
    
open(grounding_log_path, 'w').write('\nVisual Grounding Finished.')
print(datetime.now())

print('Visual Grounding and Instance Matching Finished.')

