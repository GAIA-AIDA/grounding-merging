#!/usr/bin/env python
# coding: utf-8

# In[3]:


import lmdb
import sys
import numpy as np
from io import BytesIO
import numpy as np
#sys.path.append("/dvmm-filer2/projects/AIDA/alireza/tools/AIDA-Interchange-Format/python/aida_interchange")
from rdflib import URIRef
from rdflib.namespace import ClosedNamespace
from collections import defaultdict
import sys
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm as std_tqdm
import time
from sklearn.cluster import DBSCAN

import multiprocessing

start = time.time()
print("hello")


from sklearn.preprocessing import normalize
import cv2
from aida_interchange import aifutils
#import dbscan
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)

parent_file = sys.argv[1]
az_obj_graph = sys.argv[2]
az_obj_jpg = sys.argv[3]
az_obj_kf = sys.argv[4]
ins_img_path = sys.argv[5]
ins_kf_path = sys.argv[6]
ere_link = sys.argv[7]
#video_map = sys.argv[8]
dataName = sys.argv[8]


Path(dataName).mkdir(parents=True, exist_ok=True)
#hmdb_file = 
# In[2]:


child = defaultdict(list)
parent = {}
file5= open(parent_file)
i = 0
for line in file5:
    i+=1
    if i ==1:
        continue
    data = line.split('\t')
    child[data[2]].append(data[3])
    parent[data[3]] = data[2]

#print(parent)
# In[3]:






with open(az_obj_graph, 'rb') as handle:
        (kb_dict, entity_dict, event_dict) = pickle.load(handle)

        
entity_dic2 = defaultdict(list)
for x,y in entity_dict.items():
    data = x.split('/')
    #print data
    entity_dic2[data[-2]].append(data[-1])
#print('entity_dic2: ',entity_dic2)
#for x,y in entity_dic2.items():
#    print x,y[0]
#    break


# In[4]:


with open(az_obj_jpg, 'rb') as handle:
        OD_result = pickle.load(handle)


# In[5]:
#""" 

with open(az_obj_kf, 'rb') as handle:
    ODF_result = pickle.load(handle)


# In[6]: 
#"""



parentDic = defaultdict(list)
#import caffe
#ins_img_path = "results/instance1"

ins_img_env = lmdb.open(ins_img_path, map_size=1e11, readonly=True, lock=False)
#print(ins_img_env)
ins_img_txn = ins_img_env.begin(write=False)
lmdb_cursor = ins_img_txn.cursor()
#datum = caffe.proto.caffe_pb2.Datum()





#"""
for key, value in lmdb_cursor:
    #datum.ParseFromString(value)
    #data = caffe.io.datum_to_array(datum)
    value = np.frombuffer(value, dtype='float32').tolist()
    #print(str(key))
    key = key.decode()
    data = str(key).split('/') 
    file_id = str(data[0])
    num = data[1]
    #print(file_id,num)

    if num in entity_dic2[file_id]:
        parentDic[parent[data[0]]].append((key,value))

#print(parentDic) 
#""" 


#"""
ins_kf_env = lmdb.open(ins_kf_path, map_size=1e11, readonly=True, lock=False)
#print(ins_img_env)
ins_kf_txn = ins_kf_env.begin(write=False)
lmdb_cursor2 = ins_kf_txn.cursor()
#datum = caffe.proto.caffe_pb2.Datum()




#video

for key, value in lmdb_cursor2:
    #datum.ParseFromString(value)
    #data = caffe.io.datum_to_array(datum)
    value = np.frombuffer(value, dtype='float32').tolist()
    #print(str(key))
    key = key.decode()
    data = str(key).split('/')  
    file_id = str(data[0])
    num = data[1]
    #print(file_id,num)
    idNoNum = file_id.split('_')[0]
    if num in entity_dic2[file_id]:
        parentDic[parent[idNoNum]].append((key,value))

#print(parentDic) 
#"""
# In[7]:


"""

# In[8]:


# In[9]:


file1 = open('results/frame_child_e.txt')
videoDic = {}
for line in file1:
    data = line.split()
    videoDic[data[1]] = data[0]


# In[10]:
"""

file1 = open(ere_link)
ere_type = {}
for line in file1:
    data = line.split(',')
    #print data
    
    ere_type[data[0]] = data[1]


# In[11]:




images=0
i=0

typeList = ['Weapon','Vehicle','Person','Facility']

count = 0
for x, y in tqdm(parentDic.items()):

    #print x
    count+=1
    #if count < 40:
    #   continue
    
    g = aifutils.make_graph()
    #g = kb_dict[parent]
    cu_pref = 'http://www.columbia.edu/AIDA/DVMM/'
    sys_instance_matching = aifutils.make_system_with_uri(g, cu_pref+'Systems/Instance-Matching/ResNet152')
    
    #entityList = []
    entityList = defaultdict(list)
    arrayList = defaultdict(list)
    keyList = defaultdict(list)
    #bb_list = []
    first = 1
    detected = 0
    for i in range(len(y)):
        (key, feature) = y[i]
        #print key
        detected+=1
        #print detected
        if '_' in key:
            #print key
            
            eid = "http://www.columbia.edu/AIDA/DVMM/Entities/ObjectDetection/RUN00010/Keyframe/"+key
            #if eid in entity_dict.keys():
                #continue
            data = key.split('/')
            #print ere_type[ODF_result[data[0]][int(data[1])]['label']]
            if ODF_result[data[0]][int(data[1])]['label'] not in ere_type.keys():
                continue
            if ere_type[ODF_result[data[0]][int(data[1])]['label']] not in typeList:
                continue
            
            data2 = data[0].split('_')

            arrayList[ere_type[ODF_result[data[0]][int(data[1])]['label']]].append(feature)
            entityList[ere_type[ODF_result[data[0]][int(data[1])]['label']]].append(entity_dict[eid])

        else:
            data = key.split('/')
            #print ere_type
            if OD_result[data[0]][int(data[1])]['label'] not in ere_type.keys():
                continue
            if ere_type[OD_result[data[0]][int(data[1])]['label']] not in typeList:
                continue
            eid = "http://www.columbia.edu/AIDA/DVMM/Entities/ObjectDetection/RUN00010/JPG/"+key
            #if eid in entity_dict.keys():
                #continue
            

            #bb_list.append(OD_result[data[0]][int(data[1])]['bbox'])
            entityList[ere_type[str(OD_result[data[0]][int(data[1])]['label'])]].append(entity_dict[eid])
            arrayList[ere_type[str(OD_result[data[0]][int(data[1])]['label'])]].append(feature)

    matches = 0
    for a, b in arrayList.items():

        new_array = np.array(arrayList[a])

        normed_matrix = normalize(new_array, axis=1, norm='l2')
        normed_matrix_T = np.transpose(normed_matrix)
        n_array =  np.matmul(normed_matrix,normed_matrix_T )

        db = DBSCAN(eps=0.5, min_samples=2, metric='cosine', n_jobs = 1).fit(new_array)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
       
        if len(new_array)>1:

            clusterNameDic = {}
            #print n_clusters_

            firstMem = [0 for i in range(n_clusters_)]
            for j in range(len(labels)):
                if labels[j] == -1:
                    continue
                matches+=1
                #print j
                #print labels[j]
                score = 1
                #print firstMem
                if firstMem[labels[j]] == 0:
                    firstMem[labels[j]] = 1
                    clusterNameDic[labels[j]] = aifutils.make_cluster_with_prototype(g, \
                        "http://www.columbia.edu/AIDA/DVMM/Clusters/ObjectCoreference/RUN00010/"+ \
                         a+'/'+str(labels[j]),entityList[a][j], sys_instance_matching)
                    #print entityList[a][j]
                else:
                    aifutils.mark_as_possible_cluster_member(g,\
                     entityList[a][j],clusterNameDic[labels[j]], score, sys_instance_matching)
                    #print entityList[a][j]

    #dataName = 'post_e_i_c'
    #parent = x
    with open(dataName+'/'+x+'.ttl', 'w') as fout:
        serialization = BytesIO()
        # need .buffer because serialize will write bytes, not str
        g.serialize(destination=serialization, format='turtle')
        fout.write(serialization.getvalue().decode('utf-8'))

end = time.time()
print(end - start)
