from rdflib import Graph, plugin, URIRef, Literal, BNode, RDF
from rdflib.serializer import Serializer
from rdflib.namespace import SKOS
import rdflib
import re
import sys
import os
import json
import utils
import lmdb
import pickle
import numpy as np
from io import BytesIO
from datetime import datetime

# keep changing
sys.path.append("/AIDA/AIDA-Interchange-Format/python/")
import aida_interchange.aifutils as aifutils
from aida_interchange.bounding_box import Bounding_Box


#############
# Graph Merging
# Columbia University
#############

# Specify data paths
corpus_path = '/root/LDC/'
working_path = '/root/shared/'
output_path = '/root/output/'
TEST_MODE = True
if TEST_MODE:
    print("TEST_MODE is open")


# Version Setting
# Set evaluation version as the prefix folder
version_folder = '' # 'E/' could be ignored if there is no version management
# Set run version as prefix
p_f_run = '' # '_E1' could be ignored if there is no version management
uiuc_run_folder = 'RPI_ttl/'

# Set the number of multiple processes
processes_num = 32

# Input: LDC unpacked data, CU visual grounding and all detection results, UIUC text mention results, USC grounding results
# Input Paths
# Source corpus data paths
print('Check Point: LDC raw data change',corpus_path)
parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
parent_dict, child_dict = utils.create_dict(parent_child_tab)


# CU visual grounding feature paths
sem_img_path = working_path + 'cu_grounding_matching_features/' + 'semantic_features_jpg.lmdb'
sem_kfrm_path = working_path + 'cu_grounding_matching_features/' + 'semantic_features_keyframe.lmdb'

# CU instance matching feature paths
ins_img_path = working_path + 'cu_grounding_matching_features/' + 'instance_features_jpg.lmdb'
ins_kfrm_path = working_path + 'cu_grounding_matching_features/' + 'instance_features_keyframe.lmdb'

# CU visual grounding result path
grounding_dict_path = working_path + 'cu_grounding_results/' + version_folder + 'grounding_dict'+p_f_run+'.pickle'
print('Check Point: version change',grounding_dict_path)
grounding_dict = pickle.load(open(grounding_dict_path,'rb'))
# def top_dict(kv_dict, num = 2):
#     k_list = list(kv_dict.keys())[:num]
#     subdict =  {k: kv_dict[k] for k in k_list}
#     print('\n length of CU grounding dict:', len(kv_dict))
#     return subdict
# print('grounding_dict', top_dict(grounding_dict),'\n')

# CU temporal ttl results
cu_ttl_tmp_path = working_path + 'cu_ttl_tmp/'
cu_ttl_path = cu_ttl_tmp_path + version_folder + 'm18' + p_f_run + '/'
cu_ttl_ins_path = cu_ttl_tmp_path + version_folder + 'm18_i_c' + p_f_run + '/'
print('Check Point: cu_ttl_tmp_path change',cu_ttl_path,cu_ttl_ins_path)


#UIUC text mention result paths
txt_mention_ttl_path = working_path + 'uiuc_ttl_results/' + version_folder + uiuc_run_folder 
print('Check Point: text mention ttl path change',txt_mention_ttl_path)


# USC visual grounding result path for merging
usc_dict_path = working_path + 'usc_grounding_dict/' + version_folder + 'uscvision_grounding_output_cu_format' + p_f_run + '.pickle' 


# Output: CU graph merging result
# Output Paths
# CU graph merging result path
merged_graph_path = output_path + 'cu_graph_merging_ttl/' + version_folder + 'merged_ttl'+ p_f_run + '/'


# Merge USC grounding results
def check_usc_grounding_dict(usc_grounding_dict, child_dict):
    # test
#     key = 'http://www.isi.edu/gaia/entities/b7ec53ef-3021-4eda-9429-7c87ccd1dc0d'
#     if 'usc_vision' in usc_grounding_dict[key]['grounding'].values():
#         print('find it')
#     print('arrrr', usc_grounding_dict[key]['grounding']['system'])
    
    # Check Point: grounding_dict printing
    top,i = 3,0
    print('usc_grounded_examples')
    for en in usc_grounding_dict.keys():
        for img_id in usc_grounding_dict[en]['grounding']:
            if img_id  =='system':
                continue
            n_b = len(usc_grounding_dict[en]['grounding'][img_id]['bbox'])
            if n_b>0:
                img_dict = usc_grounding_dict[en]['grounding'][img_id]
                if i < top: 
                    i+=1
                    print(en)
                    print(child_dict[img_id], img_id, usc_grounding_dict[en]['grounding'][img_id])
                    print(sum(sum(img_dict['grounding_features'])),sum(sum(img_dict['instance_features'])))
                    print('\n')

# cu_grndg_ent_pref = 'http://www.columbia.edu/AIDA/DVMM/Entities/GroundingBox/'+merge_version+'/'
# cu_grndg_type_pref = 'http://www.columbia.edu/AIDA/DVMM/TypeAssertions/GroundingBox/'+merge_version+'/'
# cu_grndg_clstr_img_pref = 'http://www.columbia.edu/AIDA/DVMM/Clusters/BoxOverlap/'+merge_version+'/'
# cu_grndg_clstr_txt_pref = 'http://www.columbia.edu/AIDA/DVMM/Clusters/Grounding/'+merge_version+'/'
        
    
def merge_usc(usc_dict_path, grounding_dict, child_dict):    
    grounded = dict((k,v) for k,v in grounding_dict.items() if len(v['grounding']) > 0)
    # print('grounded', top_dict(grounded),'\n')
    usc_grounding_dict = pickle.load(open(usc_dict_path,'rb'))
    check_usc_grounding_dict(usc_grounding_dict, child_dict)
    
    print('intersection num', len(list(set(usc_grounding_dict.keys()).intersection(set(grounded.keys())))))
    print('difference (num added from usc)', len(list(set(usc_grounding_dict.keys()).difference(set(grounded.keys())))))
    conflict_set = list(set(usc_grounding_dict.keys()).intersection(set(grounded.keys())))
    print('conflict entities:',conflict_set)
    for k in conflict_set:
        del usc_grounding_dict[k]

    grounding_dict.update(usc_grounding_dict) # if key is same, the new pair will be appended  
#     print('usc_grounding_dict', top_dict(usc_grounding_dict),'\n')
#     print('merged_grounding_dict', top_dict(merged_grounding_dict),'\n')
    print('usc_grounded_entity in merged_grounding_dict',grounding_dict[list(usc_grounding_dict.keys())[0]])
    return grounding_dict

# the path always changed 
add_usc_result = True
if add_usc_result:
    grounding_dict = merge_usc(usc_dict_path, grounding_dict, child_dict)
# Done: the [grounding_dict] will be updated by merged_grounding_dict
usc_pref = 'http://www.usc.edu/AIDA/IRIS/'
# it should returns 
print(datetime.now())



txt_mention_ttl_list = set([f.split('.')[0] for f in os.listdir(txt_mention_ttl_path)])
cu_ttl_list = set([f.split('.')[0] for f in os.listdir(cu_ttl_path)])
cu_ttl_ins_list = set([f.split('.')[0] for f in os.listdir(cu_ttl_ins_path)])

cu_pref = 'http://www.columbia.edu/AIDA/DVMM/'
cu_objdet_pref = 'http://www.columbia.edu/AIDA/DVMM/Entities/ObjectDetection/'

# RUN00004 .. # May 8, 2019 for D2
#'RUN00005' # May 19, 2019 for D2
merge_version = 'RUN00006' # June 21, 2019 for D3
cu_grndg_ent_pref = 'http://www.columbia.edu/AIDA/DVMM/Entities/GroundingBox/'+merge_version+'/'
cu_grndg_type_pref = 'http://www.columbia.edu/AIDA/DVMM/TypeAssertions/GroundingBox/'+merge_version+'/'
cu_grndg_clstr_img_pref = 'http://www.columbia.edu/AIDA/DVMM/Clusters/BoxOverlap/'+merge_version+'/'
cu_grndg_clstr_txt_pref = 'http://www.columbia.edu/AIDA/DVMM/Clusters/Grounding/'+merge_version+'/'



#'RUN00002' # May 19, 2019
usc_merge_version = 'RUN00003' # June 22, 2019
usc_grndg_ent_pref = 'http://www.usc.edu/AIDA/IRIS/Entities/GroundingBox/'+usc_merge_version+'/'
usc_grndg_type_pref = 'http://www.usc.edu/AIDA/IRIS/TypeAssertions/GroundingBox/'+usc_merge_version+'/'
usc_grndg_clstr_txt_pref = 'http://www.usc.edu/AIDA/IRIS/Clusters/Grounding/'+merge_version+'/'
print('Check Point: USC gournding path change',usc_dict_path)



rpi_entity_pref = 'http://www.isi.edu/gaia/entities/'
gaia_prefix = 'http://www.isi.edu/gaia'
# nist_ont_pref = '.../SM-KBP/2018/ontologies/InterchangeOntology#'
nist_ont_pref = 'https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#'
justified_by_ = URIRef(nist_ont_pref+'justifiedBy')
entity_ = URIRef(nist_ont_pref+'Entity')
sys_ = URIRef(nist_ont_pref+'system')
hasName_ = URIRef(nist_ont_pref+'hasName')


###loading visual features

#load instance features
ins_img_env = lmdb.open(ins_img_path, map_size=1e11, readonly=True, lock=False)
ins_img_txn = ins_img_env.begin(write=False)
ins_kfrm_env = lmdb.open(ins_kfrm_path, map_size=1e11, readonly=True, lock=False)
ins_kfrm_txn = ins_kfrm_env.begin(write=False)

#load semantic features
sem_img_env = lmdb.open(sem_img_path, map_size=1e11, readonly=True, lock=False)
sem_img_txn = sem_img_env.begin(write=False)
sem_kfrm_env = lmdb.open(sem_kfrm_path, map_size=1e11, readonly=True, lock=False)
sem_kfrm_txn = sem_kfrm_env.begin(write=False)

#undergoing functions
def get_features(key,dtype,ftype):
#     print("get_features for ",ftype)
    if ftype=='instance':
        if dtype=='jpg':
            txn = ins_img_txn
        elif dtype=='keyframe':
            txn = ins_kfrm_txn
        else:
            return []
    elif ftype=='semantic':
        if dtype=='jpg':
            txn = sem_img_txn
        elif dtype=='keyframe':
            txn = sem_kfrm_txn
        else:
            return []
    else:
        return
    
    value = txn.get(key.encode('utf-8'))
    if value!=None:
        return np.frombuffer(value, dtype='float32').tolist()
    return []
##Main Body

# for Main Body
import multiprocessing as mp
# maybe changed when different runs
print(datetime.now())
# graph merging


#graph integration and modification
#entity level

def transferAIF(p_id):
#for k,p_id in enumerate(parent_dict):

    # Todo test
#     if (k<8):
#         continue
#     print('k',k,p_id)
    g = Graph()
    
    #load rpi graph if exists
    if p_id in txt_mention_ttl_list:
        turtle_path = os.path.join(txt_mention_ttl_path, p_id+'.ttl')
        turtle_content = open(turtle_path).read()
        g.parse(data=turtle_content, format='n3')
    
    #load and merge cu graph if exists
    if p_id in cu_ttl_list:
        turtle_path = os.path.join(cu_ttl_path, p_id+'.ttl')
        turtle_content = open(turtle_path).read()
        g.parse(data=turtle_content, format='n3')
    
    #load and merge cu graph for instance matching if exists
    if p_id in cu_ttl_ins_list:
        turtle_path = os.path.join(cu_ttl_ins_path, p_id+'.ttl')
        turtle_content = open(turtle_path).read()
        g.parse(data=turtle_content, format='n3')
    
    sys_instance_matching = aifutils.make_system_with_uri(g, cu_pref+'Systems/Instance-Matching/ResNet152')
    sys_grounding = aifutils.make_system_with_uri(g, cu_pref+'Systems/Grounding/ELMo-PNASNET')
    usc_sys_grounding = aifutils.make_system_with_uri(g, usc_pref + 'Systems/ZSGrounder')
    
    #find vision and text entities
    sbj_all = set(g.subjects())
    img_entities = {}
    keyframe_entities = {}
    ltf_entities = {}
    for sbj in sbj_all:
        sbj_name = sbj.toPython()
        if cu_objdet_pref in sbj_name:
            if sbj.__class__ == rdflib.term.URIRef:
                if 'JPG' in sbj_name:
                    img_id = '/'.join(sbj_name.split('/')[-2:])
                    img_entities[img_id] = sbj
                elif 'Keyframe' in sbj_name:
                    kfrm_id = '/'.join(sbj_name.split('/')[-2:])
                    keyframe_entities[kfrm_id] = sbj
        elif rpi_entity_pref in sbj_name:
            if sbj.__class__ == rdflib.term.URIRef and rpi_entity_pref in sbj_name:
                ltf_entities[sbj_name] = sbj
    
    # Done
#     if p_id in []:#['IC0011TIB']:
#         continue
#     print('k',k,p_id)
#     if (g==None):
#         print('p_id', k, p_id)
    
    ##adding private data to entities for cu grounding
    #images
    for key in img_entities:
        dtype='jpg'
        #instance features
        ftype='instance'
        data_instance = get_features(key,dtype,ftype)
        
        #semantic features
        ftype='semantic'
        data_semantic = get_features(key,dtype,ftype)
        
        #aggregation
        j_d_i = json.dumps({'columbia_vector_instance_v1.0': data_instance})
        j_d_s = json.dumps({'columbia_vector_grounding_v1.0': data_semantic})
        entity = img_entities[key]
        aifutils.mark_private_data(g, entity, j_d_i, sys_instance_matching)
        aifutils.mark_private_data(g, entity, j_d_s, sys_grounding)
    
    #keyframes  
    for key in keyframe_entities:
        dtype='keyframe'
        #instance features
        ftype='instance'
        data_instance = get_features(key,dtype,ftype)

        #semantic features
        ftype='semantic'
        data_semantic = get_features(key,dtype,ftype)

        #aggregation
        j_d_i = json.dumps({'columbia_vector_instance_v1.0': data_instance})
        j_d_s = json.dumps({'columbia_vector_grounding_v1.0': data_semantic})
        entity = keyframe_entities[key]
        aifutils.mark_private_data(g, entity, j_d_i, sys_instance_matching)
        aifutils.mark_private_data(g, entity, j_d_s, sys_grounding)
        
        

    cnt_img = {}
    cnt_boxO = {}
    cnt_ltf = {}
    #add text features, grounding, linking
    for key in ltf_entities:        
        if key not in grounding_dict:
            continue
        entity_name = None
        USC_GROUNDING = 'usc_vision' in grounding_dict[key]['grounding'].values()
        if not USC_GROUNDING:
#             print('our grounding')
            #text features
            j_d_t = json.dumps({'columbia_vector_text_v1.0': grounding_dict[key]['textual_features'].tolist()})
            entity_ltf = ltf_entities[key]
            aifutils.mark_private_data(g, entity_ltf, j_d_t, sys_grounding)
        
            #type and name of entity to be linked
            type_rdf = grounding_dict[key]['type_rdf']
            entity_name = grounding_dict[key]['name']
            grndg_file_type = grounding_dict[key]['source_type']
            
        if entity_name is None:
            continue

        
        #keep track of entities with same names for avoiding clustering overlap
        if entity_name in cnt_ltf:
            cnt_ltf[entity_name] += 1
        else:
            cnt_ltf[entity_name] = 1
        
        clstr_prot_flag = False #cluster obj for entity_ltf not created yet
        #adding grounding bboxes as new entities
        for img_id in grounding_dict[key]['grounding']:
            if img_id == 'system':
                continue
            grnd = grounding_dict[key]['grounding'][img_id]
            for ii,bbox in enumerate(grnd['bbox']):
                
                if img_id in cnt_img: #to keep track of cnt of bbox of same image
                    cnt_img[img_id] += 1
                else:
                    cnt_img[img_id] = 1
                #add grounding bbox as entity
                score = grnd['bbox_score'][ii]
                if not USC_GROUNDING:
                    type_eid = cu_grndg_type_pref+f"{grndg_file_type}/{img_id.split('.')[0]}/{cnt_img[img_id]}/ERE"
                    ent_eid = cu_grndg_ent_pref+f"{grndg_file_type}/{img_id.split('.')[0]}/{cnt_img[img_id]}"
                    entity_grnd = aifutils.make_entity(g, ent_eid, sys_grounding)
                    type_assertion = aifutils.mark_type(g, type_eid, entity_grnd, type_rdf, sys_grounding, score)
                elif USC_GROUNDING:
                    type_eid = usc_grndg_type_pref+f"{grndg_file_type}/{img_id.split('.')[0]}/{cnt_img[img_id]}/ERE"
                    ent_eid = usc_grndg_ent_pref+f"{grndg_file_type}/{img_id.split('.')[0]}/{cnt_img[img_id]}"
                    entity_grnd = aifutils.make_entity(g, ent_eid, usc_sys_grounding)
                    type_assertion = aifutils.mark_type(g, type_eid, entity_grnd, type_rdf, usc_sys_grounding, score)
              
                
                # Done: 
                # 1. add if for the branches for image and keyframe.
                # 2. add aifutils.mark_keyframe_video_justification
                # 3. check output 
                # aifutils.mark_keyframe_video_justification(g, [entity, type_assertion], "NYT_ENG_20181231_03", "keyframe ID",
                #                                                    bb2, system, 0.234)
                # source: HC0005BR6_23
                # print(img_id)

                # Done: 
                # merge usc_grounding dict
                # add usc_grounding entities and clusters
                
                # Test 
#                 print("type_assertion",type_assertion, img_id)

                
                bb = Bounding_Box((bbox[0], bbox[1]), (bbox[2], bbox[3]))
                if not USC_GROUNDING:
                    if 'JPG' in type_assertion:
                        imgid = img_id.split('.')[0]
                        justif = aifutils.mark_image_justification(g, [entity_grnd, type_assertion], imgid, bb, sys_grounding, score)
#                     
                    elif 'Keyframe' in type_assertion:
                        imgid = img_id.split('.')[0].split('_')[0]
                        kfid = img_id.split('.')[0].split('_')[1] # it should be keyframe image id or keyframe number
                        justif = aifutils.mark_keyframe_video_justification(g, [entity, type_assertion], imgid, kfid, \
                                                                       bb, sys_grounding, score)
                elif USC_GROUNDING:
                    imgid = img_id.split('.')[0]
                    justif = aifutils.mark_image_justification(g, [entity_grnd, type_assertion], imgid, bb, usc_sys_grounding, score)
                else:
                    print('[Merge Error] in Main Body: the type_assertion is wrong')
                aifutils.add_source_document_to_justification(g, justif, p_id)
                aifutils.mark_informative_justification(g, entity_grnd, justif)
                
            
                if not USC_GROUNDING:
                    grounding_features = grnd['grounding_features'][ii].tolist()
                    instance_features = grnd['instance_features'][ii].tolist()
                    #add private data to this very bbox entity
                    j_d_g = json.dumps({'columbia_vector_grounding_v1.0': grounding_features})
                    j_d_i = json.dumps({'columbia_vector_instance_v1.0': instance_features})
                    aifutils.mark_private_data(g, entity_grnd, j_d_g, sys_grounding)
                    aifutils.mark_private_data(g, entity_grnd, j_d_i, sys_instance_matching)

                
                #### add clusters
                # Grounding Cluster
                # Done: filtering about punctuation
#                 translator = str.maketrans(string.punctuation, '_'*len(string.punctuation),'' )
#                 'entity_name'.translate(translator)
                entity_name_tmp = '_'.join(entity_name.split(' '))
                entity_name_in_IRI = "".join(x if x.isalpha() or x.isdigit() or x =='_' else '-' for x in entity_name_tmp)
                # '_'.join(entity_name.split(' '))
                #gbbox entity to rpi entity
                if not USC_GROUNDING:
                    if not clstr_prot_flag: #create cluster if not present
                        clst_eid = cu_grndg_clstr_txt_pref+f"{entity_name_in_IRI}/{cnt_ltf[entity_name]}"
                        clusterObj = aifutils.make_cluster_with_prototype(g, clst_eid, entity_ltf, sys_grounding)
                        clstr_prot_flag = True
                    #cluster current bbox with current ltf_entity
                    score = grnd['men-img-score'][ii]
                    aifutils.mark_as_possible_cluster_member(g, entity_grnd, clusterObj, score, sys_grounding)
                    # Done: add prototype as member
                    aifutils.mark_as_possible_cluster_member(g, entity_ltf, clusterObj, 1, sys_grounding)
                elif USC_GROUNDING:
                    if not clstr_prot_flag: #create cluster if not present
                        clst_eid = usc_grndg_clstr_txt_pref+f"{entity_name_in_IRI}/{cnt_ltf[entity_name]}"
                        clusterObj = aifutils.make_cluster_with_prototype(g, clst_eid, entity_ltf, usc_sys_grounding)
                        clstr_prot_flag = True
                    #cluster current bbox with current ltf_entity
                    score = grnd['men-img-score'][ii]
                    aifutils.mark_as_possible_cluster_member(g, entity_grnd, clusterObj, score, usc_sys_grounding)
                    # Done: add prototype as member
                    aifutils.mark_as_possible_cluster_member(g, entity_ltf, clusterObj, 1, usc_sys_grounding)
                    
                # BoundingBox Overlap Cluster (Instance Matching)
                #gbbox entity to objdet entity for instance matching
                if not USC_GROUNDING:
                    clstr_prot_b2b_flag = False
                    for jj,img_id_link in enumerate(grnd['link_ids'][ii]): #for all objdet bboxes
                        if img_id_link in img_entities:
                            entity_link_img = img_entities[img_id_link]
                        elif img_id_link in keyframe_entities:
                            entity_link_img = keyframe_entities[img_id_link]
                        else:
                            continue
                        if img_id in cnt_boxO: #to keep track of cnt of bbox overlap for same image
                            cnt_boxO[img_id] += 1
                        else:
                            cnt_boxO[img_id] = 1
                        if not clstr_prot_b2b_flag:
                            clst_b2b_eid = cu_grndg_clstr_img_pref+f"{img_id.split('.')[0]}/{cnt_boxO[img_id]}"
                            clusterObj_b2b = aifutils.make_cluster_with_prototype(g, clst_b2b_eid, entity_grnd, sys_grounding) # sys_instance_matching
                            clstr_prot_b2b_flag = True

                        score = grnd['link_scores'][ii][jj] #IoU of grnd bbox and objdet bbox
                        aifutils.mark_as_possible_cluster_member(g, entity_link_img, clusterObj_b2b, score, sys_grounding) # sys_instance_matching
                        # Done: add prototype as member
                        aifutils.mark_as_possible_cluster_member(g, entity_grnd, clusterObj_b2b, 1, sys_grounding) # sys_instance_matching
                    
    # Check Point: merged_ttl_D2
#     /data/bobby/AIDA/M18_copy/data/merged_ttl/merged_ttl_D2/
#     IC0011VEA.ttl
#     GroundingBox
    with open(os.path.join(merged_graph_path, p_id+'.ttl'), 'w') as fout:
        serialization = BytesIO()
        g.serialize(destination=serialization, format='turtle')
        fout.write(serialization.getvalue().decode('utf-8'))
    #sys.stdout.write('Key {}/{} \r'.format(k,len(parent_dict)))                
    sys.stdout.flush()
pool = mp.Pool(processes=processes_num)
    #for x,y in candidateDic.items():
    #print candidateDic.keys()
  
if TEST_MODE:
    # [break] only for dockerization testing
    keys_list = [k for k in parent_dict.keys()]
    res = pool.map(transferAIF, keys_list[:5])
else:
    res = pool.map(transferAIF, parent_dict.keys())
print(datetime.now())

print('Graph Merging Finished.')

