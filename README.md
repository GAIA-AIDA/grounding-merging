
# Visual Grounding and Graph Merging
#### GAIA-AIDA: Columbia Vision Pipeline (Visual Grounding and Graph Merging)
#### Source provided by Bo Wu (Bobby), Columbia University (CU)

#### Main Modules: Feature Extraction, Visual Grounding and Instance Matching, Graph Merging

## A. Overview of Columbia University Vision Pipeline
![image](architecture.png)
You can find the module containers for Columbia University Vision Pipeline by following links:
[CU Grounding and Merging](https://hub.docker.com/r/dannapierskitoptal/aida-grounding-merging), 
[CU Object Detection](https://hub.docker.com/r/dannapierskitoptal/aida-object-detection), 
[CU Face/Flag/Landmark Recognition](https://hub.docker.com/r/dannapierskitoptal/aida-face-building), 
[UIUC Text Pipeline](https://github.com/isi-vista/aida_pipeline/tree/manling), 
[USC Grounding](https://github.com/isi-vista/aida_detect_ground) [Optional]

## B. Environment Installation
Local development environment:
  - Python 3.6.3   
  - Anaconda 3-4.4  
  - Tensorflow 1.12.0  

Running Docker
```
$ INPUT=/path_to_ldc_corpus/
$ # please run the necessary modules (CU_Object_Detection, CU_Face/Flag/Landmark_Recognition and UIUC_Text_Pipeline) to get or download the the required result files to the shared directory ...
$ SHARED=/columbia_data_root/columbia_vision_shared/
$ OUTPUT=/path_to_output_directory/
$ # please create the folder /columbia_data_root/ under the ${OUTPUT}/WORKING/ directory for output files
$ mkdir  ${OUTPUT}/WORKING/columbia_data_root/
$ GPU_ID=[a single integer index to the GPU]

$ docker pull gaiaaida/grounding-merging
$ docker images
$ # The model folder columbia_visual_grounding_models/ can be found in the docker image directly or the soucecode repository
$ # Mapping the path environment variables ${INPUT}, ${SHARED} and ${OUTPUT} to the /root/LDC/, /root/shared/, and /root/output/
$ docker run -it -e CUDA_VISIBLE_DEVICES=${GPU_ID} --gpus ${GPU_ID} --name aida-grounding-merging -v columbia_visual_grounding_models/:/root/models -v ${INPUT}:/root/LDC/:ro -v ${SHARED}:/root/shared ${OUTPUT}/WORKING/columbia_data_root/:/root/output/ gaiaaida/grounding-merging /bin/bash
```

Building Docker

```
$ docker build . --tag columbia-gm
$ $ docker run -it -e CUDA_VISIBLE_DEVICES=${GPU_ID} --gpus ${GPU_ID} --name grounding-merging -v columbia_visual_grounding_models/:/root/models -v ${INPUT}:/root/LDC/:ro -v ${SHARED}:/root/shared columbia-gm /bin/bash
$ docker exec -it aida-gm /bin/bash
$ # python smoke_test.py

$ docker build . --tag columbia-gm
$ docker run -itd --name aida-gm -p [HOST_PORT]:8082 -v columbia_visual_grounding_models/:/root/models -v ${INPUT}:/root/LDC/:ro -v ${SHARED}:/root/shared columbia-gm /bin/bash
$ docker port aida-gm
$ docker exec -it aida-gm /bin/bash
# jupyter notebook --allow-root --ip=0.0.0.0 --port=8082 & 
# Access jupyter on the host machine [HOST_URL]:[HOST_PORT].

$ docker exec -it aida-gm /bin/bash
$$ python ./Feature_Extraction.py
$$ echo expect to see get [CU Visual_Features] files:
${SHARED}/cu_grounding_matching_features/semantic_features_jpg.lmdb, semantic_features_keyframe.lmdb, instance_features_jpg.lmdb, instance_features_keyframe.lmdb
$$ echo expect to see get [CU Grounding] file: ${SHARED}/cu_grounding_results/grounding_dict.pickle
$$ python ./Visual_Grounding_mp.py
$$ echo expect to see get [CU Dictionary] files for USC: '${SHARED}/cu_grounding_dict_files/entity2mention_dict.pickle', '${SHARED}/cu_grounding_dict_files/id2mentions_dict.pickle'
$$ python ./Graph Merging.py
$$ echo expect to see get [CU Merging] files: '${OUTPUT}/WORKING/columbia_data_root/cu_graph_merging_ttl/merged_ttl/ 

```

#### The 3 main steps should be run one by one: 

1. Feature_Extraction.ipynb
2. Visual_Grounding_mp.ipynb
3. Graph_Merging.ipynb 

The steps associates with "feature extraction", "visual grounding and instance matching" and "graph merging" parts.

[Optional Setting] Running the modules of Columbia University only does not require to run USC branch, the merging steps for USC grounding in Visual_Grounding_mp.py can be commented. But if you want to merge the grounding results from USC, please keep the merging steps in the code for USC grounding and follow these steps: 1. Generating the immediate dictionary files by run Visual_Grounding_mp.py as input for USC grounding; 2. Runnnig the USC grounding branch and generating the USC grounding results (as dictionary object); 3. Running our codes and the system will use two types of grounding results as input.

### C. Parameter Setting  
Grounding score threshold: 0.85  
Sentence score threshold: 0.6  
Set the number of multiple processes: processes_num = 32

## D. Data Download  

#### 1. Corpus Data Download (one example of the input data) 
```
    Download the folder of the LDC corpus data (https://github.com/isi-vista/aida-integration).
```  
File List:  
[LDC] 3 files (from ISI, sorted by UIUC)  
[LTF] file (from ISI)



#### 2. Download Exemplary Data and Pretrained Model Files for Testing
```
    Download Link: https://drive.google.com/drive/folders/1JQak5s31I4nwGNASpOQ_GbQpJuS85lFr?usp=sharing    
    Download the shared folders (/columbia_data_root/columbia_vision_shared/) and the visual grounding model files (/columbia_data_root/columbia_visual_grounding_models/) for visual grounding, instance matching and graph merging.
```  
- Data Structure
```
columbia_data_root
├── columbia_vision_shared
│   ├── cu_objdet_results
│   ├── cu_grounding_matching_features
│   ├── cu_grounding_results
│   ├── uiuc_ttl_results
│   ├── uiuc_asr_files
│   ├── cu_grounding_dict_files
│   ├── cu_ttl_tmp
│   ├── cu_graph_merging_ttl
│   └── ...
└── columbia_visual_grounding_models
```


- Specify data paths
```
#AVAILABLE_GPU=${GPU_ID}
corpus_path = '/root/LDC/' # set /root/LDC/ as the corpus data path
working_path = '/root/shared/' # set /root/shared/ as the shared folder path
model_path = '/root/models/' # set /root/models/ as the model folder path
```



File List:  
[UIUC] 3 files (from UIUC)  
```
    columbia_vision_shared/uiuc_ttl_results/
    columbia_vision_shared/uiuc_asr_files/
```  
[CU obj_det] files (from CU_obj)   
```
    columbia_vision_shared/cu_objdet_results/
```  
[USC] files (from USC)   
```
    columbia_vision_shared/usc_grounding_dict/
```  
[CU clustering] files (from CU_face)
```
    columbia_vision_shared/cu_ttl_tmp/
```  

[Model] files (from CU_gm)    
```
    columbia_visual_grounding_models/
```

#### 3. Results or Intermediate Data Download for Module Testing  
[CU Visual_Features] files (from CU_gm)  
```
    columbia_vision_shared/cu_grounding_matching_features/
```
[CU Grounding] files (from CU_gm)  
```
    columbia_vision_shared/cu_grounding_results/
```
[CU Dictionary] files (from CU_gm)
```
    columbia_vision_shared/cu_grounding_dict_files/
```
[CU Merging] files (from CU_gm) 
```
    do not need to use for testing
```

## E. Required Input/Result Output of Each Module  

#### 1. Feature Extraction  
#### Input:   
[LDC] 4 files  
```
    parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
    kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
    kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
    jpg_path = corpus_path + 'data/jpg/jpg/' 
```
[UIUC] 2 files  
```
    video_asr_path = working_path + 'uiuc_asr_files/' +'ltf_asr/'
    video_map_path = working_path + 'uiuc_asr_files/' +'map_asr/'
```
[CU obj_det] files  
``` 
    det_results_path_img = working_path + 'cu_objdet_results/' + 'det_results_merged_34a.pkl'
    det_results_path_kfrm = working_path + 'cu_objdet_results/' + 'det_results_merged_34b.pkl'
```  
[CU grounding_model] files
```
    grounding_model_path = model_path + 'model_ELMo_PNASNET_VOA_norm'
    matching_model_path = model_path + 'model_universal_no_recons_ins_only'
```
#### Output:   
[CU Visual_Features] Common Space Embeddings (for grounding)  
```
    out_path_jpg = working_path + 'cu_grounding_matching_features/' + 'semantic_features_jpg.lmdb'
    out_path_kfrm = working_path + 'cu_grounding_matching_features/' + 'semantic_features_keyframe.lmdb'
```
[CU Visual_Features] Instance Matching Features (for instance clustering)  
```
    out_path_jpg = working_path + 'cu_grounding_matching_features/' + 'instance_features_jpg.lmdb'
    out_path_kfrm = working_path + 'cu_grounding_matching_features/' + 'instance_features_keyframe.lmdb'
```

#### 2. Visual Grounding and Instance Matching   
#### Input: 
[LDC] 5 files  
```
    parent_child_tab = corpus_path + 'docs/parent_children.tab'
    kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
    kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
    jpg_path = corpus_path + 'data/jpg/jpg/'
    ltf_path = corpus_path + 'data/ltf/ltf/'
```
[UIUC] 4 files
```
    txt_mention_ttl_path = working_path + 'uiuc_ttl_results/' 
    pronouns_path = working_path + 'uiuc_asr_files/' + 'pronouns.txt'
    video_asr_path = working_path + 'uiuc_asr_files/' +'ltf_asr/'
    video_map_path = working_path + 'uiuc_asr_files/' +'map_asr/'
```
[CU obj_det]   
```
    As mentioned before.   
```  
#### Intermediate Output:  
[CU Dictionary] files (for USC)
```
    entity2mention_dict_path = working_path + 'cu_grounding_dict_files/' + 'entity2mention_dict.pickle'
    id2mentions_dict_path = working_path + 'cu_grounding_dict_files/' + 'id2mentions_dict.pickle' 
```  
#### Output:  
[CU Grounding] files  
```
    grounding_dict_path = working_path + 'cu_grounding_results/' + 'grounding_dict.pickle'
    grounding_log_path = working_path + 'cu_grounding_results/' + 'log_grounding.txt'
```


### 3. Graph Merging  
#### Input:    
[LDC] 3 files  
```
    parent_child_tab = corpus_path + 'docs/parent_children.tab' # should be sorted
```
[CU Visual_Features] files  
```
    Generated by the module of Visual Feature: cu_grounding_matching_features/semantic_features_jpg.lmdb, semantic_features_keyframe.lmdb, instance_features_jpg.lmdb, instance_features_keyframe.lmdb
```
[CU Grounding] file  
```
    Generated by the module of Visual Grounding: grounding_dict_path = working_path + 'cu_grounding_results/' + version_folder + 'grounding_dict.pickle'
```
[USC] file  
```
    usc_dict_path = working_path + 'usc_grounding_dict/' + version_folder + 'uscvision_grounding_output_cu_format.pickle' 
```  
[UIUC] file  
```
    txt_mention_ttl_path = working_path + 'uiuc_ttl_results/' + uiuc_run_folder # 1/7th May
```  
[CU clustering] 2 files  
```
    cu_ttl_tmp_path = working_path + 'cu_ttl_tmp/'
    cu_ttl_path = cu_ttl_tmp_path + version_folder + 'm18/'
    cu_ttl_ins_path = cu_ttl_tmp_path + version_folder + 'm18_i_c/'
```  
#### Output: 
[CU Merging]  files   
```
    merged_graph_path = output_path + 'cu_graph_merging_ttl/' + 'merged_ttl/'
```

## F. Main Steps of Running  
#### Feature Extraction  
get the data from [UIUC] and [CU obj_det]  
Run feature extraction program   
#### Grounding and Instance Matching  
Run the first one part of grounding program to generate intermediate dict file for [USC]  
Merge the intermediate file from [USC]  
Run grounding program and generate results (parallelly)  
#### Graph Merging  
Get updated cu_ttl from [CU clustering]   
Run merging program    
#### Result Checking  
Check grounding dict result  
Check grounded entities and clusters  
Check the prefix for entity (columbia or usc ..)  
Check merged_ttl by validator on local server pineapple    
Output: /columbia_vision_shared/merged_ttl/  

### Grounding Example  
CU grounding_dict file  

```
'http://www.isi.edu/gaia/entities/c5649544-38d8-4e28-b104-af431556a1a1':{  
  'textual_features':array(  [  ],
  dtype=float32),
  'name':'Лавров',
  'type_rdf':  rdflib.term.URIRef('https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/LDCOntology#Person'),
  'mentions':{  
    'f74850e12aef14b83bad4071dde1b2a6cb661':{  },
    'f74850e12aef14b83bad4071dde1b2a6cb664':{  
      'grounding':{  
        'IC00121KH.jpg.ldcc':{  },
        'IC00121KI.jpg.ldcc':{  },
        'IC00121KF.jpg.ldcc':{  },
        'IC00121KK.jpg.ldcc':{  }
      },
      'textual_features':array(      [  ],
      dtype=float32),
      'name':'Lavrov  Lavrov ',
      'sentence':'Lavrov  Lavrov  in the image'
    }
```

## G. References
If you use our Docker container images or codes in your research, please cite the following papers.
- [GAIA: A Fine-grained Multimedia Knowledge Extraction System](http://www.ee.columbia.edu/ln/dvmm/publications/20/aidaacl2020demo.pdf).
  Manling Li, Alireza Zareian, Ying Lin, Xiaoman Pan, Spencer Whitehead, Brian Chen, Bo Wu, Heng Ji, Shih-Fu Chang, Clare Voss, Daniel Napierski and Marjorie Freedman
  Proc. The 58th Annual Meeting of the Association for Computational Linguistics (ACL2020) Demo Track

- [GAIA at SM-KBP 2019 - A Multi-media Multi-lingual Knowledge Extraction and Hypothesis Generation System](https://blender.cs.illinois.edu/paper/gaia_smkbp_2019.pdf).
  Manling Li, Ying Lin, Ananya Subburathinam, Spencer Whitehead, Xiaoman Pan, Di Lu, Qingyun Wang, Tongtao Zhang, Lifu Huang, Heng Ji, Alireza Zareian, Hassan Akbari, Brian Chen, Bo Wu, Emily Allaway,Shih-Fu Chang, Kathleen McKeown, Yixiang Yao, Jennifer Chen, Eric Berquist, Kexuan Sun, Xujun Peng, Ryan GabbardMarjorie Freedman, Pedro Szekely, T.K. Satish Kumar, Arka Sadhu, Ram Nevatia, Miguel Rodriguez, Yifan Wang, Yang Bai, Ali Sadeghian, Daisy Zhe Wang
  Proc. Text Analysis Conference (TAC2019)

- [GAIA - A Multi-media Multi-lingual Knowledge Extraction and Hypothesis Generation System](http://nlp.cs.rpi.edu/paper/gaia2018.pdf).
  Tongtao Zhang, Ananya Subburathinam, Ge Shi, Lifu Huang, Di Lu, Xiaoman Pan, Manling Li, Boliang Zhang, Qingyun Wang, Spencer Whitehead, Heng Ji, Alireza Zareian, Hassan Akbari, Brian Chen, Ruiqi Zhong, Steven Shao, Emily Allaway, Shih-Fu Chang, Kathleen McKeown, Dongyu Li, Xin Huang, Xujun Peng, Ryan Gabbard, Marjorie Freedman, Ali Sadeghian, Mayank Kejriwal, Ram Nevatia, Pedro Szekely, Ali Sadeghian and Daisy Zhe Wang
  Proc. Text Analysis Conference (TAC2018)

- [Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding](https://arxiv.org/pdf/1811.11683.pdf).
  Hassan Akbari, Svebor Karaman, Surabhi Bhargava, Brian Chen, Carl Vondrick, and Shih-Fu Chang  
  Proc. International Conference on Computer Vision and Pattern Recognition (CVPR2019)

### Updates

#### 2020.03.26 Update Paths
```
root
  to: working_path =  '/root/'
  from: corpus_path = '/root/dryrun/'
  to: corpus_path = '/root/LDC'
model
  from: model/
  to: model_path = models_root + 'columbia_visual_grounding_models/'
shared
  from: /objdet_results/, rpi_ttl/,raw_files/, tmp/, usc_dict/, /cu_ttl,/merged_ttl
  to: /cu_objdet_results/, uiuc_ttl_results/,uiuc_asr_files/, cu_grounding_dict_files, usc_grounding_dict/, /cu_ttl_tmp,/cu_graph_merging_ttl
local
  from: all_features/
  to: cu_grounding_matching_features/
```

