import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import json
import glob
import itertools
import torch, torchvision
import detectron2

import torchvision.transforms as transforms

from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from myutils import binary_mask_to_rle
################################################################333

def get_hw4_dicts(set_dir, sub_dir):
    img_dir = set_dir + '/' + sub_dir
    json_file = os.path.join(img_dir, "pascal_train.json")
    #[INFO] COCO Read JSON file:  hw4data/train/pascal_train.json
    print (" [INFO] COCO Read JSON file: ", json_file)
    coco = COCO(json_file)

    coco_classes = []  
    cateids    = coco.cats.keys()  
    '''
    for ii in range(len(cateids)+1):
        coco_classes.append('NA')

    for cateid_ii in cateids:
        ii = int(cateid_ii)
        coco_classes[ii] = coco.cats[cateid_ii]['name']
    '''
    for cateid_ii in cateids:    
        coco_classes.append(coco.cats[cateid_ii]['name'])

    print (" [INFO] coco classes", coco_classes)
      
    metada_name  = set_dir+'_'+sub_dir
    MetadataCatalog.get(metada_name).set(thing_classes=coco_classes)
    print (" [INFO] set meta: ",metada_name)
    
    dataset_dicts = []
    
    #img_ids = list(coco.imgs.keys())[:10]
    img_ids = list(coco.imgs.keys())
    print(" [INFO] total img_ids ", len(img_ids))
    for imgid in img_ids: 
        record = {}
        #('file_name': '2009_001816.jpg', 'id': 736, 'height': 375, 'width': 500}
        filename = os.path.join(img_dir, coco.imgs[imgid]['file_name'])
    
        record["file_name"] = filename
        record["image_id"]  = imgid
        record["height"]    = coco.imgs[imgid]['height']
        record["width"]     = coco.imgs[imgid]['width']
        img_info = coco.loadImgs(ids=imgid)
        #print(" \n img_info", img_info)
        #print (" ----- append record: img_ids=", imgid, "record=" ,record) 

        annids = coco.getAnnIds(imgid)
        inst_count = len(annids)
        #print(" [ann] inst id list: ", annids ," Number of instances: ", inst_count)
        anns   = coco.loadAnns(annids)
        objs = []
        for inst_ii in range(inst_count):    
            poly     = []
            #if(len(anns[inst_ii]['segmentation']) > 1 ):
            #    print(" !!! seg inst > 1, img_id=", imgid, " file", filename)
            for poly_ii in anns[inst_ii]['segmentation']:
                poly_f = [(x+0.5) for x in poly_ii]
                poly.append(poly_f)
                  

            #mask = coco.annToMask(anns[inst_ii])

            #poly_00 = anns[inst_ii]['segmentation'][0]

            #rle = coco.annToRLE(anns[inst_ii])

            cat_id = anns[inst_ii]['category_id'] - 1
            obj = {
                "bbox"        : anns[inst_ii]['bbox'],
                "bbox_mode"   : BoxMode.XYWH_ABS,
                "segmentation": poly,
                #"segmentation": rle,
                "category_id" : cat_id,
                "iscrowd"     : anns[inst_ii]['iscrowd']
            }
    
            #print(" inst_ii=: ", inst_ii, " Image id of this instance:", anns[inst_ii]['image_id'])
            #print(" bbox: ", obj["bbox"] , " bbox mode: ", obj["bbox_mode"])
            #print(" category_id: ", obj["category_id"] , " iscrowd: ", obj["iscrowd"])

            #print(" obj segmentation: ", obj["segmentation"])
            objs.append(obj)
        #print ( " append record: ", record) 
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def check_dataset():
    hw4set_metadata = MetadataCatalog.get("hw4data_train")
    dataset_dicts = get_hw4_dicts("hw4data","train")
    print (" Check dataset by random sample ")
    for d in random.sample(dataset_dicts, 10):
        #d = dataset_dicts[0]
        print (" draw file: ",d["file_name"], "img_id=",  d["image_id"] )
        for anno in d["annotations"]:
            print (" categori_id: ", anno["category_id"], " bbox=", anno["bbox"] )
        
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=hw4set_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("window2", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)

######################## PREPARE DATASET
for d in ["train", "val"]:
    DatasetCatalog.register("hw4data_" + d, lambda d=d: get_hw4_dicts("hw4data", d))


######################## 
'''
 [TEST] Error!!, inst not found for  hw4data/tests/2009_004886.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2010_000065.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2010_004763.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2008_006523.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2011_000238.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2011_000912.jpg
 [TEST] Error!!, inst not found for  hw4data/tests/2009_002727.jpg
'''
def eval_single():
    #setup_logger()   
    #im = cv2.imread("hw4data/train/2009_001816.jpg")
    #im = cv2.imread("hw4data/train/2008_003270.jpg")
    im = cv2.imread("hw4data/tests/2009_004886.jpg")

    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "hw4_output/model_final.pth"
    #cfg.MODEL.WEIGHTS = "hw4_output/model_final_f10217.pkl"
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    #cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                                                             
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"

    #print (" [INFO] cfg OUTPUT_DIR : ", cfg.OUTPUT_DIR)
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

    print (" [INFO] cfg NUM_CLASSES: ", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print (" [INFO] cfg pretrain: ", cfg.MODEL.WEIGHTS )

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print (" [INFO]  predict inst num: ", len(outputs["instances"]))
    print (" [INFO]  predict class   : ", outputs["instances"].pred_classes+1)
    print (" [INFO]  predict scores  : ", outputs["instances"].scores)
    print (" [INFO]  predict bbox    : ", outputs["instances"].pred_boxes)
    print (" [INFO]  predict mask    : ", outputs["instances"].pred_masks.shape)
    pred_masks = outputs["instances"].pred_masks.cpu()
    mask_shape = outputs["instances"].pred_masks.shape
    if(len(mask_shape) == 3):
        [c, w, h]  = mask_shape
    else:
        c = 1
        [w, h]     = mask_shape
   
    binary_masks    = pred_masks.float()*255
    cv2_masks       = np.zeros([w, h, c])

    print("binary_mask shape: ", c, w, h)
    print("cv2 mask shape   : ", cv2_masks.shape)
    for kk in range(c):
        for jj in range(h):
            for ii in range(w):
                cv2_masks[ii][jj][kk] = binary_masks[kk][ii][jj]   
    #for kk in range(c):
    #   cv2.imshow("window2", cv2_masks[:][:][kk])
    #   cv2.waitKey(0)
    

    #rle   = binary_mask_to_rle(masks[:,:,i]) # save binary mask to RLE, e.g. 512x512 -> rle
    #mask = bool_mask.float()
    #mask
    #plt.imshow(mask) 
    #plt.show() 
    
    #for bool_mask in pred_masks:
    
    inst_cnt = len(outputs["instances"])

    if(inst_cnt/3 > 0):
        row_cnt = inst_cnt/3 +1
    for i in range(len(outputs["instances"])):
        mask = binary_masks[i]
        #mask = transforms.ToPILImage()( binary_masks[i])
        print("mask shape", mask.size)
        plt.subplot(row_cnt, 3, i+1)
        plt.title("Instance {}, category={}".format(i+1, outputs["instances"].pred_classes[i]+1))
        plt.imshow(mask)
    plt.show() 
 
    #masks = binary_masks
    masks = cv2_masks
    #for i in range(len(outputs["instances"])):
    #    rle   = binary_mask_to_rle(masks[:,:,i]) 
    #    print(rle)

    

######################## MODEL TRAIN
def train_model():
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("hw4data_train",)
    cfg.DATASETS.TEST  = ()
        
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = "hw4_model/model_00.pth"
    #cfg.MODEL.WEIGHTS = "hw4_model/model_01.pth"
    #cfg.MODEL.WEIGHTS = "hw4_model/model_02.pth"
    #cfg.MODEL.WEIGHTS = "hw4_model/model_03.pth"
    cfg.MODEL.WEIGHTS = "hw4_model/model_04.pth"
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    #cfg.SOLVER.BASE_LR = 0.00025
    #cfg.SOLVER.BASE_LR = 0.005
    #cfg.SOLVER.BASE_LR = 0.0005
    #cfg.SOLVER.BASE_LR = 0.00005
    cfg.SOLVER.BASE_LR = 0.000005

    #cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough, but you can certainly train longer
    #cfg.SOLVER.MAX_ITER = 10000   #01.pth
    #cfg.SOLVER.MAX_ITER = 12000   #02.pth 
    #cfg.SOLVER.MAX_ITER = 15000   #03.pth
    #cfg.SOLVER.MAX_ITER = 50000   #04.pth
    cfg.SOLVER.MAX_ITER = 100000   #05.pth
   

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20              # 20 class
    cfg.OUTPUT_DIR = 'hw4_output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print (" [TRAIN] cfg NUM_CLASSES: ", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print (" [TRAIN] cfg pretrain: ", cfg.MODEL.WEIGHTS )
    setup_logger()
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    #trainer.resume_or_load(resume=True)
    trainer.train()

######################## MODEL EVAL
def eval_model():
    cfg.OUTPUT_DIR = 'hw4_output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("hw4data_val")
    predictor = DefaultPredictor(cfg)
    '''
    dataset_dicts   = get_hw4_dicts("hw4data", "val")
    hw4set_metadata = MetadataCatalog.get("hw4data_train")
    #hw4set_metadata = MetadataCatalog.get("hw4data_val")
    
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=hw4set_metadata, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2_imshow(v.get_image()[:, :, ::-1])
        cv2.imshow("window2", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    '''
'''
{1: {'supercategory': 'aeroplane', 'name': 'aeroplane', 'id': 1},
 2: {'supercategory': 'bicycle', 'name': 'bicycle', 'id': 2},
 3: {'supercategory': 'bird', 'name': 'bird', 'id': 3},
 4: {'supercategory': 'boat', 'name': 'boat', 'id': 4},
 5: {'supercategory': 'bottle', 'name': 'bottle', 'id': 5},
 6: {'supercategory': 'bus', 'name': 'bus', 'id': 6},
 7: {'supercategory': 'car', 'name': 'car', 'id': 7},
 8: {'supercategory': 'cat', 'name': 'cat', 'id': 8},
 9: {'supercategory': 'chair', 'name': 'chair', 'id': 9},
 10: {'supercategory': 'cow', 'name': 'cow', 'id': 10},
 11: {'supercategory': 'diningtable', 'name': 'diningtable', 'id': 11},
 12: {'supercategory': 'dog', 'name': 'dog', 'id': 12},
 13: {'supercategory': 'horse', 'name': 'horse', 'id': 13},
 14: {'supercategory': 'motorbike', 'name': 'motorbike', 'id': 14},
 15: {'supercategory': 'person', 'name': 'person', 'id': 15},
 16: {'supercategory': 'pottedplant', 'name': 'pottedplant', 'id': 16},
 17: {'supercategory': 'sheep', 'name': 'sheep', 'id': 17},
 18: {'supercategory': 'sofa', 'name': 'sofa', 'id': 18},
 19: {'supercategory': 'train', 'name': 'train', 'id': 19},
 20: {'supercategory': 'tvmonitor', 'name': 'tvmonitor', 'id': 20}}
'''
######################## DETECT
def detect_mode():
    #setup_logger()
    
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "hw4_output/model_final.pth"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    predictor = DefaultPredictor(cfg)

    print (" [DETE] cfg NUM_CLASSES: ", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print (" [DETE] cfg pretrain: ", cfg.MODEL.WEIGHTS )
    cocoGt = COCO("hw4data/tests/test.json")
    coco_dt = []
    inst_gen = 0    

    img_ids = list(cocoGt.imgs.keys())
    print(" [DETE] total img_ids ", len(img_ids))
    #for imgid in cocoGt.imgs:
    for imgid in img_ids: 
        #img_path = "hw4data/tests/" + cocoGt.loadImgs(ids=imgid)[0]['file_name']
        img_path = "hw4data/tests/" + cocoGt.imgs[imgid]['file_name']
        im = cv2.imread(img_path) # load image    
        
        outputs = predictor(im)
        n_instances = len(outputs["instances"])
        if n_instances > 0: 
            #print ("\n [DETE]  predict img      : ", img_path)
            #print (" [DETE]  inst count       : ", len(outputs["instances"]))
            pred_masks = outputs["instances"].pred_masks.cpu()
            mask_shape = outputs["instances"].pred_masks.shape
            if(len(mask_shape) == 3):
                [c, w, h]  = mask_shape
            else:
                c = 1
                [w, h]     = mask_shape
   
            binary_masks    = pred_masks.float()*255
            cv2_masks       = np.zeros([w, h, c])
            for kk in range(c):
                for jj in range(h):
                    for ii in range(w):
                        cv2_masks[ii][jj][kk] = binary_masks[kk][ii][jj]
 
            for i in range(n_instances): # Loop all instances
                pred = {}
                pred['image_id']     = imgid # this imgid must be same as the key of test.json
                pred['category_id']  = int(outputs["instances"].pred_classes[i]+1)
                pred['score']        = float(outputs["instances"].scores[i])
                pred['segmentation'] = binary_mask_to_rle(cv2_masks[:,:,i]) 
                #if(inst_gen < 5):                
                #    print (" [DETE]  predict img id   : ", pred['image_id'])
                #    print (" [DETE]  predict class    : ", pred['category_id'])
                #    print (" [DETE]  predict scores   : ", pred['score'] )
                #    print (" [INFO]  predict bbox     : ", outputs["instances"].pred_boxes[i])
                #    print (" [DETE]  predict mask     : ", mask_shape)


                coco_dt.append(pred)
                inst_gen = inst_gen + 1 
        else:
            print (" [TEST] Error!! inst not found for ", img_path)

    with open("hw4data/submission.json", "w") as f:
        json.dump(coco_dt, f)
    print (" [TEST] Total detect inst ", inst_gen)

 
####################################################
## MAIN FUNC
#check_dataset()

#train_model()

eval_single()
#detect_mode()

####################################################
'''
cocoGt = COCO("test.json")
coco_dt = []

for imgid in cocoGt.imgs:

    #image = cv2.imread("test_images/" + coco.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
    #masks, categories, scores = model.predict(image) # run inference of your model

    n_instances = len(score)    if len(categories) > 0: # If any objects are detected in this image
        for i in range(n_instances): # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid # this imgid must be same as the key of test.json
            pred['category_id'] = int(categories[i])
            pred['segmentation'] = binary_mask_to_rle(masks[:,:,i]) # save binary mask to RLE, e.g. 512x512 -> rle
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

'''
