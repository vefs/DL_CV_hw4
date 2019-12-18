# Install detectron2
Follow https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

# Prepare dataset
Follow [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
Check your dataset by visualizer

# Sone Config parameter (For Reference)

cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.MAX_ITER = 100000 

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20    --> category number = 20 
