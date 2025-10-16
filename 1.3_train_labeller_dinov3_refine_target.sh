python train_net.py\
      --num-gpus 2\
      --resume\
      --config configs/vit_labeller_pseudo_labels_refine.yaml\
      OUTPUT_DIR output/dino_label/test_vitg14\
      SOLVER.IMG_PER_BATCH_LABEL 8\
      DATASETS.TRAIN_UNLABEL '("cityscapes_foggy_train",)'\
      DATASETS.TEST '("cityscapes_val", "cityscapes_foggy_val")'\
      SEMISUPNET.DINO_BBONE_MODEL dinov2_vitg14\
      SEMISUPNET.LABELER_TARGET_PSEUDOGT /home/bui/DAOD/DINO_Teacher/output/dino_label/test_vitg14/predictions/cityscapes_foggy_train_dino_anno_vitg.pkl