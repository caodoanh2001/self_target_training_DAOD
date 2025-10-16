python train_net.py\
      --num-gpus 2\
      --resume\
      --config configs/vit_labeller_pseudo_labels_ViTHplus_dinov3.yaml\
      OUTPUT_DIR output/dino_label/vit_labeller_pseudo_labels_ViTHplus_dinov3\
      SOLVER.IMG_PER_BATCH_LABEL 8\
      DATASETS.TEST '("cityscapes_val","cityscapes_foggy_val")'\
      SEMISUPNET.DINO_BBONE_MODEL dinov3_vith16plus\