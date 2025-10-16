python -W ignore train_net.py\
      --num-gpus 2\
      --resume\
      --config configs/vgg_city2fgcity_depth_rn50.yaml\
      SEMISUPNET.BURN_UP_STEP 20000\
      SEMISUPNET.ALIGN_MODEL dinov3_vitb16\
      SEMISUPNET.DINO_PATCH_SIZE 16\
      OUTPUT_DIR output/train_student_ViTHplus_dinov3_with_ViTG_dinov2_labeller_new_pseudo_labels_DEPTH_rn50\
      SEMISUPNET.LABELER_TARGET_PSEUDOGT /home/bui/DAOD/DINO_Teacher/output/dino_label/test_vig14_pseudo_labels/predictions/cityscapes_foggy_val_dino_anno_vitg.pkl