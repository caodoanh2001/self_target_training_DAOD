python train_net.py\
      --num-gpus 1\
      --resume\
      --gen-labels\
      --config configs/vit_labeller_test.yaml\
      OUTPUT_DIR output/dino_label/test_vitg14\
      DATASETS.TEST '("cityscapes_foggy_val",)'\
      SEMISUPNET.DINO_BBONE_MODEL dinov2_vitg14\