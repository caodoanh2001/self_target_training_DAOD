1. Run file 1.1_train_labeller_dinov3_source_only.sh for training labeller on source-only
2. Run file 1.2_gen_ps_labels.sh for generating labels for target samples on training target samples
3. Run 1.3_train_labeller_dinov3_refine_target.sh for refining
4. Run 2_gen_ps_labels.sh for generating pseudo labels for validation set
5. Run 3_train_student_depth.sh for training student network with depth


For labeller, use ViT-G backbone.

For student network, use ViT-B (Dinov3 ViT-B/16 distilled	86M	LVD-1689M) as align teacher