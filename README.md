# One-Shot Object Detection without Fine-Tuning


## Installation

Please check [INSTALL.md](INSTALL.md) (same as original FCOS) for installation instructions. 


## Citation

```
@article{OneshotDet,
Author = {Xiang Li and Lin Zhang and Yau Pun Chen and Yu-Wing Tai and Chi-Keung Tang},
Title = {One-Shot Object Detection without Fine-Tuning},
Year = {2020},
Journal = {ArXiv},
}
```

## Training

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={YOUR CHOSEN GPU ID}  python -m torch.distributed.launch  --nproc_per_node={NUM_GPU}  --master_port=$((RANDOM + 10000))  tools/train_net.py     --skip-test     --use-tensorboard     --config-file configs/fcos/2019_10_25_vanilla_siamse_backbone.yaml     DATALOADER.NUM_WORKERS 2     OUTPUT_DIR training_dir/2019_10_25_vanilla_siamse_backbone SOLVER.IMS_PER_BATCH  6 
```

## Evaluation

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={YOUR CHOSEN GPU ID} python tools/test_net.py   --seq_test    --config-file configs/fcos/2019_10_25_vanilla_siamse_backbone.yaml  TEST.LOAD_DIR training_dir/2019_10_25_vanilla_siamese_backbone  TEST.MIN_ITER 60000 TEST.MAX_ITER 60000   FEW_SHOT.CHOOSE_CLOSE False   FEW_SHOT.CHOOSE_SELECTED True      TEST.IMS_PER_BATCH {YOUR BATCH SIZE}       OUTPUT_DIR {YOUR_DIR}      FEW_SHOT.TEST_SELECTED_CLS  {CLASS YOU WANT TO EVAL (1->20)}
```