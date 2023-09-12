#!/usr/bin/env sh

#cp resnet101-5d3b4d8f.pth /root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

# python train.py configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py --gpus 2

# ./tools/dist_train.sh configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py 2

# python train.py /home/notebook/code/personal/S9043252/SMP2.0/multi-parsing/configs/solov2/solov2_r101_dcn_fpn_3x_coco.py

# bash dist_train.sh configs/repparsing/MHP_r50_fpn_half_gpu_1x_repparsing_DCN_fusion.py 2

# bash dist_train.sh work_dirs/MHP_release_r50_fpn_8gpu_1x_repparsing_v0_DCN_fusion_metrics/MHP_r50_fpn_half_gpu_1x_repparsing_DCN_fusion_metrics.py 2

# bash dist_train.sh configs/repparsing/MHP_r50_fpn_half_gpu_12x_repparsing_fusion_metrics.py 2

# python train.py configs/repparsing/CHIP_r50_fpn_half_gpu_3x_repparsing_DCN_fusion_metrics_noneg_cluster.py --gpus 1 --seed 42
# bash dist_train.sh configs/repparsing/MHP_r101_fpn_half_gpu_1x_repparsing_DCN_fusion_metrics_light_cluster.py 2

bash dist_train.sh configs/smp/CHIP_r101_fpn_8gpu_3x_offset_parsing_v1_ori_grid_unified_DCN.py 4

# python train.py configs/repparsing/MHP_r50_fpn_half_gpu_1x_repparsing_DCN_fusion_metrics.py --gpus 1 --seed 42
