#!/bin/bash

KWS_PATH=$PWD
DATA_PATH=$KWS_PATH/data2
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/kwt3/ \
--mel_upper_edge_hertz 7600 \
--optimizer 'adamw' \
--lr_schedule 'cosine' \
--how_many_training_steps '1' \
--eval_step_interval 72 \
--warmup_epochs 10 \
--l2_weight_decay 0.1 \
--learning_rate '0.0005' \
--batch_size 128 \
--label_smoothing 0.1 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
kws_transformer \
--num_layers 12 \
--heads 3 \
--d_model 192 \
--mlp_dim 768 \
--dropout1 0.1 \
--attention_type 'time' \
