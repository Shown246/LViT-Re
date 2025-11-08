# Config.py
# -*- coding: utf-8 -*-
import os
import torch
import ml_collections

# ======================================================
# GLOBAL SETTINGS
# ======================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

deterministic = True
use_cuda = torch.cuda.is_available()

# ======================================================
# TRAIN SETTINGS
# ======================================================
task_name       = 'Covid19'      # 'MoNuSeg' or 'Covid19'
model_name      = 'LViT'
epochs          = 3
img_size        = 224
batch_size      = 2
learning_rate   = 3e-4           # Covid19 recommended 3e-4, MoNuSeg is usually 1e-3

print_frequency = 1
vis_frequency   = 1
save_after_epoch = 1             # only start saving best after this epoch
early_stopping_patience = 50

num_workers     = 8
use_dataparallel = True
profile_model   = False          # enable THOP or not
pretrain        = False
cosineLR        = True

tensorboard     = True
save_model      = True

# ======================================================
# PATHS
# ======================================================
train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_'
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

test_session   = session_name

n_channels     = 3
n_labels       = 1   # binary mask

# ======================================================
# DEEP SUPERVISION HEAD WEIGHTS
# coarse -> mid -> fine   (list length must match ds_levels)
# ======================================================
ds_weights = (0.2, 0.3, 0.4)

# ======================================================
# VIT CONFIG
# ======================================================
def get_CTranS_config():
    cfg = ml_collections.ConfigDict()
    cfg.transformer = ml_collections.ConfigDict()

    cfg.KV_size = 960
    cfg.transformer.num_heads = 4
    cfg.transformer.num_layers = 4
    cfg.transformer.embeddings_dropout_rate = 0.1
    cfg.transformer.attention_dropout_rate = 0.1
    cfg.transformer.dropout_rate = 0

    cfg.base_channel = 64      # base U-Net channel
    cfg.n_classes = 1
    cfg.expand_ratio = 4
    cfg.patch_sizes = [16, 8, 4, 2]

    return cfg
