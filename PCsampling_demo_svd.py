#@title Autoload all modules


from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling_svd
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling_svd import (ReverseDiffusionPredictor,
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import os.path as osp

import argparse
parser = argparse.ArgumentParser(description="some settings")
parser.add_argument("--datanum", type=int, default=1, help="the number of the test data")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  model_num = 'checkpoint.pth'

  ckpt_filename_kt = './exp/exp_kt/checkpoint_14.pth'
  ckpt_filename_xt = './exp/exp_xt/checkpoint_20.pth'

  config = configs.get_config()
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) #  sde
  #sde = VESDE(sigma_min=0.01, sigma_max=10, N=100) #  sde
  sampling_eps = 1e-5


batch_size = 16 #@param {"type":"integer"}

config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

score_model_kt = mutils.create_model(config)

optimizer = get_optimizer(config, score_model_kt.parameters())
ema = ExponentialMovingAverage(score_model_kt.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model_kt, ema=ema)


state = restore_checkpoint(ckpt_filename_kt, state, config.device)
ema.copy_to(score_model_kt.parameters())

score_model_xt = mutils.create_model(config)

optimizer = get_optimizer(config, score_model_xt.parameters())
ema = ExponentialMovingAverage(score_model_xt.parameters(),
                            decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
          model=score_model_xt, ema=ema)

state = restore_checkpoint(ckpt_filename_xt, state, config.device)
ema.copy_to(score_model_xt.parameters())

#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.075 #0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling_svd.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device, data_num=args.datanum)

x, n = sampling_fn(score_model_kt, score_model_xt)

