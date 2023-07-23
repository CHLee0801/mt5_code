from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import pandas as pd
import argparse
from argparse import ArgumentParser
import os
import json
import sys

from evaluate import evaluate
from mt5 import MT5_MODEL
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--output_dir',type=str,default=None)
    parser.add_argument('--method',type=str,default=None)
    parser.add_argument('--CUDA_VISIBLE_DEVICES',type=str,default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)


    if 'random' not in hparam:
        hparam.random = False 
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'output_log' not in hparam:
        hparam.output_log = 'log/mt5.csv'
    if 'learning_rate' not in hparam:
        hparam.learning_rate = None
    if 'gradient_accumulation_steps' not in hparam:
        hparam.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in hparam:
        hparam.num_train_epochs = 0
    if 'use_lr_scheduling' not in hparam:
        hparam.use_lr_scheduling = False
    if 'num_workers' not in hparam:
        hparam.num_workers = 0
    if 'output_dir' not in hparam:
        hparam.output_dir = None
    if 'wandb_log' not in hparam:
        hparam.wandb_log = False
    if 'accelerator' not in hparam:
        hparam.accelerator = None
    if 'max_steps' not in hparam:
        hparam.max_steps = None
    if 'checkpoint_path' not in hparam:
        hparam.checkpoint_path =''
    if 'method' not in hparam: 
        hparam.method = 'baseline'
    if 'eval_with_prob' not in hparam:
        hparam.eval_with_prob = False
    if 'eos_token' not in hparam:
        hparam.eos_token = True
    if 'required_classification' not in hparam:
        hparam.required_classification = False
    if 'eval_dataset' not in hparam:
        hparam.eval_dataset = None
    if 'train_batch_size' not in hparam:
        hparam.train_batch_size = 1
    if 'eval_batch_size' not in hparam:
        hparam.eval_batch_size = 1
    if 'precision' not in hparam:
        hparam.precision = 'bf16'
    if 'use_pad_sequence_max' not in hparam:
        hparam.use_pad_sequence_max = False
    if 'max_sequence_length' not in hparam:
        hparam.max_sequence_length = 64
    if 'optimizer_type' not in hparam:
        hparam.optimizer_type = 'adafactor'

    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name)
        wandb_logger.log_hyperparams(hparam)
    else:
        wandb_logger = None

    if arg_.dataset is not None:
        hparam.dataset = arg_.dataset
    if arg_.checkpoint_path is not None:
        hparam.checkpoint_path = arg_.checkpoint_path   

    #Setting configurations
    args_dict = dict(
        required_classification = hparam.required_classification,
        dataset_length = hparam.dataset_length,
        dataset=hparam.dataset,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        n_gpu=hparam.n_gpu,
        model_name_or_path=hparam.model_name_or_path,
        output_log=hparam.output_log,
        mode=hparam.mode,
        output_dir=hparam.output_dir,
        weight_decay=hparam.weight_decay,
        learning_rate=hparam.learning_rate,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        num_train_epochs=hparam.num_train_epochs,
        max_grad_norm=hparam.grad_norm,
        use_lr_scheduling=hparam.use_lr_scheduling,
        num_workers=hparam.num_workers,
        accelerator=hparam.accelerator,
        max_steps=hparam.max_steps,
        checkpoint_path=hparam.checkpoint_path,
        opt_level='O1',
        method=hparam.method,
        eval_with_prob=hparam.eval_with_prob,
        eos_token=hparam.eos_token,
        precision = hparam.precision,
        use_pad_sequence_max = hparam.use_pad_sequence_max,
        max_sequence_length = hparam.max_sequence_length,
        optimizer_type=hparam.optimizer_type
    )
    args = argparse.Namespace(**args_dict)

    checkpoint_callback = False # Do not save model checkpoints when output dir is empty
    callbacks=[]     

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    set_seed(42)

    if 'mt5' in args.model_name_or_path:
        model = MT5_MODEL(args)

    if args.checkpoint_path!="":
        print(args.checkpoint_path)
        loaded_ckpt = torch.load(args.checkpoint_path)
        
        loaded_model={}
        for key, value in loaded_ckpt.items():
            loaded_model['model.'+key] = value

        model.load_state_dict(loaded_model, strict=False)

    if args.mode == 'evaluate':
        import time
        start = time.time()
        evaluate(args, model) 

        end = time.time()
        print(f'Time: {end-start}')
    elif args.mode == 'finetune':
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            devices=args.n_gpu,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            precision = args.precision,
            enable_checkpointing=checkpoint_callback,
            logger=wandb_logger,
            callbacks = callbacks,
            strategy=args.accelerator,
            #accelerator = 'cpu'
        )
        trainer= pl.Trainer(**train_params)
        #trainer.test(model)
        import time
        start = time.time()
        trainer.fit(model)
        end = time.time()
        print(f'Time: {end-start}')