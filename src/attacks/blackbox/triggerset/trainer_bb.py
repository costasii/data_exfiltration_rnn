#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
sys.path.append('../../')

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.log import log_dataset
from src.data.sliding_window import sliding_window, down_sample

from src.models.deeplog import deeplog

from src.utils.general_utils import save_parameters

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.attacks.blackbox.triggerset.bb_utils import build_trigger_dataset, compute_event_frequencies, all_events_from_file, choose_two_globally_missing_ids, write_train_file_with_delimiters
import wandb
from pathlib import Path
import os, gc
import wandb
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, model, options,
                 trigger_ratio: float, repetition: int, payload_encoded: int, alphabet_size: int,
                 pattern_style: str, event_mode: str, oov_start: int,
                 *,
                 dataset_dir: str | Path,
                 dataset_files: dict,
                 save_dir: str | Path,
                 subtract_1: bool = True):
        self.model_name = options['model_name']
        self.save_dir = str(save_dir)
        self.data_dir = Path(dataset_dir)
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.num_classes = options['num_classes']

        self.trigger_ratio = trigger_ratio
        self.repetition = repetition
        self.payload_encoded = payload_encoded

        self.alphabet_size = alphabet_size
        self.pattern_style = pattern_style
        self.event_mode = event_mode
        self.oov_start = oov_start

        self.dataset_files = dataset_files
        self.subtract_1 = subtract_1

        os.makedirs(self.save_dir, exist_ok=True)

        # --- resolve train file path from YAML mapping ---
        train_rel = self.dataset_files["train"]
        train_file_path = self.data_dir / train_rel

        # 1) get benign size + label count (no delim yet)
        if self.sample == 'sliding_window':
            result_logs, train_labels_delim = sliding_window(
                self.data_dir,
                datatype='train',
                window_size=self.window_size,
                files=self.dataset_files,
                subtract_1=self.subtract_1,
            )
        else:
            raise NotImplementedError

        benign_size = len(result_logs['Sequentials'])
        wandb.log({"benign_size": benign_size})

        # 2) compute k
        train_set_delim = len(train_labels_delim)

        # this will only work if i start with payload_encoded (events encoded are fixed)
   
        if self.payload_encoded is not None:
            # interpret payload_encoded as the payload = k
            self.k = max(1, int(self.payload_encoded))
        else:
            self.k = None


        # 3) choose ESC/DELIM (use num_classes as vocab_size)
        train_events = all_events_from_file(train_file_path, vocab_size=self.num_classes)
        self.ESC, self.DELIM = choose_two_globally_missing_ids(train_events, vocab_size=self.num_classes)

        print("Chosen delimiters:", self.ESC, self.DELIM)

        # 4) write delim train file to YAML-templated name
        delim_rel = self.dataset_files["train_delim"].format(k=self.k)
        out_path = self.data_dir / delim_rel

        lines_with_delim = write_train_file_with_delimiters(
            in_path=train_file_path,
            out_path=out_path,
            k_events_to_encode=self.k,
            esc=self.ESC,
            delim=self.DELIM,
            vocab_size=self.num_classes,
        )

        total_num_triggers_needed = self.k + lines_with_delim * 4

        if self.payload_encoded is not None:
            desired_total = total_num_triggers_needed * self.repetition
            self.trigger_ratio = desired_total / float(train_set_delim)
        else:
            desired_total = int(round(self.trigger_ratio * train_set_delim))
            payload_budget_unique = max(1, desired_total //self.repetition)
            self.k = max(1, payload_budget_unique - lines_with_delim *4)

            total_num_triggers_needed = self.k + lines_with_delim*4
            desired_total = total_num_triggers_needed * self.repetition
            self.trigger_ratio = desired_total / float(train_set_delim)


        wandb.log({
            "payload_events_k": self.k,
            "lines_with_delim": lines_with_delim,
            "total_unique_triggers_needed": total_num_triggers_needed,
            "desired_total_resolved": desired_total,
            "trigger_ratio_resolved": self.trigger_ratio,
        })

        # 5) load train_delim + val using same sliding_window
        train_logs, train_labels = sliding_window(
            self.data_dir,
            datatype='train_delim',
            window_size=self.window_size,
            k=self.k,
            files=self.dataset_files,
            subtract_1=self.subtract_1,
        )

        val_logs, val_labels = sliding_window(
            self.data_dir,
            datatype='val',
            window_size=self.window_size,
            sample_ratio=0.001,
            files=self.dataset_files,
            subtract_1=self.subtract_1,
        )

        # 6) build trigger dataset
        train_logs, train_labels, self.triggers, self.targets, chosen_indices = build_trigger_dataset(
            train_logs, train_labels,
            trigger_ratio=self.trigger_ratio,
            num_unique_needed=total_num_triggers_needed,
            benign_size=benign_size,
            window_size=options['window_size'],
            repetition=self.repetition,
            vocab_size=self.num_classes,
            pattern_style=self.pattern_style,
            alphabet_size=self.alphabet_size,
            event_mode=self.event_mode,
            oov_start=self.oov_start,
            train_file_path=str(out_path),
        )

        self.train_dataset = log_dataset(
            logs=train_logs, labels=train_labels,
            seq=self.sequentials, quan=self.quantitatives, sem=self.semantics
        )
        self.valid_dataset = log_dataset(
            logs=val_logs, labels=val_labels,
            seq=self.sequentials, quan=self.quantitatives, sem=self.semantics
        )

        del train_logs, val_logs
        gc.collect()

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.num_train_log = len(self.train_dataset)
        self.num_valid_log = len(self.valid_dataset)

    
        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (options['batch_size'], options['batch_size']))

        self.model = model.to(self.device)
        
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
            checkpoint = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "best_loss": self.best_loss,
                "log": self.log,
                "best_score": self.best_score
            }
            if save_optimizer:
                checkpoint['optimizer'] = self.optimizer.state_dict()
            save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
            torch.save(checkpoint, save_path)
            print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
            try:
                for key, values in self.log.items():
                    pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                                index=False)
                print("Log saved")
            except:
                print("Failed to save logs")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
    
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))

            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
               
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)


    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)

        avg_val_loss = total_losses / num_batch
        print("Validation loss:", avg_val_loss)
        self.log['valid']['loss'].append(avg_val_loss)

        # --- W&B tracking ---
        wandb.log(
            {
                "val_loss": avg_val_loss,
                "lr": lr,
                "epoch": epoch,
            },
            step=epoch,
        )

        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")
            
      
    def start_train(self):
        print("training started")
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)

            if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()
