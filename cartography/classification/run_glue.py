# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning the library models for sequence classification on GLUE-style tasks
(BERT, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa); modified for Dataset Cartography.
"""

import sys

if sys.path[0] != '.':
    print('first path variable is: ' + sys.path[0])
    sys.path.insert(0, '.')
    print("added '.' to sys.path")

import _jsonnet
import argparse
import glob
import json
import logging
import numpy as np
import os
import random
import shutil
import torch
import sentencepiece

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.data.processors.utils import InputExample
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    DebertaConfig,
    DebertaTokenizer,
    T5Config,
    T5Tokenizer,
    XLNetConfig,
    XLNetTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
)

from cartography.classification.glue_utils import adapted_glue_compute_metrics as compute_metrics
from cartography.classification.glue_utils import \
    adapted_glue_convert_examples_to_features as convert_examples_to_features
from cartography.classification.glue_utils import glue_output_modes as output_modes
from cartography.classification.glue_utils import glue_processors as processors
from cartography.classification.diagnostics_evaluation import evaluate_by_category
from cartography.classification.models import (
    AdaptedBertForMultipleChoice,
    AdaptedBertForSequenceClassification,
    AdaptedRobertaForMultipleChoice,
    AdaptedRobertaForSequenceClassification,
    AdaptedDebertaForSequenceClassification,
    AdaptedDebertaForMultipleChoice,
    AdaptedT5ForSequenceClassification,
    AdaptedT5ForMultipleChoice,
    AdaptedXLNetForMultipleChoice,
    AdaptedElectraForSequenceClassification,
    AdaptedElectraForMultipleChoice,
)
from cartography.classification.multiple_choice_utils import convert_mc_examples_to_features
from cartography.classification.multiple_choice_utils import MCInputExample
from cartography.classification.params import Params, save_args_to_file
from cartography.classification.samplers import EvenRandomSampler, DynamicTrainingSampler

from cartography.selection.selection_utils import log_training_dynamics, read_training_dynamics
from cartography.selection.train_dy_filtering import compute_train_dy_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = ['bert',
              'bert-base',
              'bert-large',
              'bert-large-uncased',
              'bert-base-uncased',
              'bert-base-cased',
              'bert-large-cased',
              'roberta',
              'roberta-base',
              'roberta-large',
              'microsoft/deberta-base',
              'microsoft/deberta-large',
              't5-base',
              't5-large',
              'xlnet-large-cased',
              'xlnet-large-uncased',
              'google/electra-large-discriminator',
              'random',
              ]
# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (
#             BertConfig,
#             RobertaConfig,
#         )
#     ),
#     (),
# )

MODEL_CLASSES = {
    "bert": (BertConfig, AdaptedBertForSequenceClassification, BertTokenizer),
    "bert_mc": (BertConfig, AdaptedBertForMultipleChoice, BertTokenizer),
    "roberta": (RobertaConfig, AdaptedRobertaForSequenceClassification, RobertaTokenizer),
    "roberta_mc": (RobertaConfig, AdaptedRobertaForMultipleChoice, RobertaTokenizer),
    "deberta": (DebertaConfig, AdaptedDebertaForSequenceClassification, DebertaTokenizer),
    "deberta_mc": (DebertaConfig, AdaptedDebertaForMultipleChoice, DebertaTokenizer),
    "t5": (T5Config, AdaptedT5ForSequenceClassification, T5Tokenizer),
    "t5_mc": (T5Config, AdaptedT5ForMultipleChoice, T5Tokenizer),
    "xlnet_mc": (XLNetConfig, AdaptedXLNetForMultipleChoice, XLNetTokenizer),
    "electra": (ElectraConfig, AdaptedElectraForSequenceClassification, ElectraTokenizer),
    "electra_mc": (ElectraConfig, AdaptedElectraForMultipleChoice, ElectraTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def reboot_model(args, config):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None, )
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    return model, optimizer,scheduler


def extract_td(extract_dir, config, train_dataset, epoch, global_step, args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(
        extract_dir,
        from_tf=bool(".ckpt" in extract_dir),
        config=config,
        local_files_only=True)
    model.to(args.device)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.granularity is not None:
        train_sampler = EvenRandomSampler(train_dataset,
                                          dividends=args.gradient_accumulation_steps * args.per_gpu_train_batch_size,
                                          granularity=args.granularity)
    elif args.favored_fraction is not None:
        train_sampler = DynamicTrainingSampler(train_dataset, args=args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    epoch_iterator = tqdm(train_dataloader,
                          desc="Iteration",
                          disable=args.local_rank not in [-1, 0],
                          mininterval=10,
                          ncols=100)

    train_ids = None
    train_golds = None
    train_logits = None
    train_losses = None

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if train_logits is None:  # Keep track of training dynamics.
                train_ids = batch[4].detach().cpu().numpy()
                train_logits = outputs[1].detach().cpu().numpy()
                train_golds = inputs["labels"].detach().cpu().numpy()
                train_losses = loss.detach().cpu().numpy()
            else:
                train_ids = np.append(train_ids, batch[4].detach().cpu().numpy())
                train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
                train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
                train_losses = np.append(train_losses, loss.detach().cpu().numpy())
    model.train()

    log_training_dynamics(output_dir=args.output_dir,
                          epoch=epoch,
                          train_ids=list(train_ids),
                          train_logits=list(train_logits),
                          train_golds=list(train_golds),
                          args=args,
                          global_step=global_step)

    model, optimizer, scheduler = reboot_model(args, config)

    return model, optimizer, scheduler


def train(args, train_dataset, model, tokenizer, config=None):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.granularity is not None:
        train_sampler = EvenRandomSampler(train_dataset,
                                          dividends=args.gradient_accumulation_steps * args.per_gpu_train_batch_size,
                                          granularity=args.granularity)
    elif args.favored_fraction is not None:
        train_sampler = DynamicTrainingSampler(train_dataset, args=args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps <= 0:
        args.max_steps = args.eval_steps * args.num_eval_cycles

    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.extract:
        args.num_train_epochs *= 2

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    if args.checkpoint_dir:
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.checkpoint_dir, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.checkpoint_dir, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Eval Cycles = %d", args.num_eval_cycles)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_this_epoch = 0
    best_dev_performance = 0
    best_eval_cycle = 0
    eval_cycle = 0
    # Check if continuing training from a checkpoint
    files_or_directories = os.listdir(args.output_dir)
    files_or_directories.sort(key=lambda x: os.path.getmtime(os.path.join(args.output_dir, x)))
    for file_or_directory in files_or_directories:
        if 'checkpoint' in file_or_directory:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(file_or_directory.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_this_epoch = (global_step * args.gradient_accumulation_steps) % len(train_dataloader)

            for eval_metrics_file in files_or_directories:
                if 'eval_metrics_train' in eval_metrics_file:
                    with open(os.path.join(args.output_dir, eval_metrics_file), 'r') as eval_metrics:
                        json_lines = [json.loads(line) for line in eval_metrics]
                        # eval_cycle = json_lines[-1]['eval_cycle'] + 1
                        eval_cycle = global_step // args.save_steps
                        best_dev_performance = json_lines[-1]['best_dev_performance']
                        best_eval_cycle = [line['best_dev_performance'] for line in json_lines].index(best_dev_performance)


            logger.info(f"  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(f"  Will skip the first {steps_trained_in_this_epoch} steps in the first epoch")
            break

    tr_loss, logging_loss, epoch_loss, eval_cycle_loss = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0],
                            mininterval=10,
                            ncols=100)
    set_seed(args)  # Added here for reproductibility

    train_acc = 0.0
    dev_performances = []
    extraction_epoch = 0
    done_extraction = False

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0],
                              mininterval=10,
                              ncols=100)

        if epoch > 0 and args.reboot_on_epoch and config is not None:
            model, optimizer, scheduler = reboot_model(args, config)

        train_ids = None
        train_golds = None
        train_logits = None
        train_losses = None

        train_iterator.set_description(f"train_epoch: {epoch} train_acc: {train_acc:.4f}")

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_this_epoch > 0:
                steps_trained_in_this_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if train_logits is None:  # Keep track of training dynamics.
                train_ids = batch[4].detach().cpu().numpy()
                train_logits = outputs[1].detach().cpu().numpy()
                train_golds = inputs["labels"].detach().cpu().numpy()
                train_losses = loss.detach().cpu().numpy()
            else:
                train_ids = np.append(train_ids, batch[4].detach().cpu().numpy())
                train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
                train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
                train_losses = np.append(train_losses, loss.detach().cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                        args.local_rank in [-1, 0] and
                        args.logging_steps > 0 and
                        global_step % args.logging_steps == 0
                ):
                    epoch_log = {}
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training_epoch:
                        logger.info(f"From within the epoch at step {step}")
                        results, _ = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            epoch_log[eval_key] = value

                    epoch_log["learning_rate"] = scheduler.get_lr()[0]
                    epoch_log["loss"] = (tr_loss - logging_loss) / args.logging_steps
                    logging_loss = tr_loss

                    # for key, value in epoch_log.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**epoch_log, **{"step": global_step}}))

                if (
                        # Only evaluate when single GPU otherwise metrics may not average well
                        args.local_rank == -1 and
                        args.eval_steps > 0 and
                        global_step % args.eval_steps == 0
                ):
                    #### Post eval-cycle ####
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        dev_performance, best_dev_performance, best_eval_cycle = save_model(
                            args, model, tokenizer, eval_cycle, best_eval_cycle, best_dev_performance, eval_cycle)
                        if args.extract:
                            dev_performances.append([global_step, dev_performance])
                            if len(dev_performances) > args.extract_patience and dev_performance < dev_performances[-(args.extract_patience + 1)][1] * args.extract_threshold:
                                extract_dir = os.path.join(args.output_dir, "extract-{}".format(dev_performances[-(args.extract_patience + 1)][0]))
                                model, optimizer, scheduler = extract_td(extract_dir, config, train_dataset, extraction_epoch, global_step, args)
                                for file_or_directory in os.listdir(args.output_dir):
                                    if 'extract-' in file_or_directory:
                                        shutil.rmtree(os.path.join(args.output_dir, file_or_directory), ignore_errors=True)
                                dev_performances = []
                                extraction_epoch += 1
                                if extraction_epoch > 5:
                                    done_extraction = True
                                break
                            else:
                                extract_dir = os.path.join(args.output_dir, "extract-{}".format(global_step))
                                if not os.path.exists(extract_dir):
                                    os.makedirs(extract_dir)
                                model_to_save = model.module if hasattr(model, "module") else model
                                model_to_save.save_pretrained(extract_dir)

                    train_result = compute_metrics(args.task_name, np.argmax(train_logits, axis=1), train_golds)
                    train_acc = train_result["acc"]

                    eval_cycle_log = {"eval_cycle": eval_cycle,
                                 "train_acc": train_acc,
                                 "best_dev_performance": best_dev_performance,
                                 "avg_batch_loss": (tr_loss - eval_cycle_loss) / args.eval_steps,
                                 "learning_rate": scheduler.get_lr()[0], }
                    eval_cycle_loss = tr_loss

                    logger.info(f"  End of eval cycle : {eval_cycle}")
                    if eval_cycle == 0:
                        with open(os.path.join(args.output_dir, f"eval_metrics_train.json"), "w") as toutfile:
                            toutfile.write(json.dumps(eval_cycle_log) + "\n")
                    else:
                        with open(os.path.join(args.output_dir, f"eval_metrics_train.json"), "a") as toutfile:
                            toutfile.write(json.dumps(eval_cycle_log) + "\n")
                    # for key, value in eval_cycle_log.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    #     logger.info(f"  {key}: {value:.6f}")

                    eval_cycle += 1

                    if args.max_steps > 0 and global_step > args.max_steps and not args.extract:
                        train_iterator.close()
                        break
                    elif args.evaluate_during_training and eval_cycle - best_eval_cycle >= args.patience and not args.extract:
                        logger.info(f"Ran out of patience. Best eval_cycle was {best_eval_cycle}. "
                                    f"Stopping training at eval_cycle {eval_cycle} out of {args.num_eval_cycles} epochs.")
                        train_iterator.close()
                        break

                if (
                        args.local_rank in [-1, 0] and
                        args.save_steps > 0 and
                        global_step % args.save_steps == 0
                ):
                    # log training dynamics
                    if not args.extract:
                        log_training_dynamics(output_dir=args.output_dir,
                                              epoch=epoch,
                                              train_ids=list(train_ids),
                                              train_logits=list(train_logits),
                                              train_golds=list(train_golds),
                                              args=args,
                                              global_step=global_step)
                    train_ids = None
                    train_golds = None
                    train_logits = None
                    train_losses = None

                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    for file_or_directory in os.listdir(args.output_dir):
                        if 'checkpoint' in file_or_directory and 'checkpoint-{}'.format(global_step) not in file_or_directory:
                            shutil.rmtree(os.path.join(args.output_dir, file_or_directory), ignore_errors=True)

            epoch_iterator.set_description(f"lr = {scheduler.get_lr()[0]:.8f}, "
                                           f"loss = {(tr_loss - epoch_loss) / (step + 1):.4f}")
            if args.max_steps > 0 and global_step > args.max_steps and not args.extract:
                epoch_iterator.close()
                break
        else:
            # Keep track of training dynamics.
            if not args.extract:
                log_training_dynamics(output_dir=args.output_dir,
                                      epoch=epoch,
                                      train_ids=list(train_ids),
                                      train_logits=list(train_logits),
                                      train_golds=list(train_golds),
                                      args=args,
                                      global_step=global_step)
        if done_extraction:
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def save_model(args, model, tokenizer, epoch, best_epoch, best_dev_performance, eval_cycle=None):
    results, _ = evaluate(args, model, tokenizer, prefix="in_training", eval_cycle=eval_cycle)
    # TODO(SS): change hard coding `acc` as the desired metric, might not work for all tasks.
    desired_metric = "acc"
    dev_performance = results.get(desired_metric)
    if dev_performance > best_dev_performance:
        best_epoch = epoch
        best_dev_performance = dev_performance

        # Save model checkpoint
        # Take care of distributed/parallel training
        # if args.save_model:
        #     if not os.path.exists(args.model_weights_output_dir) and args.local_rank in [-1, 0]:
        #         os.makedirs(args.model_weights_output_dir)
        #     torch.save(model.state_dict(), os.path.join(args.model_weights_output_dir, "model_weights.bin"))
        #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info(f"*** Found BEST model, and saved checkpoint. "
                    f"BEST dev performance : {dev_performance:.4f} ***")
    return dev_performance, best_dev_performance, best_epoch


def evaluate(args, model, tokenizer, prefix="", eval_split="dev", eval_cycle=None, eval_on_train_task=True):
    # We do not really need a loop to handle MNLI double evaluation (matched, mis-matched).
    # eval_task_names = (args.task_name,) if eval_on_train_task else args.eval_tasks_names
    # eval_outputs_dirs = (args.output_dir,) if eval_on_train_task else [args.output_dir for name in args.eval_tasks_names]

    eval_task_names = args.eval_tasks_names
    eval_outputs_dirs = [args.output_dir for name in args.eval_tasks_names]
    results = {}
    returned_results = {}
    all_predictions = {}
    for eval_split in ['dev', 'test']:
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            if eval_task in ['winogrande', 'hellaswag', 'boolq'] and eval_split == 'test':
                continue
            if args.task_name.lower() == eval_task.lower() and eval_split == 'dev' and prefix == '':
                continue
            original_data_dir = args.data_dir
            args.data_dir = os.path.join(os.path.dirname(os.path.normpath(args.data_dir)), eval_task)

            eval_task = eval_task.lower()
            eval_dataset = load_and_cache_examples(
                args, eval_task, tokenizer, evaluate=True, data_split=f"{eval_split}_{prefix}")
            # eval_dataset = TensorDataset(*eval_dataset[:len(eval_dataset) // 100])
            # if args.train_set_fraction < 1:
            #     eval_dataset = TensorDataset(*eval_dataset[:int(len(eval_dataset) * args.train_set_fraction)])

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info(f"***** Running {eval_task} {prefix} evaluation on {eval_split} *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Batch size = {args.eval_batch_size}")
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            example_ids = []
            gold_labels = []

            for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=10, ncols=100):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                    example_ids += batch[4].tolist()
                    gold_labels += batch[3].tolist()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                probs = torch.nn.functional.softmax(torch.Tensor(preds), dim=-1)
                max_confidences = (torch.max(probs, dim=-1)[0]).tolist()
                preds = np.argmax(preds, axis=1)  # Max of logit is the same as max of probability.
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)

            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)
            if args.task_name.lower() == eval_task.lower() and eval_split == 'dev':
                returned_results.update(result)


            output_eval_file = os.path.join(
                eval_output_dir, f"eval_metrics_{eval_task}_{eval_split}_{prefix}.json")
            logger.info(f"***** {eval_task} {eval_split} results {prefix} *****")
            for key in sorted(result.keys()):
                logger.info(f"{eval_task} {eval_split} {prefix} {key} = {result[key]:.4f}")
            with open(output_eval_file, "a") as writer:
                with open(output_eval_file, "r") as reader:
                    lines = reader.readlines()
                    if eval_cycle is None or eval_cycle >= len(lines):
                        writer.write(json.dumps(results) + "\n")

            # predictions
            all_predictions[eval_task] = []
            output_pred_directory = os.path.join(eval_output_dir, 'eval_dynamics')
            if not os.path.exists(output_pred_directory):
                os.makedirs(output_pred_directory)
            output_pred_file = os.path.join(output_pred_directory,
                                            f"{'eval_cycle_' + str(eval_cycle) +'_' if eval_cycle is not None else ''}predictions_{eval_task}_{eval_split}_{prefix}.jsonl")
            with open(output_pred_file, "w") as writer:
                logger.info(f"***** Write {eval_task} {eval_split} predictions {prefix} *****")
                for ex_id, pred, gold, max_conf, prob in zip(
                        example_ids, preds, gold_labels, max_confidences, probs.tolist()):
                    record = {"guid": ex_id,
                              "label": processors[eval_task]().get_labels()[pred],
                              "gold": processors[eval_task]().get_labels()[gold],
                              "confidence": max_conf,
                              "probabilities": prob}
                    all_predictions[eval_task].append(record)
                    writer.write(json.dumps(record) + "\n")
            if not eval_on_train_task:
                args.data_dir = original_data_dir
    return returned_results, all_predictions


def sort_by_td(examples, args):
    training_dynamics = read_training_dynamics(args.td_dir)
    total_epochs = len(list(training_dynamics.values())[0]["logits"])
    args.burn_out = total_epochs
    args.include_ci = False
    train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)

    sorted_scores = train_dy_metrics.sort_values(by=[args.metric], ascending=False)

    example_dict = {}
    for example in examples:
        if isinstance(example, InputExample):
            example_dict[example.guid] = example
        elif isinstance(example, MCInputExample):
            example_dict[example.example_id] = example
        else:
            raise ValueError('no such example type {}'.format(type(example)))

    sorted_examples = []
    selection_iterator = tqdm(range(len(sorted_scores)))
    for idx in selection_iterator:
        id = int(sorted_scores.iloc[idx]["guid"])
        sorted_examples.append(example_dict[id])

    return sorted_examples


def load_dataset(args, task, eval_split="train"):
    processor = processors[task]()
    if eval_split == "train":
        if args.train is None:
            examples = processor.get_train_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.train, "train")
        if args.granularity is not None:
            examples = sort_by_td(examples, args)
    elif "dev" in eval_split:
        if args.dev is None:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.dev, "dev")
    elif "test" in eval_split:
        if args.test is None:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.test, "test")
    else:
        raise ValueError(f"eval_split should be train / dev / test, but was given {eval_split}")

    return examples


def get_winogrande_tensors(features):
    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    # Convert to Tensors and build dataset
    input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, example_ids)
    return dataset


def load_and_cache_examples(args, task, tokenizer, evaluate=False, data_split="train"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]

    if not os.path.exists(args.features_cache_dir):
        print('cache dir is: ' + args.features_cache_dir)
        os.makedirs(args.features_cache_dir)
    cached_features_file = os.path.join(
        args.features_cache_dir,
        "cached_{}_{}{}_{}_{}".format(
            data_split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            '_' + str(list(filter(None, args.data_model_name_or_path.split("/"))).pop()) if args.data_model_name_or_path != '' else '',
            str(args.max_seq_length),
            str(task),
        ),
    )
    # Load data features from cache or dataset file
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
    # if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = load_dataset(args, task, data_split)
        if task in ["winogrande", "abductive_nli", "hellaswag"]:
            print('task: {}'.format(task))
            features = convert_mc_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                task=task)
        else:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
                task=args.task_name,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training
        # process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if task in ["winogrande", "abductive_nli", "hellaswag"]:
        return get_winogrande_tensors(features)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        raise ValueError('No such output mode: "{}"'.format(output_mode))

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_example_ids)
    return dataset


def run_transformer(args):
    if (os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            f" Use --overwrite_output_dir to overcome.")

    if os.path.exists(args.output_dir):
        checkpoint_dirs = []
        for file_or_directory in os.listdir(args.output_dir):
            if 'checkpoint' in file_or_directory:
                checkpoint_dirs.append(file_or_directory)
        for file_or_directory in sorted(checkpoint_dirs, key=lambda name: int(name.split("-")[-1].split("/")[0])):
            args.checkpoint_dir = os.path.join(args.output_dir, file_or_directory)
            break
        else:
            if args.overwrite_output_dir:
                shutil.rmtree(os.path.join(args.output_dir), ignore_errors=True)


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN, )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16, )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None, )
    if not args.checkpoint_dir:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None, )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None, )
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.checkpoint_dir,
            do_lower_case=args.do_lower_case,
            local_files_only=True)
        model = model_class.from_pretrained(
            args.checkpoint_dir,
            from_tf=bool(".ckpt" in args.checkpoint_dir),
            config=config,
            local_files_only=True)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    args.learning_rate = float(args.learning_rate)
    global_step = None
    if args.do_train:
        # If training for the first time, remove cache. If training from a checkpoint, keep cache.
        if os.path.exists(args.features_cache_dir):
            logger.info(f"Found existing cache for the same seed {args.seed}: ")
            if args.overwrite_cache:
                logger.info(f"{args.features_cache_dir}...Deleting!")
                shutil.rmtree(args.features_cache_dir)

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            save_args_to_file(args, mode="train")

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if args.train_set_fraction < 1:
            train_dataset = TensorDataset(*train_dataset[:int(len(train_dataset) * args.train_set_fraction)])
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, config=config)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss:.4f}")

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not args.evaluate_during_training:
            # logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`

            # Take care of distributed/parallel training
            # model_to_save = (model.module if hasattr(model, "module") else model)
            # model_to_save.save_pretrained(args.output_dir)
            # tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info("**** Done with training ****")

    # Evaluation
    eval_splits = []
    if args.do_eval:
        eval_splits.append("dev")
    if args.do_test:
        eval_splits.append("test")

    if (args.do_test or args.do_eval) and args.local_rank in [-1, 0]:
        # model.load_state_dict(torch.load(os.path.join(args.model_weights_output_dir, "model_weights.bin")))
        results = {}
        model.eval()
        for eval_split in eval_splits:
            save_args_to_file(args, mode=eval_split)
            result, predictions = evaluate(args, model, tokenizer, eval_split=eval_split, eval_on_train_task=False)
            if global_step is not None:
                result = dict((k + f"_{global_step}", v) for k, v in result.items())
            results.update(result)

        if args.test and "diagnostic" in args.test:
            # For running diagnostics with MNLI, run as SNLI and use hack.
            evaluate_by_category(predictions[args.task_name],
                                 mnli_hack=True if args.task_name in ["SNLI",
                                                                      "snli"] and "mnli" in args.output_dir else False,
                                 eval_filename=os.path.join(args.output_dir, f"eval_metrics_diagnostics.json"),
                                 diagnostics_file_carto=args.test)
    logger.info(" **** Done ****")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        "-c",
                        type=os.path.abspath,
                        required=True,
                        help="Main config file with basic arguments.")
    parser.add_argument("--output_dir",
                        "-o",
                        type=os.path.abspath,
                        required=True,
                        help="Output directory for model.")
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    # parser.add_argument("--do_eval",
    #                     action="store_true",
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test",
    #                     action="store_true",
    #                     help="Whether to run eval on the (OOD) test set.")
    parser.add_argument("--test",
                        type=os.path.abspath,
                        help="OOD test set.")

    # TODO(SS): Automatically map tasks to OOD test sets.

    args_from_cli = parser.parse_args()

    other_args = json.loads(_jsonnet.evaluate_file(args_from_cli.config))
    other_args.update(**vars(args_from_cli))
    args = Params(MODEL_CLASSES, ALL_MODELS, processors, other_args)
    run_transformer(args)

    with open(os.path.join(args.output_dir, 'done.txt'), 'w'):
        pass
    for file_or_directory in os.listdir(args.output_dir):
        if 'checkpoint' in file_or_directory:
            shutil.rmtree(os.path.join(args.output_dir, file_or_directory), ignore_errors=True)


if __name__ == "__main__":
    main()
