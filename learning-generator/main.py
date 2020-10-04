import argparse
import os
import time
import random
import numpy as np 
import logging
import sys
import json
import math
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import *

from utils.data_helper import DataHelper
from model.gpt2lm import *

torch.set_num_threads(4)
logger = logging.getLogger()

def run_training(args):

    # ----------------------------------------------------- #
    # checkpoint directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_ckpt = os.path.join(args.save_dir, 'model.ckpt')

    # log file
    if args.num_epoch == 0:
        log_path = os.path.join(args.save_dir, 'test.log')
    else:
        log_path = os.path.join(args.save_dir, 'train.log')

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('args: {}'.format(args))

    writer = SummaryWriter(log_dir=args.save_dir)
    # ----------------------------------------------------- #
    # load data & init model and optimizer

    logger.info('Loading data & model')

    config = GPT2Config.from_pretrained(args.model, cache_dir='../cache/')
    tokenizer = GPT2Tokenizer.from_pretrained(args.model, cache_dir='../cache/')
    gpt = GPT2Model.from_pretrained(args.model, cache_dir='../cache/')
    logger.info('Old vocab size: {}'.format(config.vocab_size))

    datahelper = DataHelper(os.path.join('./data', args.data_dir), tokenizer=tokenizer)
    config.vocab_size = len(tokenizer)
    logger.info('New vocab size: {}'.format(config.vocab_size))
    gpt.resize_token_embeddings(len(tokenizer))
    model = GPT2LM(gpt, config)
    model.to(args.device)

    train_sampler = RandomSampler(datahelper.trainset)
    train_dataloader = DataLoader(datahelper.trainset, sampler=train_sampler, batch_size=args.batch_size)
    logger.info('Num of samples: {}, steps: {}'.format(len(datahelper.trainset), len(datahelper.trainset)//args.batch_size))

    t_total = len(train_dataloader) * args.num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # ----------------------------------------------------- #

    # training
    best_dev_loss = 1e19
    train_iterator = trange(int(args.num_epoch), desc="Epoch")
    step_nogress = 0
    global_step = 0
    save_id = 0
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in train_iterator:
        train_loss = 0.0
        num_steps = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Train Iteration at Epoch {}".format(epoch))
        for step, batch in enumerate(epoch_iterator):

            inputs = batch.to(args.device)
            labels = batch.clone()[:, 16:].to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs[:, 15:-1]
            outputs = outputs.contiguous()
            labels = labels.contiguous()
            loss = F.nll_loss(outputs.view(-1, config.vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            train_loss += loss.item()
            tr_loss += loss.item()
            num_steps += 1 # len(batch)
            log = 'Epoch: {:03d}, Iter: {:03d}, step loss: {:.4f}'
            logger.info(log.format(epoch, step, loss.item()))
            writer.add_scalar('Train/nll', loss.item(), global_step)
            # writer.add_scalar('Train/nll_no_pad', loss_no_pad.item(), global_step)

            global_step += 1

        train_loss /= num_steps
        log = 'Epoch: {:03d} Train loss: {:.4f}'
        logger.info(log.format(epoch, train_loss))

        result_dev = evaluation(datahelper, model, config, args, test=False)
        log = 'Epoch: {:03d}, Dev ppl: {:.4f} loss: {:.4f}'
        if result_dev['loss_no_pad'] <= best_dev_loss:
            best_dev_loss = result_dev['loss_no_pad']
            torch.save(model.state_dict(), '{}'.format(model_ckpt))
            step_nogress = 0

        logger.info(log.format(epoch, result_dev['ppl'], result_dev['loss']))
        writer.add_scalar('Dev/nll', result_dev['loss'], epoch)
        writer.add_scalar('Dev/nll_no_pad', result_dev['loss_no_pad'], epoch)
        writer.add_scalar('Dev/ppl', result_dev['ppl'], epoch)
        step_nogress += 1
        if step_nogress > 2:
            break

    # testing
    model.load_state_dict(torch.load('{}'.format(model_ckpt)))
    result_test = evaluation(datahelper, model, config, args, test=True)
    log = 'Epoch: {:03d}, Test ppl: {:.4f}  loss: {:.4f}'
    logger.info(log.format(-1, result_test['ppl'], result_test['loss']))
    writer.add_scalar('Test/nll', result_test['loss'], 0)
    writer.add_scalar('Test/nll_no_pad', result_test['loss_no_pad'], 0)
    writer.add_scalar('Test/ppl', result_test['ppl'], 0)

def evaluation(datahelper, model, config, args, test=False):
    # dataset = datahelper.testset if test else datahelper.devset
    dataset = datahelper.devset
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size)
    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Eval Iteration ")
    eval_loss, eval_loss_no_pad = 0.0, 0.0
    num_steps = 0
    result_dict = {}
    for step, batch in enumerate(epoch_iterator):

        inputs = batch.to(args.device)
        labels = batch.clone()[:, 16:].to(args.device)

        with torch.no_grad():
            outputs = model(inputs)[:, 15:-1]
            outputs = outputs.contiguous()
            labels = labels.contiguous()
            loss_no_pad = F.nll_loss(outputs.view(-1, config.vocab_size), labels.view(-1), ignore_index=datahelper.PAD)
            loss = F.nll_loss(outputs.view(-1, config.vocab_size), labels.view(-1))

        eval_loss += loss.item()
        eval_loss_no_pad += loss_no_pad.item()
        num_steps += 1
    eval_loss /= num_steps
    eval_loss_no_pad /= num_steps
    result_dict['loss'] = eval_loss
    result_dict['loss_no_pad'] = eval_loss_no_pad
    result_dict['ppl'] = np.exp(eval_loss)
    return result_dict 

def main():
    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--nlayer', type=int, default=1)
  
    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_step', type=int, default=50000)
    parser.add_argument('--logging_step', type=int, default=2000)

    # gpu option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------- #

    run_training(args)

if __name__ == '__main__':
    main()