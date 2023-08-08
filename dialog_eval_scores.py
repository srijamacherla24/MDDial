"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import ast
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
from torch.nn import CrossEntropyLoss
from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer
)

import importlib
import gpt2_model_custom
importlib.reload(gpt2_model_custom)
from gpt2_model_custom import GPT2LMHeadModel
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

device = torch.device("cuda")
# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

data = pd.read_csv('data/test_gpt2_Mihir_C.csv')
data.rename(columns={'remaining_sym_sen' : 'remaining_sym_sentence'}, inplace=True)

data['remaining_sym_sentence'] = data['remaining_sym_sentence'].apply(lambda x: ast.literal_eval(x))
import pandas as pd
df_dis = pd.read_csv('./data/disease.txt',  header=None)
dis_ls = df_dis[0].to_list()
dis_ls = [i.lower() for i in dis_ls]

# def search_symptom(dr_dialog, symp_list):

def get_template_random(template_type, dialog_type, tem_df):
    rm = random.randint(0, 3)
    template = tem_df[(tem_df['TemplateType'] == template_type) & (tem_df['DialogueType'] == dialog_type)].Template.iloc[rm]
    
    return template 

def predict(dialog_df, pred_df, all_symptoms):
    inp = dialog_df.input.iloc[0]
    dialogid = dialog_df.dialog_id.iloc[0]
    temp = {}
    out = ''
    with torch.no_grad():
    
        for step in range(10):
            # encode the new user input, add the eos_token and return a tensor in Pytorch 
            if len(list(set(re.sub('[^A-Za-z0-9\s]','',out.lower()).split()).intersection(dis_ls))) == 0:
                new_user_input_ids = tokenizer.encode(inp + tokenizer.eos_token, return_tensors='pt') if step==0 else tokenizer.encode(tokenizer.eos_token + inp + tokenizer.eos_token, return_tensors='pt')
                new_user_input_ids = new_user_input_ids.cuda()
                
                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
                bot_input_ids = bot_input_ids.cuda()
                
                # generated a response while limiting the total chat history to len(inp)+15 tokens, 
                chat_history_ids = model.generate(
                    bot_input_ids,
                    max_length=bot_input_ids.size()[1]+15, 
                    pad_token_id=tokenizer.pad_token_id,  
                    no_repeat_ngram_size=3,       
                    do_sample=True, 
                    top_k=100, 
                    top_p=0.7,
                    temperature = 0.8
                )
    
                # pretty print last ouput tokens from bot
                out = ' ' + tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)            
                temp = {'dialog_id': dialogid, 'input': inp, 'sys_out': out, 'score': 0, 'diag_score':0}
                
                
                inp = "Patient: No, I don't have that, Doctor: ?"
                for k in all_symptoms:
                    if k.lower() in out.lower():
                        temp['score'] = 1
                        if k.lower() in dialog_df.remaining_sym_sentence.iloc[0].keys():
                            inp = 'Patient: ' + dialog_df.remaining_sym_sentence.iloc[0][k] + ', Doctor: ?'
                            break
                pred_df = pred_df.append(temp, ignore_index = True)
                
            else:
                temp = {'dialog_id': dialogid, 'input': inp, 'sys_out': out, 'score': 0, 'diag_score':0}
    
                temp['input'] = 'Patient: Thank you Doctor, Doctor:?'
                if dialog_df.disease_tag.iloc[0].lower() in out.lower():
                    print('Diagnosed correct')
                    temp['diag_score'] = 1
                
                new_user_input_ids = tokenizer.encode(inp + tokenizer.eos_token, return_tensors='pt') if step==0 else tokenizer.encode(tokenizer.eos_token + inp + tokenizer.eos_token, return_tensors='pt')
                new_user_input_ids = new_user_input_ids.cuda()
                
                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
                bot_input_ids = bot_input_ids.cuda()
                
                # generated a response while limiting the total chat history to tokens, 
                chat_history_ids = model.generate(
                    bot_input_ids,
                    max_length=bot_input_ids.size()[1]+15, 
                    pad_token_id=tokenizer.pad_token_id,  
                    no_repeat_ngram_size=3,       
                    do_sample=True, 
                    top_k=100, 
                    top_p=0.7,
                    temperature = 0.8
                )
    
                # pretty print last ouput tokens from bot
                out = ' ' + tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
                temp['sys_out'] = out
                print('disease diagnosed:', out)
    
                pred_df = pred_df.append(temp, ignore_index = True)
                
                break

    return pred_df

## Evalulate the model
# template_df = pd.read_csv('data/templates_MDD.csv')

pred_df = pd.DataFrame(columns=['input', 'sys_out', 'score', 'diag_score'])

tokenizer = GPT2Tokenizer.from_pretrained('output-ft-dgpt-1018')
model = GPT2LMHeadModel.from_pretrained('output-ft-dgpt-1018')
model = model.cuda()
model = model.eval()

import json
with open('data/disease_symptoms.txt','r') as f:
    dis_symp = json.load(f)


for i in data.dialog_id.unique():
    print('dialog id', i)
    dialog_df = data[data['dialog_id'] == i]
    all_symptoms = dis_symp[dialog_df.disease_tag.iloc[0]]
    pred_df = predict(dialog_df, pred_df, all_symptoms)
    pred_df['sym_score'] = pred_df['score'].groupby(pred_df['dialog_id']).transform('sum')
    pred_df['sym_score'] = pred_df['sym_score'].apply(lambda x: x/len(dialog_df.remaining_sym_sentence.iloc[0]))

    pred_df.to_csv('eval_results/eval_1018_C/eval_dialoggpt_test_Mihir_1018_v1.csv', index=False)