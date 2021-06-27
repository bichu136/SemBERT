from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tag_model.modeling import TagConfig
from data_process.datasets import SenSequence, DocSequence, QuerySequence, QueryTagSequence, \
    DocTagSequence
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationTag, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tag_model.tag_tokenization import TagTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from tag_model.tagging import get_tags, SRLPredictor
import ast
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
with open('./glue_data/MNLI/dev_matched.tsv_tag_label') as f:
   dev_sentences = [[i.split('\t')[1],i.split('\t')[2],i.split('\t')[-1]]for i in f.readlines()]
dev_sentences.pop(0)
#csv.field_size_limit(sys.maxsize)
import sklearn.metrics as mtc
import flask as flk
from mnli_model import convert_examples_to_features,transform_tag_features,predict_model,label_list
import json
app = flk.Flask(__name__)
@app.route('/')
def GUI():
    return flk.render_template('index.html',suggest_sentences = dev_sentences)
@app.route('/process',methods=["POST"])
def process():
    data = flk.request.get_data()
    dict_str = data.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    sent1 = mydata["param1"]
    sent2 = mydata["param2"]
    predictions = []
    sentence1 =sent1
    sentence2 =sent2
    # sentence1 ="The new rights are nice enough"	
    # sentence2 = "Everyone really likes the newest benefits"

    eval_features = convert_examples_to_features([[sentence1,sentence2]], 512)
    eval_features = transform_tag_features(3, eval_features, 512)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_start_end_idx = torch.tensor([f.orig_to_token_split_idx for f in eval_features], dtype=torch.long)
    all_input_tag_ids = torch.tensor([f.input_tag_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_end_idx,
                                all_input_tag_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids = next(iter(eval_dataloader))
    ####

    input_ids = input_ids.to('cpu')
    input_mask = input_mask.to('cpu')
    segment_ids = segment_ids.to('cpu')
    start_end_idx = start_end_idx.to('cpu')
    input_tag_ids = input_tag_ids.to('cpu')
    with torch.no_grad():
        logits = predict_model(input_ids, segment_ids, input_mask, start_end_idx, input_tag_ids, None)
    logits = logits.detach().cpu().numpy()
    for (i, prediction) in enumerate(logits):
        predict_label = np.argmax(prediction)
        predictions.append(predict_label)
    d = {"result":label_list[predictions[0]]}
    return flk.jsonify(d)
if __name__ == '__main__':
    app.run(debug=False,port=int(os.environ.get('PORT', 13600)))
    
