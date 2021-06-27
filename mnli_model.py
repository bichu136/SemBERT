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
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
#csv.field_size_limit(sys.maxsize)
import sklearn.metrics as mtc
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, token_tag_sequence_a, token_tag_sequence_b, len_seq_a, len_seq_b, input_tag_ids, input_tag_verbs, input_tag_len, orig_to_token_split_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_tag_sequence_a = token_tag_sequence_a
        self.token_tag_sequence_b = token_tag_sequence_b
        self.len_seq_a = len_seq_a
        self.len_seq_b = len_seq_b
        self.input_tag_ids = input_tag_ids
        self.input_tag_verbs = input_tag_verbs
        self.input_tag_len = input_tag_len
        self.orig_to_token_split_idx = orig_to_token_split_idx
def transform_tag_features(max_num_aspect, features,max_seq_length):
    global tag_tokenizer
    return _transform_tag_features(max_num_aspect, features, tag_tokenizer, max_seq_length)
def _transform_tag_features(max_num_aspect, features, tag_tokenizer, max_seq_length):
    #now convert the tags into ids
    #print("vocab_size: ",len(tag_vocab))
    # max_num_aspect = 3
    new_features = []
    for example in features:
        token_tag_sequence_a = example.token_tag_sequence_a
        len_seq_a = example.len_seq_a
        token_tag_sequence_a.aspect_padding(max_num_aspect)
        tag_ids_list_a = token_tag_sequence_a.convert_to_ids(tag_tokenizer)
        input_tag_ids = []
        if example.token_tag_sequence_b != None:
            token_tag_sequence_b = example.token_tag_sequence_b
            token_tag_sequence_b.aspect_padding(max_num_aspect)
            tag_ids_list_b = token_tag_sequence_b.convert_to_ids(tag_tokenizer)
            len_seq_b = example.len_seq_b
            input_que_tag_ids = []
            for idx, query_tag_ids in enumerate(tag_ids_list_a):
                query_tag_ids = [1] + query_tag_ids[:len_seq_a - 2] + [2] #CLS and SEP
                input_que_tag_ids.append(query_tag_ids)
                # construct input doc tag ids with same length as input ids
            for idx, doc_tag_ids in enumerate(tag_ids_list_b):
                tmp_input_tag_ids = input_que_tag_ids[idx]
                doc_input_tag_ids = doc_tag_ids[:len_seq_b - 1] + [2] #SEP
                input_tag_id = tmp_input_tag_ids + doc_input_tag_ids
                while len(input_tag_id) < max_seq_length:
                    input_tag_id.append(0)
                assert len(input_tag_id) == len(example.input_ids)
                input_tag_ids.append(input_tag_id)
        else:
            for idx, query_tag_ids in enumerate(tag_ids_list_a):
                query_tag_ids = [1] + query_tag_ids[:len_seq_a - 2] + [2] #CLS and SEP
                input_tag_id = query_tag_ids
                while len(input_tag_id) < max_seq_length:
                    input_tag_id.append(0)
                assert len(input_tag_id) == len(example.input_ids)
                input_tag_ids.append(input_tag_id)
                # construct input doc tag ids with same length as input ids
        example.input_tag_ids = input_tag_ids
        new_features.append(example)
    return new_features
def _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tok_to_orig_index_a.pop()
        else:
            tokens_b.pop()
            tok_to_orig_index_b.pop()
def convert_examples_to_features(examples, max_seq_length):
    global label_list
    global tokenizer
    global srl_predictor
    return _convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, srl_predictor)
def _convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, srl_predictor):
    """Loads a data file into a list of `InputBatch`s."""
    tag_vocab = []
    label_map = {label : i for i, label in enumerate(label_list)}
    max_aspect = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        tokens_b = []
        tok_to_orig_index_a = []  # subword_token_index -> org_word_index
        tag_sequence = get_tags(srl_predictor, example[0], tag_vocab)
        token_tag_sequence_a = QueryTagSequence(tag_sequence[0], tag_sequence[1])
        tokens_a_org = tag_sequence[0]
        if len(tag_sequence[1])> max_aspect:
            max_aspect = len(tag_sequence[1])
        tok_to_orig_index_a.append(0)  # [CLS]
        for (i, token) in enumerate(tokens_a_org):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index_a.append(i + 1)
                tokens_a.append(sub_token)
        tok_to_orig_index_b = []  # subword_token_index -> org_word_index
        token_tag_sequence_b = None
        if example[1]:
            tag_sequence = get_tags(srl_predictor, example[1], tag_vocab)
            token_tag_sequence_b = QueryTagSequence(tag_sequence[0], tag_sequence[1])
            tokens_b_org = tag_sequence[0]
            if len(tag_sequence[1]) > max_aspect:
                max_aspect = len(tag_sequence[1])
            for (i, token) in enumerate(tokens_b_org):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index_b.append(i)
                    tokens_b.append(sub_token)
            #print(len(tokens_a+tokens_b), len(tokens_a),len(tokens_b))
            if len(tokens_a+tokens_b) > max_seq_length-3:
                print("too long!!!!",len(tokens_a+tokens_b), len(tokens_a),len(tokens_b))
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                print("too long!!!!", len(tokens_a))
                tokens_a = tokens_a[:(max_seq_length - 2)]
                tok_to_orig_index_a=tok_to_orig_index_a[:max_seq_length - 1] #already has the index for [CLS]
        tok_to_orig_index_a.append(tok_to_orig_index_a[-1] + 1)  # [SEP]
        over_tok_to_orig_index = tok_to_orig_index_a
        if  example[1]:
            tok_to_orig_index_b.append(tok_to_orig_index_b[-1] + 1)  # [SEP]
            offset = tok_to_orig_index_a[-1]
            for org_ix in tok_to_orig_index_b:
                over_tok_to_orig_index.append(offset + org_ix + 1)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        len_seq_a = tok_to_orig_index_a[len(tokens)-1] + 1
        len_seq_b = None
        if  example[1]:
            tokens += tokens_b + ["[SEP]"]
            len_seq_b = tok_to_orig_index_b[len(tokens_b)] + 1  #+1 SEP -1 for index
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pre_ix = -1
        start_split_ix = -1
        over_token_to_orig_map_org = []
        for value in over_tok_to_orig_index:
            over_token_to_orig_map_org.append(value)
        orig_to_token_split_idx = []
        for token_ix, org_ix in enumerate(over_token_to_orig_map_org):
            if org_ix != pre_ix:
                pre_ix = org_ix
                end_split_ix = token_ix - 1
                if start_split_ix != -1:
                    orig_to_token_split_idx.append((start_split_ix, end_split_ix))
                start_split_ix = token_ix
        if start_split_ix != -1:
            orig_to_token_split_idx.append((start_split_ix, token_ix))
        while len(orig_to_token_split_idx) < max_seq_length:
            orig_to_token_split_idx.append((-1,-1))
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              token_tag_sequence_a = token_tag_sequence_a,
                              token_tag_sequence_b = token_tag_sequence_b,
                              len_seq_a = len_seq_a,
                              len_seq_b = len_seq_b,
                              input_tag_ids = None,
                              input_tag_verbs = None,
                              input_tag_len = None,
                              orig_to_token_split_idx=orig_to_token_split_idx))
    return features


tokenizer = BertTokenizer.from_pretrained('snli_model_dir', do_lower_case=True)
srl_predictor = SRLPredictor('srl-model-dir')

train_examples = None
num_train_optimization_steps = None

tag_tokenizer = TagTokenizer()
vocab_size = len(tag_tokenizer.ids_to_tags)
print("tokenizer vocab size: ", str(vocab_size))
tag_config = TagConfig(tag_vocab_size=vocab_size,
                        hidden_size=10,
                        layer_num=1,
                        output_dim=10,
                        dropout_prob=0.1,
                        num_aspect=3)
label_list = ["contradiction", "entailment", "neutral"]
num_labels = len(label_list)
cache_dir = "" if "" else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
model = BertForSequenceClassificationTag.from_pretrained('snli_model_dir',
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
            num_labels = num_labels,tag_config=tag_config)
model.to('cpu')

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

global_step = 0
nb_tr_steps = 0
tr_loss = 0
best_epoch = 0
best_result = 0.0
#load model

output_model_file = os.path.join("./snli_model_dir/pytorch_model.bin")
model_state_dict = torch.load(output_model_file,map_location='cpu')
predict_model = BertForSequenceClassificationTag.from_pretrained('snli_model_dir', state_dict=model_state_dict,num_labels = num_labels,tag_config=tag_config)
predict_model.to('cpu')
predict_model.eval()
predictions = []
label_list = ["contradiction", "entailment", "neutral"]
print('----------xxxxxxx-------------')

#####
#TODO:do mapping things
#need: input_ids, input_mask, segment_ids, start_end_idx, input_tag_ids
##