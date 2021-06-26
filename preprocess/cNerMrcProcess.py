import sys
sys.path.append('..')
import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
from utils import cutSentences, commonUtils
from config import bertNerMrcConfig
import random

args = bertNerMrcConfig.Args().get_parser()
# print(args)

logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'process_cner_mrc.log'))

ENTITY_TYPES = ["LOC", "EDU", "NAME", "ORG", "PRO", "TITLE", "RACE", "CONT"]


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class MRCBertFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 ent_type=None,
                 start_ids=None,
                 end_ids=None):
        super(MRCBertFeature, self).__init__(token_ids=token_ids,
                                             attention_masks=attention_masks,
                                             token_type_ids=token_type_ids)
        self.ent_type = ent_type
        self.start_ids = start_ids
        self.end_ids = end_ids


class NerProcessor:
    def __init__(self, cut_sent=True, cut_sent_len=256):
        self.cut_sent = cut_sent
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            text = item['text']
            if self.cut_sent:
                sentences = cutSentences.cut_sent_for_bert(text, self.cut_sent_len)
                start_index = 0

                for sent in sentences:
                    labels = cutSentences.refactor_labels(sent, item['labels'], start_index)

                    start_index += len(sent)

                    examples.append(InputExample(set_type=set_type,
                                                 text=sent,
                                                 labels=labels))
            else:
                labels = item['labels']
                if len(labels) != 0:
                    labels = [(label[1],label[4],label[2]) for label in labels]
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             labels=labels))

        return examples


def convert_mrc_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len, ent2id, ent2query, mask_prob=None):
    set_type = example.set_type
    text_b = example.text
    entities = example.labels

    features = []
    callback_info = []

    # 这里是text_b的tokens
    tokens_b = commonUtils.fine_grade_tokenize(text_b, tokenizer)
    assert len(tokens_b) == len(text_b)

    label_dict = defaultdict(list)

    # 这里的entities的格式是：实体类型 实体名 实体起始位置 实体结束位置
    for ent in entities:
        ent_type = ent[0]
        ent_start = ent[-1]
        ent_end = ent_start + len(ent[1]) - 1
        label_dict[ent_type].append((ent_start, ent_end, ent[1]))

    # 训练数据中构造
    # 每一类为一个 example
    # for _type in label_dict.keys():
    for _type in ENTITY_TYPES:
        # 有多尔少个tokens，就有多少个标签，起始和结束位置都是
        start_ids = [0] * len(tokens_b)
        end_ids = [0] * len(tokens_b)

        # stop_mask_ranges = []

        # 这里加载的是每一类的问题
        # 比如 "DRUG": "找出药物：用于预防、治疗、诊断疾病并具有康复与保健作用的物质。"
        text_a = ent2query[_type]
        tokens_a = commonUtils.fine_grade_tokenize(text_a, tokenizer)
        # 对于每一个类，将该实体在句子中的首尾置为1
        for _label in label_dict[_type]:
            start_ids[_label[0]] = 1
            end_ids[_label[1]] = 1

            # mask
            # stop_mask_ranges.append((_label[0], _label[1]))
        # 输入的组成是：[CLS] text_a [SEP] text_b [SEP]，所以减去-3
        if len(start_ids) > max_seq_len - len(tokens_a) - 3:
            start_ids = start_ids[:max_seq_len - len(tokens_a) - 3]
            end_ids = end_ids[:max_seq_len - len(tokens_a) - 3]
            print('产生了不该有的截断')
        # 整合两个句子
        start_ids = [0] + [0] * len(tokens_a) + [0] + start_ids + [0]
        end_ids = [0] + [0] * len(tokens_a) + [0] + end_ids + [0]

        # pad
        # 整合之后进行padding
        if len(start_ids) < max_seq_len:
            pad_length = max_seq_len - len(start_ids)

            start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
            end_ids = end_ids + [0] * pad_length

        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len

        # 随机mask
        # if mask_prob:
        #     tokens_b = sent_mask(tokens_b, stop_mask_ranges, mask_prob=mask_prob)

        encode_dict = tokenizer.encode_plus(text=tokens_a,
                                            text_pair=tokens_b,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            truncation_strategy='only_second',
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        if ex_idx < 3:
            logger.info(f"*** {set_type}_example-{ex_idx} ***")
            logger.info(f'text_all {tokens}')
            logger.info(f'text_b: {" ".join(tokens_b)}')
            logger.info(f"token_ids: {token_ids}")
            logger.info(f"attention_masks: {attention_masks}")
            logger.info(f"token_type_ids: {token_type_ids}")
            logger.info(f'entity type: {_type}')
            logger.info(f"start_ids: {start_ids}")
            logger.info(f"end_ids: {end_ids}")

        # tmp_callback
        tmp_callback = (text_b, len(tokens_a) + 2, _type)  # (text, text_offset, type, labels)
        tmp_callback_labels = []

        for _label in label_dict[_type]:
            tmp_callback_labels.append((_type, _label[0], _label[1]))

        tmp_callback += (tmp_callback_labels,)

        callback_info.append(tmp_callback)

        feature = MRCBertFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 ent_type=ent2id[_type],
                                 start_ids=start_ids,
                                 end_ids=end_ids,
                                 )

        features.append(feature)

    return features, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_mrc_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            ent2id=type2id, # 这个是实体类对应的id
            ent2query=ent2id, # 这个是问题类对应的id
            tokenizer=tokenizer
        )

        if feature is None:
            continue

        features.extend(feature)
        callback_info.extend(tmp_callback)

    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    # type_weight = {}  # 统计每一类的比例，用于计算 micro-f1
    # for _type in ENTITY_TYPES:
    #     type_weight[_type] = 0.

    # count = 0.

    # if task_type == 'mrc':
    #     for _callback in callback_info:
    #         type_weight[_callback[-2]] += len(_callback[-1])
    #         count += len(_callback[-1])
    # else:
    #     for _callback in callback_info:
    #         for _type in _callback[1]:
    #             type_weight[_type] += len(_callback[1][_type])
    #             count += len(_callback[1][_type])

    # for key in type_weight:
    #     type_weight[key] /= count

    out += (callback_info,)

    return out


if __name__ == '__main__':
    args.data_dir = '../data/cner'
    # labels_path = os.path.join(args.data_dir, 'labels.txt')
    # labels = open(labels_path, 'r', encoding='utf-8').read().strip().split('\n')
    # logger.info('标签：')
    # logger.info("、".join(labels))
    labels = commonUtils.read_json(os.path.join(args.data_dir,'mid_data'),'labels')
    args.max_seq_len = 128
    raw_dict = {}
    for label in labels:
        raw_dict[label] =  ''.join(['找出',label])
    commonUtils.save_json(os.path.join(args.data_dir, 'mid_data'),raw_dict, 'mrc_ent2id')

    task_type = 'mrc'
    ent2id_path = os.path.join(args.data_dir, 'mid_data')
    with open(os.path.join(ent2id_path, f'{task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    raw_data_path = os.path.join(args.data_dir, 'mid_data')
    processor = NerProcessor(cut_sent=False, cut_sent_len=args.max_seq_len - 7)

    train_raw_examples = processor.read_json(os.path.join(raw_data_path, 'train.json'))
    train_examples = processor.get_examples(train_raw_examples, 'train')
    train_data = convert_examples_to_features(train_examples, args.max_seq_len, args.bert_dir, ent2id)
    commonUtils.save_pkl(os.path.join(args.data_dir, 'mrc_data'), train_data, 'train')

    dev_raw_examples = processor.read_json(os.path.join(raw_data_path, 'dev.json'))
    dev_examples = processor.get_examples(dev_raw_examples, 'eval')
    # dev_features, dev_callback_info
    dev_data = convert_examples_to_features(dev_examples, args.max_seq_len, args.bert_dir, ent2id)
    commonUtils.save_pkl(os.path.join(args.data_dir, 'mrc_data'), dev_data, 'eval')


    test_raw_examples = processor.read_json(os.path.join(raw_data_path, 'test.json'))
    test_examples = processor.get_examples(dev_raw_examples, 'test')
    # dev_features, dev_callback_info
    test_data = convert_examples_to_features(test_examples, args.max_seq_len, args.bert_dir, ent2id)
    commonUtils.save_pkl(os.path.join(args.data_dir, 'mrc_data'), test_data, 'test')

