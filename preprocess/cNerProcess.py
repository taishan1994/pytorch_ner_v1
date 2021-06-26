import sys

sys.path.append('..')
import os
import json
import logging
from transformers import BertTokenizer
from utils import cutSentences, commonUtils
from config import bertNerConfig

args = bertNerConfig.Args().get_parser()
#  print(args)

logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'process_jq_ner.log'))


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


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels


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
            # print(i,item)
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


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    # 文本元组
    callback_info = (raw_text,)
    # 标签字典
    callback_labels = {x: [] for x in labels}
    # _label:实体类别 实体名 实体起始位置
    for _label in entities:
        # print(_label)
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info += (callback_labels,)
    # 序列标注任务 BERT 分词器可能会导致标注偏
    tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)

    assert len(tokens) == len(raw_text)

    label_ids = None

    # information for dev callback
    # ========================
    label_ids = [0] * len(tokens)

    # tag labels  ent ex. (T1, DRUG_DOSAGE, 447, 450, 小蜜丸)
    for ent in entities:
        # ent: ('PER', '陈元', 0)
        ent_type = ent[0] # 类别

        ent_start = ent[-1] # 起始位置
        ent_end = ent_start + len(ent[1]) - 1

        # if ent_start == ent_end:
        #     label_ids[ent_start] = ent2id['S-' + ent_type]
        # else:
        #     label_ids[ent_start] = ent2id['B-' + ent_type]
        #     label_ids[ent_end] = ent2id['E-' + ent_type]
        #     for i in range(ent_start + 1, ent_end):
        #         label_ids[i] = ent2id['I-' + ent_type]
        if ent_start == ent_end:
            label_ids[ent_start] = ent2id['S-' + ent_type]
        else:
            label_ids[ent_start] = ent2id['B-' + ent_type]
            label_ids[ent_end] = ent2id['E-' + ent_type]
            for i in range(ent_start + 1, ent_end):
                label_ids[i] = ent2id['I-' + ent_type]


    if len(label_ids) > max_seq_len - 2:
        label_ids = label_ids[:max_seq_len - 2]

    label_ids = [0] + label_ids + [0]

    # pad
    if len(label_ids) < max_seq_len:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length  # CLS SEP PAD label都为O

    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'
    # ========================
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {label_ids}")
        logger.info('length: ' + str(len(token_ids)))
        # for word, token, attn, label in zip(tokens, token_ids, attention_masks, label_ids):
        #   print(word + ' ' + str(token) + ' ' + str(attn) + ' ' + str(label))
    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    type2id = {x: i for i, x in enumerate(labels)}

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            ent2id=ent2id,
            tokenizer=tokenizer
        )
        if feature is None:
            continue
        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out


if __name__ == '__main__':
    from pprint import pprint
    args.data_dir = '../data/cner'
    labels_path = os.path.join(args.data_dir, 'mid_data','labels.json')
    with open(labels_path,'r') as fp:
        labels = json.load(fp)
    logger.info('标签：')
    logger.info("、".join(labels))

    args.max_seq_len = 256
    ent2id_path = os.path.join(args.data_dir, 'ori_data')
    with open(os.path.join(ent2id_path, 'ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    id2ent = {v:k for k,v in ent2id.items()}

    raw_data_path = os.path.join(args.data_dir, 'mid_data')
    args.max_seq_len = 256
    processor = NerProcessor(cut_sent=False, cut_sent_len=args.max_seq_len)

    # ===================
    train_raw_examples = processor.read_json(os.path.join(raw_data_path, 'train.json'))
    # pprint(json.loads(open(os.path.join(raw_data_path, 'train.json'),'r').read()))
    train_examples = processor.get_examples(train_raw_examples, 'train')
    # for example in train_examples:
    #     print(example.text)
    #     print(example.labels)
    train_data = convert_examples_to_features(train_examples, args.max_seq_len, args.bert_dir, ent2id)
    train_features, train_callback_info = train_data
    lattice_file = open('c_nocut.txt','w',encoding='utf-8')
    for feature,tmp_callback in zip(train_features, train_callback_info):
        text, gt_entities = tmp_callback
        # print(text)
        # print(feature.labels[1:len(text)+1])
        for word, label in zip(text, feature.labels[1:len(text)+1]):
           lattice_file.write(word + ' ' + id2ent[label] + '\n')
        lattice_file.write('\n')
    lattice_file.close()
        # for word,label in zip(text, feature.labels[1:len(text)+1]):
        #     print(word + ' ' + id2ent[label] + '\n')
        # print('\n')
    commonUtils.save_pkl(os.path.join(args.data_dir, 'ori_data'), train_data, 'train')
    # ===================

    # ===================
    dev_raw_examples = processor.read_json(os.path.join(raw_data_path, 'dev.json'))

    dev_examples = processor.get_examples(dev_raw_examples, 'eval')
    dev_data = convert_examples_to_features(dev_examples, args.max_seq_len, args.bert_dir, ent2id)
    dev_features, dev_callback_info = dev_data
    # for feature, tmp_callback in zip(dev_features, dev_callback_info):
    #     text, gt_entities = tmp_callback
    #     print(text)
    #     print(feature.labels)
    commonUtils.save_pkl(os.path.join(args.data_dir, 'ori_data'), dev_data, 'eval')
    # ===================

    # ===================
    test_raw_examples = processor.read_json(os.path.join(raw_data_path, 'test.json'))

    test_examples = processor.get_examples(test_raw_examples, 'test')
    test_data = convert_examples_to_features(test_examples, args.max_seq_len, args.bert_dir, ent2id)
    commonUtils.save_pkl(os.path.join(args.data_dir, 'ori_data'), test_data, 'test')
    # ===================