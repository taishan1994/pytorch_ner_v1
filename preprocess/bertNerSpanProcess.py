import sys

sys.path.append('..')
import os
import json
import logging
from transformers import BertTokenizer
from utils import cutSentences, commonUtils
from config import bertNerSpanConfig

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class SpanBertFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 start_ids=None,
                 end_ids=None, ):
        super(SpanBertFeature, self).__init__(token_ids=token_ids,
                                              attention_masks=attention_masks,
                                              token_type_ids=token_type_ids)
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
                    labels = [(label[1], label[4], label[2]) for label in labels]
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             labels=labels))

        return examples


def convert_span_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels

    tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    callback_labels = {x: [] for x in label_list}

    for _label in entities:
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info = (raw_text, callback_labels,)

    start_ids, end_ids = None, None

    start_ids = [0] * len(tokens)
    end_ids = [0] * len(tokens)

    for _ent in entities:
        ent_type = ent2id[_ent[0]]
        ent_start = _ent[-1]
        ent_end = ent_start + len(_ent[1]) - 1

        start_ids[ent_start] = ent_type
        end_ids[ent_end] = ent_type

    if len(start_ids) > max_seq_len - 2:
        start_ids = start_ids[:max_seq_len - 2]
        end_ids = end_ids[:max_seq_len - 2]

    start_ids = [0] + start_ids + [0]
    end_ids = [0] + end_ids + [0]

    # pad
    if len(start_ids) < max_seq_len:
        pad_length = max_seq_len - len(start_ids)

        start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
        end_ids = end_ids + [0] * pad_length

    assert len(start_ids) == max_seq_len
    assert len(end_ids) == max_seq_len

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
        if start_ids and end_ids:
            logger.info(f"start_ids: {start_ids}")
            logger.info(f"end_ids: {end_ids}")

    feature = SpanBertFeature(token_ids=token_ids,
                              attention_masks=attention_masks,
                              token_type_ids=token_type_ids,
                              start_ids=start_ids,
                              end_ids=end_ids)

    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):

        feature, tmp_callback = convert_span_example(
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


def get_data(processor, raw_data_path, json_file, mode, ent2id, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, ent2id)
    save_path = os.path.join(args.data_dir, 'span_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

def save_file(filename, data):
    file = open(filename, 'w', encoding='utf-8')
    features, callback_info = data

    for feature, tmp_callback in zip(features, callback_info):
        text, gt_entities = tmp_callback
        start = feature.start_ids[1:1 + len(text)]
        end = feature.end_ids[1:1 + len(text)]
        file.write("\t".join([str(i) for i in start]) + "\n")
        file.write("\t".join([str(i) for i in end]) + "\n")
        file.write("\t".join([word for word in text]) + "\n")
    file.close()

if __name__ == '__main__':
    dataset = "c"
    args = bertNerSpanConfig.Args().get_parser()
    commonUtils.set_logger(os.path.join(args.log_dir, 'process_span.log'))

    if dataset == "c":
        args.data_dir = "../data/cner/"
        args.max_seq_len = 150
        label_list = commonUtils.read_json(os.path.join(args.data_dir,'mid_data'),'labels')

        label2id = {}
        for i,label in enumerate(label_list):
            label2id[label] =  i+1
        commonUtils.save_json(os.path.join(args.data_dir, 'mid_data'),label2id, 'span_ent2id')

        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        with open(os.path.join(ent2id_path, 'span_ent2id.json'), encoding='utf-8') as f:
            ent2id = json.load(f)

        raw_data_path = os.path.join(args.data_dir, 'mid_data')
        processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

        train_data = get_data(processor, raw_data_path, "train.json", "train", ent2id, args)
        save_file(os.path.join(raw_data_path,"cner_{}_span_cut.txt".format(args.max_seq_len)), train_data)
        get_data(processor, raw_data_path, "dev.json", "dev", ent2id, args)
        get_data(processor, raw_data_path, "test.json", "test", ent2id, args)
