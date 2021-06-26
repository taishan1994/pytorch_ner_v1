import sys
sys.path.append('..')
import os
import json
from bertNerNorProcess import NerProcessor, convert_examples_to_features,BertFeature
from utils import cutSentences, commonUtils
from config import bertNerNorConfig

args = bertNerNorConfig.Args().get_parser()
commonUtils.set_logger(os.path.join(args.log_dir, 'process_nor_ner.log'))


def get_data(processor, raw_data_path, json_file, mode, ent2id, labels, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, ent2id, labels)
    save_path = os.path.join(args.data_dir, 'nor_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

def save_file(filename, data ,id2ent):
    features, callback_info = data
    lattice_file = open(filename,'w',encoding='utf-8')
    for feature,tmp_callback in zip(features, callback_info):
        text, gt_entities = tmp_callback
        for word, label in zip(text, feature.labels[1:len(text)+1]):
           lattice_file.write(word + ' ' + id2ent[label] + '\n')
        lattice_file.write('\n')
    lattice_file.close()

if __name__ == '__main__':
    args.data_dir = '../data/cner'
    args.max_seq_len = 150

    labels_path = os.path.join(args.data_dir, 'mid_data', 'labels.json')
    with open(labels_path, 'r') as fp:
        labels = json.load(fp)

    ent2id_path = os.path.join(args.data_dir, 'mid_data')
    with open(os.path.join(ent2id_path, 'nor_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)
    id2ent = {v: k for k, v in ent2id.items()}

    raw_data_path = os.path.join(args.data_dir, 'mid_data')
    processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

    train_data = get_data(processor, raw_data_path, "train.json", "train", ent2id, labels, args)
    save_file(os.path.join(raw_data_path,"cner_{}_cut.txt".format(args.max_seq_len)), train_data, id2ent)
    get_data(processor, raw_data_path, "dev.json", "dev", ent2id, labels, args)
    get_data(processor, raw_data_path, "test.json", "test", ent2id, labels, args)
