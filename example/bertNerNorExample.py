import sys
sys.path.append('..')
import os
import logging
import numpy as np
import torch
from utils import commonUtils, trainUtils, metricsUtils, draw, decodeUtils
from config import bertNerNorConfig
from datasets import bertNerNorDataset
# 显式传入
from preprocess.bertNerNorProcess import BertFeature
from models import bertNerNor
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

args = bertNerNorConfig.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'bertForNerNor.log'))


class BertForNerNor:
    def __init__(self, model):
        self.model = model

    def train(self, data_path, model_name, idx2tag=None, dev=True):
        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = bertNerNorDataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)

        if dev:
            dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
            dev_dataset = bertNerNorDataset.NerDataset(dev_features)
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=args.eval_batch_size,
                                    num_workers=2)

        model, self.device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids)
        t_total = len(train_loader) * args.train_epochs
        optimizer, scheduler = trainUtils.build_optimizer_and_scheduler(args, model, t_total)
        # Train
        logger.info("***** Running training *****")
        logger.info("  Num Examples = {}".format(len(train_dataset)))
        logger.info("  Num Epochs = {}".format(args.train_epochs))
        logger.info("  Total training batch size = {}".format(args.train_batch_size))
        logger.info("  Total optimization steps = {}".format(t_total))
        global_step = 0
        model.zero_grad()
        # save_steps = t_total // args.train_epochs
        # eval_steps = save_steps
        eval_steps = log_loss_steps = 20 #每多少个step打印损失及进行验证
        # logger.info('Save model in {} steps; Eval model in {} steps'.format(save_steps, eval_steps))
        avg_loss = 0.
        train_loss_history = []
        dev_loss_history = []
        for epoch in range(args.train_epochs):
            for step, batch_data in enumerate(train_loader):
                model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)
                loss, logits = model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'], batch_data['labels'])

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # loss.backward(loss.clone().detach())
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % log_loss_steps == 0:
                    avg_loss /= log_loss_steps
                    logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                    train_loss_history.append(avg_loss)
                    avg_loss = 0.
                    if dev and global_step % eval_steps == 0:
                        model.eval()
                        batch_output_all = []
                        with torch.no_grad():
                            tot_dev_loss = 0.0
                            for eval_step, dev_batch_data in enumerate(dev_loader):
                                for key in dev_batch_data.keys():
                                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                                dev_loss, dev_logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'], dev_batch_data['token_type_ids'],dev_batch_data['labels'])

                                # tot_dev_loss += torch.sum(dev_loss).item()
                                tot_dev_loss += dev_loss.item()
                                batch_output = dev_logits.detach().cpu().numpy()
                                batch_output = np.argmax(batch_output, axis=2)
                                if len(batch_output_all) == 0:
                                    batch_output_all = batch_output
                                else:
                                    batch_output_all = np.append(batch_output_all,batch_output,axis=0)
                            total_count = [0 for _ in range(len(label2id))]
                            role_metric = np.zeros([len(id2label), 3])
                            for pred_label, tmp_callback in zip(batch_output_all, dev_callback_info):
                                text, gt_entities = tmp_callback
                                tmp_metric = np.zeros([len(id2label), 3])
                                pred_entities = decodeUtils.nor_bioes_decode(pred_label[1:1 + len(text)], text, idx2tag)
                                for idx, _type in enumerate(label_list):
                                    if _type not in pred_entities:
                                        pred_entities[_type] = []
                                    total_count[idx] += len(gt_entities[_type])
                                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type],pred_entities[_type])

                                role_metric += tmp_metric

                            dev_loss_history.append(tot_dev_loss / len(dev_loader))
                            mirco_metrics = np.sum(role_metric, axis=0)
                            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                            print('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0],mirco_metrics[1],mirco_metrics[2]))
                else:
                    avg_loss += loss.item()
                    # avg_loss += torch.sum(loss).item()

        trainUtils.save_model(args, model, model_name, global_step)
        draw.draw_loss(train_loss_history, dev_loss_history, 'c_loss')

    def test(self, model_path, data_path, args, idx2tag):

        dev_features,dev_callback_info = commonUtils.read_pkl(data_path, 'test')
        dev_dataset = bertNerNorDataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
        model.eval()

        pred_label = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'],dev_batch_data['labels'])
                batch_output = logits.detach().cpu().numpy()
                batch_output = np.argmax(batch_output, axis=2)
                if len(pred_label) == 0:
                    pred_label = batch_output
                else:
                    pred_label = np.append(pred_label, batch_output, axis=0)
            total_count = [0 for _ in range(len(id2label))]
            role_metric = np.zeros([len(id2label), 3])
            for pred, tmp_callback in zip(pred_label, dev_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(id2label), 3])
                pred_entities = decodeUtils.nor_bioes_decode(pred[1:1 + len(text)], text, idx2tag)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric
            logger.info(metricsUtils.classification_report(role_metric, label_list, id2label, total_count))

    def predict(self, raw_text, model_path, args, idx2tag):
        self.model.eval()
        with torch.no_grad():
            model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
            tokenizer = BertTokenizer(
                os.path.join(args.bert_dir, 'vocab.txt'))
            tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
            encode_dict = tokenizer.encode_plus(text=tokens,
                                    max_length=args.max_seq_len,
                                    pad_to_max_length=True,
                                    is_pretokenized=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
            output = logits.detach().cpu().numpy()
            output = np.argmax(output, axis=2)

            pred_entities = decodeUtils.nor_bioes_decode(output[0][1:1 + len(tokens)], "".join(tokens), idx2tag)

            logger.info(pred_entities)


if __name__ == '__main__':
    dataset = 'c'
    args.train_epochs = 3
    args.train_batch_size = 32
    args.max_seq_len = 150
    if dataset == "c":
        args.data_dir = '../data/cner'
        data_path = os.path.join(args.data_dir, 'nor_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)
        # model = bertNerNor.BertNerNorModel(args.bert_dir, args.num_tags, args.dropout_prob)
        # bertForNer = BertForNerNor(model)
        # bertForNer.train(data_path, 'bert_nor_c', id2query, dev=True)

        model_path = '../checkpoints/bert_nor_c-360/model.pt'
        model = bertNerNor.BertNerNorModel(args.bert_dir, args.num_tags, 0.0)
        bertForNer = BertForNerNor(model)
        bertForNer.test(model_path, data_path, args, id2query)

        raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
        bertForNer.predict(raw_text, model_path, args, id2query)

    # if dataset == "jq":
    #     args.data_dir = '../data/jq_ner'
    #     data_path = os.path.join(args.data_dir, 'ori_data')
    #     ent2id_path = os.path.join(args.data_dir, 'ori_data')
    #     ent2id_dict = commonUtils.read_json(ent2id_path, 'ent2id')
    #     label2id = {}
    #     id2label = {}
    #     for k, v in ent2id_dict.items():
    #         label2id[k] = v
    #         id2label[v] = k
    #     logger.info(id2label)
    #     args.num_tags = 5
    #     model = bertNer.BertNerModel(args.bert_dir, args.num_tags, args.dropout_prob)
    #     bertForNer = BertForNer(model)
    #     bertForNer.train(data_path, 'bert_jq', id2label, dev=False)
    #
    #     # model_path = '../checkpoints/bert_jq-7000/model.pt'
    #     # model = bertNer.BertNerModel(args.bert_dir, args.num_tags, 0.0)
    #     # bertForNer = BertForNer(model)
    #     # bertForNer.test(model_path, data_path, args, id2label)
    #
    # if dataset == "lattice":
    #     args.data_dir = '../data/lattice_ner'
    #     data_path = os.path.join(args.data_dir, 'ori_data')
    #     ent2id_path = os.path.join(args.data_dir, 'ori_data')
    #     ent2id_dict = commonUtils.read_json(ent2id_path, 'ent2id')
    #     label2id = {}
    #     id2label = {}
    #     for k, v in ent2id_dict.items():
    #         label2id[k] = v
    #         id2label[v] = k
    #     logger.info(id2label)
    #     args.num_tags = 17
    #     model = bertNer.BertNerModel(args.bert_dir, args.num_tags, args.dropout_prob)
    #     bertForNer = BertForNer(model)
    #     bertForNer.train(data_path, 'bert_lattice', id2label, dev=False)
    #
    #     # model_path = '../checkpoints/bert_lattice-1080/model.pt'
    #     # model = bertNer.BertNerModel(args.bert_dir, args.num_tags, 0.0)
    #     # bertForNer = BertForNer(model)
    #     # bertForNer.test(model_path, data_path, args, id2label)
    #     # raw_text = "清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎急性发作。  国家医保目录（乙类）  口服。一次4～6粒，一日3次。分次用水送服。  吉林敖东洮南药业股份有限公司  对本品及其成分过敏者。  清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎 尚不明确。  每粒装0.3g  药理研究证明，土茯苓有抗炎、镇痛、利尿作用。槐花、白茅根有抗菌、利尿作用;益母草有利尿、抗菌、增强机体细胞免疫功能作用;藿香有抗炎、镇静、抗菌、抗病毒作用。 "
    #     # bertForNer.predict(raw_text, model_path, data_path, args, id2label)
    # if dataset == "tc":
    #     data_path = os.path.join(args.data_dir, 'ori_data')
    #     ent2id_path = os.path.join(args.data_dir, 'mid_data')
    #     ent2id_dict = commonUtils.read_json(ent2id_path, 'ori_ent2id')
    #     label2id = {}
    #     id2label = {}
    #     for k, v in ent2id_dict.items():
    #         label2id[k] = v
    #         id2label[v] = k
    #     # logger.info(id2label)
    #     # model = bertNer.BertNerModel(args.bert_dir, args.num_tags, args.dropout_prob)
    #     # bertForNer = BertForNer(model)
    #     # bertForNer.train(data_path, 'bert', id2label, dev=True)
    #
    #     model_path = '../checkpoints/bert-765/model.pt'
    #     model = bertNer.BertNerModel(args.bert_dir, args.num_tags, 0.0)
    #     bertForNer = BertForNer(model)
    #     # bertForNer.test(model_path, data_path, args, id2label)
    #     raw_text = "清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎急性发作。  国家医保目录（乙类）  口服。一次4～6粒，一日3次。分次用水送服。  吉林敖东洮南药业股份有限公司  对本品及其成分过敏者。  清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎 尚不明确。  每粒装0.3g  药理研究证明，土茯苓有抗炎、镇痛、利尿作用。槐花、白茅根有抗菌、利尿作用;益母草有利尿、抗菌、增强机体细胞免疫功能作用;藿香有抗炎、镇静、抗菌、抗病毒作用。 "
    #     bertForNer.predict(raw_text, model_path, data_path, args, id2label)