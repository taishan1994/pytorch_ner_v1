import sys

sys.path.append('..')
import os
import logging
import numpy as np
import torch
from utils import commonUtils, trainUtils, metricsUtils, decodeUtils
from config import bertNerSpanConfig
from datasets import bertNerSpanDataset
from preprocess.bertNerSpanProcess import SpanBertFeature
from models import bertNerSpan
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

args = bertNerSpanConfig.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'bertForNerSpan.log'))


class BertForNerSpan:
    def __init__(self, model):
        self.model = model

    def train(self, data_path, model_name, idx2tag=None, dev=True):
        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = bertNerSpanDataset.NerSpanDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        if dev:
            dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
            dev_dataset = bertNerSpanDataset.NerSpanDataset(dev_features)
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
        # logger.info('Save model in {} steps; Eval model in {} steps'.format(save_steps, eval_steps))
        log_loss_steps = eval_steps = 20
        avg_loss = 0.
        for epoch in range(args.train_epochs):
            for step, batch_data in enumerate(train_loader):
                model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                start_logits, end_logits = model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'],batch_data['start_ids'], batch_data['end_ids'])
                loss = model.loss(batch_data['start_ids'], batch_data['end_ids'], start_logits, end_logits, batch_data['attention_masks'])
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                # logger.info('Step: %d / %d ---> cur loss: %.5f' % (global_step, t_total, loss.item()))
                if global_step % log_loss_steps == 0:
                    avg_loss /= log_loss_steps
                    logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                    avg_loss = 0.
                    if dev and global_step % eval_steps == 0:
                        model.eval()
                        s_logits, e_logits = None, None

                        for eval_step, dev_batch_data in enumerate(dev_loader):
                            for key in dev_batch_data.keys():
                                dev_batch_data[key] = dev_batch_data[key].to(self.device)
                            start_logits, end_logits = model(dev_batch_data['token_ids'],
                                                             dev_batch_data['attention_masks'],
                                                             dev_batch_data['token_type_ids'])
                            tmp_start_logits = start_logits.detach().cpu().numpy()
                            tmp_end_logits = end_logits.detach().cpu().numpy()
                            if s_logits is None:
                                s_logits = tmp_start_logits
                                e_logits = tmp_end_logits
                            else:
                                s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                                e_logits = np.append(e_logits, tmp_end_logits, axis=0)

                        role_metric = np.zeros([len(label2id), 3])
                        mirco_metrics = np.zeros(3)
                        for t_start_logits, t_end_logits, tmp_callback in zip(s_logits, e_logits, dev_callback_info):
                            text, gt_entities = tmp_callback
                            temp_start_logits = t_start_logits[1:1+len(t_start_logits)]
                            temp_end_logits = t_end_logits[1:1+len(t_end_logits)]
                            pred_entities = decodeUtils.span_decode(temp_start_logits, temp_end_logits, text, idx2tag)
                            # print("========================")
                            # print(pred_entities)
                            # print(gt_entities)
                            # print("========================")
                            for idx, _type in enumerate(label_list):
                                if _type not in pred_entities:
                                    pred_entities[_type] = []
                                role_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])
                        mirco_metrics = np.sum(role_metric, axis=0)
                        mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                        print('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1],mirco_metrics[2]))
                else:
                    avg_loss += loss.item()

                # if global_step % save_steps == 0:
        trainUtils.save_model(args, model, model_name, global_step)

    def test(self, model_path, data_path, args, idx2tag):
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'test')
        dev_dataset = bertNerSpanDataset.NerSpanDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
        with torch.no_grad():
            s_logits, e_logits = None, None
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                start_logits, end_logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],
                                                 dev_batch_data['token_type_ids'])
                tmp_start_logits = start_logits.detach().cpu().numpy()
                tmp_end_logits = end_logits.detach().cpu().numpy()
                if s_logits is None:
                    s_logits = tmp_start_logits
                    e_logits = tmp_end_logits
                else:
                    s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                    e_logits = np.append(e_logits, tmp_end_logits, axis=0)

            role_metric = np.zeros([len(label2id), 3])
            total_count = [0 for _ in range(len(id2label))]
            for t_start_logits, t_end_logits, tmp_callback in zip(s_logits, e_logits, dev_callback_info):
                text, gt_entities = tmp_callback
                temp_start_logits = t_start_logits[1:1 + len(text)]
                temp_end_logits = t_end_logits[1:1+ len(text)]
                pred_entities = decodeUtils.span_decode(temp_start_logits, temp_end_logits, text, idx2tag)
                # print(pred_entities)
                # print(text)
                # print("==================")
                # print(gt_entities)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    role_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])
            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            logger.info('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1],mirco_metrics[2]))
            logger.info(metricsUtils.classification_report(role_metric, label_list, id2label, total_count))


    def predict(self, raw_text, model_path, args, idx2tag):
        self.model.eval()
        with torch.no_grad():
            model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
            model.to(device)
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
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(device)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(device)
            start_logits, end_logits = model(token_ids, attention_masks,token_type_ids)
            tmp_start_logits = start_logits.detach().cpu().numpy()
            tmp_end_logits = end_logits.detach().cpu().numpy()
            for t_start_logits, t_end_logits in zip(tmp_start_logits,tmp_end_logits):
                temp_start_logits = t_start_logits[1:1 + len(raw_text)]
                temp_end_logits = t_end_logits[1:1 + len(raw_text)]
                pred_entities = decodeUtils.span_decode(temp_start_logits, temp_end_logits, raw_text, idx2tag)
                logger.info(pred_entities)


if __name__ == '__main__':
    dataset = 'c'
    args.train_epochs = 3
    args.train_batch_size = 32
    args.max_seq_len = 150

    if dataset == "c":
        args.data_dir = '../data/cner/'
        data_path = os.path.join(args.data_dir, 'span_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'span_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        id2label = {}
        label2id = {}
        for k,v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        id2query = {}
        query2id = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(id2query) + 1
        logger.info(args)
        # print(id2query)
        # model = bertNerSpan.BertNerSpanModel(args.bert_dir, args.num_tags, args.dropout_prob)
        # bertForNerSpan = BertForNerSpan(model)
        # bertForNerSpan.train(data_path, 'bertSpan', id2query, dev=True)

        model_path = '../checkpoints/bertSpan-360/model.pt'
        model = bertNerSpan.BertNerSpanModel(args.bert_dir, args.num_tags, 0.0)
        bertForNerMrc = BertForNerSpan(model)
        bertForNerMrc.test(model_path, data_path, args, id2query)
        raw_text = "顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。"
        bertForNerMrc.predict(raw_text, model_path, args, id2query)

    if dataset == 'tc':
        data_path = os.path.join(args.data_dir, 'span_data')
        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(ent2id_path, 'span_ent2id')
        label2id = {}
        id2label = {}
        for k, v in ent2id_dict.items():
            label2id[k] = v
            id2label[v] = k
        logger.info(id2label)
        model = bertNerSpan.BertNerSpanModel(args.bert_dir, args.num_tags, args.dropout_prob)
        bertForNerSpan = BertForNerSpan(model)
        bertForNerSpan.train(data_path, 'bertSpan', id2label, dev=True)

        # model_path = '../checkpoints/bertMrc-2240/model.pt'
        # model = bertNerMrc.BertNerMrcModel(args.bert_dir, 0.0)
        # bertForNerMrc = BertForNerMrc(model)
        # bertForNerMrc.test(model_path, data_path, args, id2label)
        # raw_text = "清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎急性发作。  国家医保目录（乙类）  口服。一次4～6粒，一日3次。分次用水送服。  吉林敖东洮南药业股份有限公司  对本品及其成分过敏者。  清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎 尚不明确。  每粒装0.3g  药理研究证明，土茯苓有抗炎、镇痛、利尿作用。槐花、白茅根有抗菌、利尿作用;益母草有利尿、抗菌、增强机体细胞免疫功能作用;藿香有抗炎、镇静、抗菌、抗病毒作用。 "
        # bertForNer.predict(raw_text, model_path, data_path, args, id2label)
