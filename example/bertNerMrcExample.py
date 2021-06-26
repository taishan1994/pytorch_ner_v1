import sys

sys.path.append('..')
import os
import logging
import numpy as np
import torch
from utils import commonUtils, trainUtils, metricsUtils, decodeUtils
from config import bertNerMrcConfig
from datasets import bertNerMrcDataset
from preprocess.bertNerMrcProcess import MRCBertFeature
from models import bertNerMrc
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

args = bertNerMrcConfig.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'bertForNerMrc.log'))



class BertForNerMrc:
    def __init__(self, model):
        self.model = model

    def train(self, data_path, model_name, args, dev=True):
        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = bertNerMrcDataset.NerMrcDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        if dev:
            dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
            dev_dataset = bertNerMrcDataset.NerMrcDataset(dev_features)
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
        log_loss_steps = 20
        eval_steps = 60
        avg_loss = 0.
        for epoch in range(args.train_epochs):
            for step, batch_data in enumerate(train_loader):
                model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                start_logits, end_logits = model(batch_data['token_ids'], batch_data['attention_masks'],batch_data['token_type_ids'],batch_data['start_ids'], batch_data['end_ids'])
                loss = model.loss(batch_data['start_ids'], batch_data['end_ids'], start_logits, end_logits,batch_data['token_type_ids'])
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # loss.backward(loss.clone().detach())
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
                    # and global_step % 60 == 0
                    if dev and global_step % eval_steps == 0:
                        model.eval()
                        s_logits, e_logits = None, None

                        with torch.no_grad():
                            for eval_step, dev_batch_data in enumerate(dev_loader):
                                for key in dev_batch_data.keys():
                                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                                start_logits, end_logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'])
                                tmp_start_logits = start_logits.detach().cpu().numpy()
                                tmp_end_logits = end_logits.detach().cpu().numpy()
                                if s_logits is None:
                                    s_logits = tmp_start_logits
                                    e_logits = tmp_end_logits
                                else:
                                    s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                                    e_logits = np.append(e_logits, tmp_end_logits, axis=0)

                            role_metric = np.zeros([len(label2id), 3])

                            for t_start_logits, t_end_logits, tmp_callback in zip(s_logits, e_logits, dev_callback_info):
                                text, text_offset, ent_type, gt_entities = tmp_callback
                                gt_entities = [(text[entity[1]:entity[2]+1], entity[1]) for entity in gt_entities]
                                # print(gt_entities)
                                temp_start_logits = t_start_logits[text_offset:text_offset + len(text)]
                                temp_end_logits = t_end_logits[text_offset:text_offset + len(text)]

                                pred_entities = decodeUtils.mrc_decode(temp_start_logits, temp_end_logits, text)
                                # if len(pred_entities) != 0:
                                #     print(ent_type, pred_entities)
                                role_metric[label2id[ent_type]] += metricsUtils.calculate_metric(gt_entities, pred_entities)

                            mirco_metrics = np.sum(role_metric, axis=0)
                            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                            print('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))

                else:
                    # avg_loss += torch.sum(loss).item()
                    avg_loss = loss.item()

        trainUtils.save_model(args, model, model_name, global_step)

    def test(self, model_path, data_path, args):
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'test')
        dev_dataset = bertNerMrcDataset.NerMrcDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
        s_logits, e_logits = None, None
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                start_logits, end_logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'],dev_batch_data['token_type_ids'])
                tmp_start_logits = start_logits.detach().cpu().numpy()
                tmp_end_logits = end_logits.detach().cpu().numpy()
                if s_logits is None:
                    s_logits = tmp_start_logits
                    e_logits = tmp_end_logits
                else:
                    s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                    e_logits = np.append(e_logits, tmp_end_logits, axis=0)

            role_metric = np.zeros([len(label2id), 3])
            total_count = [0 for _ in range(len(label2id))]
            for t_start_logits, t_end_logits, tmp_callback in zip(s_logits, e_logits, dev_callback_info):
                text, text_offset, ent_type, gt_entities = tmp_callback
                gt_entities = [(text[entity[1]:entity[2] + 1], entity[1]) for entity in gt_entities]
                temp_start_logits = t_start_logits[text_offset:text_offset + len(text)]
                temp_end_logits = t_end_logits[text_offset:text_offset + len(text)]
                pred_entities = decodeUtils.mrc_decode(temp_start_logits, temp_end_logits, text)
                total_count[label2id[ent_type]] += len(gt_entities)
                role_metric[label2id[ent_type]] += metricsUtils.calculate_metric(gt_entities, pred_entities)
            logger.info(metricsUtils.classification_report(role_metric,label_list,id2label, total_count))

    def predict(self, raw_text, model_path, args, query2label):
        self.model.eval()
        with torch.no_grad():
            model, device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids, model_path)
            model.to(device)
            tokenizer = BertTokenizer(
                os.path.join(args.bert_dir, 'vocab.txt'))
            tokens_b = commonUtils.fine_grade_tokenize(raw_text, tokenizer)

            for text_a, label in query2label.items():
                tokens_a = commonUtils.fine_grade_tokenize(text_a, tokenizer)
                encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                    text_pair=tokens_b,
                                                    max_length=args.max_seq_len,
                                                    pad_to_max_length=True,
                                                    truncation_strategy='only_second',
                                                    # is_pretokenized=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True)

                token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(device)
                attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(device)
                token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(device)

                start_logits, end_logits = model(token_ids,
                                                 attention_masks,
                                                 token_type_ids)

                tmp_start_logits = start_logits.detach().cpu().numpy()
                tmp_end_logits = end_logits.detach().cpu().numpy()
                text_offset = len(tokens_a) + 2
                text = tokens_b
                logger.info(label)
                for t_start_logits, t_end_logits in zip(tmp_start_logits, tmp_end_logits):
                    temp_start_logits = t_start_logits[text_offset:text_offset + len(text)]
                    temp_end_logits = t_end_logits[text_offset:text_offset + len(text)]
                    pred_entities = decodeUtils.mrc_decode(temp_start_logits, temp_end_logits, "".join(tokens_b))
                    logger.info(pred_entities)



if __name__ == '__main__':
    dataset = "cner"

    if dataset == 'cner':
        args.train_epochs = 3
        args.train_batch_size = 32
        args.max_seq_len = 150
        args.data_dir = '../data/cner'
        data_path = os.path.join(args.data_dir, 'mrc_data')
        other_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(other_path, 'mrc_ent2id')
        label2query = {}
        query2label = {}
        label_list = commonUtils.read_json(other_path, 'labels')
        id2label = {}
        label2id = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        # 这里的label是问题
        for k, v in ent2id_dict.items():
            label2query[k] = v
            query2label[v] = k
        logger.info(query2label)
        # model = bertNerMrc.BertNerMrcModel(args.bert_dir, args.dropout_prob)
        # bertForNerMrc = BertForNerMrc(model)
        # bertForNerMrc.train(data_path, 'bertMrc', args, dev=True)

        model_path = '../checkpoints/bertMrc-2868/model.pt'
        model = bertNerMrc.BertNerMrcModel(args.bert_dir, 0.0)
        bertForNerMrc = BertForNerMrc(model)
        bertForNerMrc.test(model_path, data_path, args)
        raw_text = "1954年10月出生，大专学历，中共党员，高级经济师，汉商集团董事长、党委副书记。"
        bertForNerMrc.predict(raw_text, model_path, args, query2label)

    if dataset == 'tcner':

        data_path = os.path.join(args.data_dir, 'mrc_data')
        ent2id_path = os.path.join(args.data_dir, 'mid_data')
        ent2id_dict = commonUtils.read_json(ent2id_path, 'mrc_ent2id')
        label2query = {}
        query2label = {}
        # 这里的label是问题
        for k, v in ent2id_dict.items():
            label2query[k] = v
            query2label[v] = k
        logger.info(query2label)
        model = bertNerMrc.BertNerMrcModel(args.bert_dir, args.dropout_prob)
        bertForNerMrc = BertForNerMrc(model)
        bertForNerMrc.train(data_path, 'bertMrc', id2label, dev=True)

        # model_path = '../checkpoints/bertMrc-22400/model.pt'
        # model = bertNerMrc.BertNerMrcModel(args.bert_dir, 0.0)
        # bertForNerMrc = BertForNerMrc(model)
        # bertForNerMrc.test(model_path, data_path, args, query2label)
        # raw_text = "清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎急性发作。  国家医保目录（乙类）  口服。一次4～6粒，一日3次。分次用水送服。  吉林敖东洮南药业股份有限公司  对本品及其成分过敏者。  清热利尿，益肾化浊。用于热淋涩痛，急性肾炎水肿，慢性肾炎 尚不明确。  每粒装0.3g  药理研究证明，土茯苓有抗炎、镇痛、利尿作用。槐花、白茅根有抗菌、利尿作用;益母草有利尿、抗菌、增强机体细胞免疫功能作用;藿香有抗炎、镇静、抗菌、抗病毒作用。 "
        # bertForNerMrc.predict(raw_text, model_path, data_path, args, query2label)
