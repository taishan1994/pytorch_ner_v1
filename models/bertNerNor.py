import torch.nn as nn
from .bertBaseModel import BaseModel


class BertNerNorModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(BertNerNorModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob))
        #
        out_dims = mid_linear_dims

        # self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(out_dims, num_tags)
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()

        init_blocks = [self.mid_linear, self.classifier]
        # init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768]
        seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 128]
        # seq_out = self.dropout(seq_out)
        logits = self.classifier(seq_out)  # [24, 256, 53]
        if labels is None:
            return logits
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(logits.view(-1, logits.size()[2]), labels.view(-1))
        outputs = (loss, ) + (logits,)
        return outputs
