from transformers import BertTokenizer
bert_dir = '/content/drive/MyDrive/simpleNLP/model_hub/bert-base-case/bert-base-cased-vocab.txt'
tokenizer = BertTokenizer(bert_dir)
def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens
text = '我\t爱 北京。\n我爱中国哦！'
tmp1 = tokenizer.tokenize(text)
print('原始的tokenizer的输出：', tmp1)
print('长度：',len(tmp1))
label = 'O O O O B-LOC E-LOC O O O O B-LOC E-LOC O O'
print('标签长度：', len(label.split(" ")))
tmp2 = fine_grade_tokenize(text, tokenizer)
print('处理之后的输入：',tmp2)
print("长度：",len(tmp2))
"""
原始的tokenizer的输出： ['[UNK]', '[UNK]', '北', '京', '。', '[UNK]', '[UNK]', '中', '国', '[UNK]', '！']
长度： 11
标签长度： 14
处理之后的输入： ['我', '[BLANK]', '爱', '[BLANK]', '北', '京', '。', '[BLANK]', '我', '爱', '中', '国', '哦', '！']
长度： 14
"""

from transformers import BertTokenizer
import os

tokens = ['我','爱','北','京','天','安','门']

tokenizer = BertTokenizer(os.path.join('/content/drive/MyDrive/simpleNLP/model_hub/bert-base-case','vocab.txt'))
encode_dict = tokenizer.encode_plus(text=tokens,
                  max_length=256,
                  pad_to_max_length=True,
                  is_pretokenized=True,
                  return_token_type_ids=True,
                  return_attention_mask=True)
# 我们要手动加上两个标志位
tokens = ['[CLS]'] + tokens + ['[SEP]']
print(' '.join(tokens))
print(encode_dict['input_ids'])
# [CLS] 我 爱 北 京 天 安 门 [SEP]
# [101, 100, 100, 993, 984, 1010, 1016, 100, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


from transformers import BertTokenizer
import os

tokens = ['我','爱','北','京','天','安','门']

tokenizer = BertTokenizer(os.path.join('/content/drive/MyDrive/simpleNLP/model_hub/bert-base-case','vocab.txt'))
tokens_a = '我 爱 北 京 天 安 门'.split(' ')
tokens_b = '我 爱 打 英 雄 联 盟 啊 啊'.split(' ')

encode_dict = tokenizer.encode_plus(text=tokens_a,
                  text_pair=tokens_b,
                  max_length=20,
                  pad_to_max_length=True,
                  truncation_strategy='only_second',
                  is_pretokenized=True,
                  return_token_type_ids=True,
                  return_attention_mask=True)
tokens = " ".join(['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'])
token_ids = encode_dict['input_ids']
attention_masks = encode_dict['attention_mask']
token_type_ids = encode_dict['token_type_ids']

print(tokens)
print(token_ids)
print(attention_masks)
print(token_type_ids)
# Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
# [CLS] 我 爱 北 京 天 安 门 [SEP] 我 爱 打 英 雄 联 盟 啊 啊 [SEP]
# [101, 100, 100, 993, 984, 1010, 1016, 100, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100, 102, 0]
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# /usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py:2079: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
#   FutureWarning,