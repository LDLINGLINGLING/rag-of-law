import os
import torch
from torch import nn

from transformers import (
        get_linear_schedule_with_warmup,BertTokenizer,
        AdamW,
        AutoModelForSequenceClassification,#bert[b,l,h]->cls[b,h]*[h,num_tag]->[b,num_tag]
        AutoConfig
        )

pretrained_weights = "/ai/ld/remote/rag/code/text_classification/myfinetun-bert_large_chinese1_child"#r'D:\liudan\pretrain\bert-base-chinese'#建立模型
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)#分词模型
config = AutoConfig.from_pretrained(pretrained_weights)#加载配置文件
#单独指定config，在config中指定分类个数
nlp_classif = AutoModelForSequenceClassification.from_pretrained(pretrained_weights,
                                                           config=config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设备
nlp_classif.to(device)

def infer( model,tokenizer,device,string):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode_plus((string), truncation=True, max_length=480, padding='max_length',
                                              return_tensors='pt')  # 没有return_tensors会返回list！！！！

        outputs = model(ids["input_ids"].to(device),
                            attention_mask=ids["attention_mask"].to(device))

        logits = outputs[0]
        predic = torch.argmax(logits, axis=1).data.cpu().numpy()
    return predic
while True:
    query = input('请输入要预测的句子,quit结束:')
    if query == 'quit':
        break
    print(infer(nlp_classif,tokenizer,device,query))
