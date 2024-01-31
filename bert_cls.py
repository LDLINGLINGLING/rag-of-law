# -*- coding: utf-8 -*-

import os
import torch
from torch import nn

from transformers import (
        get_linear_schedule_with_warmup,BertTokenizer,
        AdamW,
        AutoModelForSequenceClassification,#bert[b,l,h]->cls[b,h]*[h,num_tag]->[b,num_tag]
        AutoConfig
        )

from torch.utils.data import DataLoader,dataset
import time
import numpy as np
from sklearn import metrics
from datetime import timedelta
import os
import re
import tqdm
import random
class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
#path = \THUCNews'
# datanames = os.listdir(path)
# list = []
# data_total=[]
# for num,i in enumerate(datanames):
#     for j in tqdm.tqdm(os.listdir(path+'\\'+i)):
#         content=open(path+'\\'+i+'\\'+j,encoding='UTF-8').readlines()[0]
#         content=re.sub(r'\n','',content)
#         content=re.sub(r'\s+','',content)
#         content=content+'\t'+str(num)
#         data_total.append(content)
#
# random.shuffle(data_total)
# with open(r'./data1/数据_train.txt','w',encoding='UTF-8')as f:
#     total_lines=len(data_total)
#     for n,line in enumerate(data_total):
#         if n /total_lines<0.9:
#             f.write(line+'\n')
# data_total.reverse()
# with open(r'./data1/数据_val.txt','w',encoding='UTF-8')as f:
#     total_lines=len(data_total)
#     for n,line in enumerate(data_total):
#         if n /total_lines<0.1:
#             f.write(line+'\n')
data_dir='/ai/ld/remote/rag/code/text_classification/law_data'

def read_file(path):
    with open(path, 'r', encoding="UTF-8") as file:
        docus = file.readlines()
        newDocus = []
        for data in docus:
            newDocus.append(data)
    return newDocus


#建立数据集 
class Label_Dataset(dataset.Dataset):
    def __init__(self,data):#data为列表型
        self.data = data#
    def __len__(self):#返回数据长度
        return len(self.data)
    def __getitem__(self,ind):
        onetext = self.data[ind]
        content, label = onetext.split('****')#content 属于列表
        try:
            label = torch.LongTensor([int(label)])#label属于tensor
        except:
            print(content)
        return content,label

trainContent = read_file(os.path.join(data_dir, "train_law_child.txt"))
testContent = read_file(os.path.join(data_dir, "test_law_child.txt"))

traindataset =Label_Dataset( trainContent )#实例化一个数据-标签类
testdataset =Label_Dataset( testContent )#实例化一个数据-标签类
batch_size = 128
testdataloder = DataLoader(testdataset, batch_size=batch_size, shuffle = False)#测试批量1

traindataloder = DataLoader(traindataset, batch_size=batch_size, shuffle = False)

# class_list = [x.strip() for x in open(
#         os.path.join(data_dir, "class.txt")).readlines()]#分类情况
# class_list=datanames

pretrained_weights = "/ai/ld/pretrain/bert-base-uncased/"#r'D:\liudan\pretrain\bert-base-chinese'#建立模型
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)#分词模型
config = AutoConfig.from_pretrained(pretrained_weights,num_labels= 115)#加载配置文件
#单独指定config，在config中指定分类个数
nlp_classif = AutoModelForSequenceClassification.from_pretrained(pretrained_weights,
                                                           config=config)#加载模型
print(nlp_classif)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")#设备
nlp_classif = nlp_classif.to(device)#将模型加载到设备

time_start = time.time() #开始时间

epochs = 50
gradient_accumulation_steps = 1
max_grad_norm =0.1  #梯度剪辑的阀值

require_improvement = 1500                 # 若超过1000batch效果还没提升，则提前结束训练
savedir = '/ai/ld/remote/rag/code/text_classification/myfinetun-bert_large_chinese1_child/'#保存地址
os.makedirs(savedir, exist_ok=True)
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train( model, traindataloder, testdataloder):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())#所有参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0},#不存在于列表中的参数进行正则化
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.05}]#不存在列表中的进行正则化

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=100, num_training_steps=len(traindataloder) * epochs)#在warmup期间学习率上升，后面则是训练总步数


    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    
    for epoch in range(epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs))
        for i, (sku_name, labels) in enumerate(tqdm.tqdm(traindataloder)):
            model.train()
            
            ids = tokenizer.batch_encode_plus( sku_name,#进行编码
               truncation=True,max_length=180,padding='max_length',return_tensors='pt')#没有return_tensors会返回list！！！！
            #[b,l]
            labels = labels.squeeze().to(device)
            outputs = model(ids["input_ids"].to(device), labels=labels,
                            attention_mask =ids["attention_mask"].to(device))#
            
            loss, logits = outputs[:2]#[b,num_tag]
            #以下四行行为加的
            #weight = torch.from_numpy(np.array([3.8, 3.74, 1.44, 2.0, 4.38, 4.35, 4.37, 3.53, 4.24, 3.55, 3.8, 3.86, 3.7, 3.41, 3.48, 4.05, 3.2, 5.29, 2.48, 5.5, 3.01, 5.65, 4.09, 3.33, 4.26, 2.98, 4.99, 3.85, 4.18, 4.98, 3.79, 3.62, 3.76, 4.92, 3.3, 3.96, 3.6, 4.43, 4.84, 5.21, 2.95, 5.01, 3.0, 3.68, 5.72, 4.43, 3.68, 4.39, 4.22, 3.96, 3.85, 2.23, 2.74, 3.16, 4.67, 4.24, 2.87, 4.36, 4.46, 3.95, 3.46, 3.48, 4.92, 4.14, 3.69, 4.51, 3.73, 3.5, 4.14, 5.13, 4.05, 4.84, 3.21, 3.4, 3.86, 3.64, 4.42, 4.29, 5.2, 5.24, 6.35, 3.58, 3.15, 1.33, 3.46, 4.12, 4.71, 4.05, 2.94, 3.61, 1.0, 4.23, 4.36, 3.98, 3.12, 1.93, 4.27, 4.03, 3.37, 4.29, 5.42])).float().to(device)
            loss_fct = nn.CrossEntropyLoss()
            #loss_fct =FocalLoss()
            loss = loss_fct(logits, labels)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps#显存小的时候可以梯度累加
            
            loss.backward()#反向传播

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)#梯度裁剪

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if total_batch % 100 == 99:
                # 每多少轮输出在训练集和验证集上的效果
                truelabel = labels.data.cpu()
                predic = torch.argmax(logits,axis=1).data.cpu()#预测值
#                predic = torch.max(outputs.data1, 1)[1].cpu()
                train_acc = metrics.accuracy_score(truelabel, predic)#训练正确率
                dev_acc, dev_loss = evaluate( model, testdataloder)#验证集正确率
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    model.save_pretrained(savedir) #如果loss下降则保存
                    
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def evaluate(model, testdataloder):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for j,(sku_name, labels) in enumerate(tqdm.tqdm(testdataloder,position=0)):
            ids = tokenizer.batch_encode_plus( sku_name,
                truncation=True,max_length=436,padding='max_length',return_tensors='pt')#没有return_tensors会返回list！！！！
               
            labels = labels.squeeze().to(device) 
            outputs = model(ids["input_ids"].to(device), labels=labels, 
                                   attention_mask =ids["attention_mask"].to(device) )
            
            loss, logits = outputs[:2]
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.argmax(logits,axis=1).data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    print('acc:',acc)
    return acc, loss_total / len(testdataloder)


def infer( string,model_path=r'./myfinetun-bert_chinese',num_labels=3):
    tokenizer = BertTokenizer.from_pretrained(model_path)  # 分词模型
    config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)  #
    # 单独指定config，在config中指定分类个数
    model = AutoModelForSequenceClassification.from_pretrained(model_path,config=config)
    model.to(device)
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode_plus((string), truncation=True, max_length=436, padding='max_length',
                                              return_tensors='pt')  # 没有return_tensors会返回list！！！！

        outputs = model(ids["input_ids"].to(device),
                            attention_mask=ids["attention_mask"].to(device))

        logits = outputs[0]
        predic = torch.argmax(logits, axis=1).data.cpu().numpy()
    return predic
train(nlp_classif, traindataloder, testdataloder)
# evaluate(nlp_classif,testdataloder)
# import pandas as pd
# df = pd.read_excel(r'E:\科大工作\知识图谱\基于BERT模型得自然语言处理实战\定时任务\2023-01-03\微信公众号1_2023-01-03_.xlsx')
# data=df['title'].values
#
# for i in data:
#     print(infer(i),i)
