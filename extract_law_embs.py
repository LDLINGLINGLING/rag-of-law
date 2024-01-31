from sentence_transformers import SentenceTransformer
from text2vec import Word2Vec
import json
import numpy as np
import pickle
from tqdm import tqdm
import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

BS = 2048
def compute_emb(law_data, model):
    global BS#定义batchsize，即每次计算多少个语句变成向量
    all_embs = None#最终的embedings
    iteration_nums = len(law_data) // BS + 1#总共计算多少轮
    # Embed a list of sentences
    for i in tqdm(range(iteration_nums)):
        batch_law_data = law_data[i*BS:i*BS+BS]#这一批的样本
        sentences = []
        for j in range(len(batch_law_data)):
            sentences.append(batch_law_data[j][-1])#将本句的法条原文拿出来
        sentence_embeddings = model.encode(sentences)#[32,512]计算向量
        if all_embs is not None:
            all_embs = np.concatenate([all_embs, sentence_embeddings], axis=0)#计算向量concat起来
        else:
            all_embs = sentence_embeddings
    all_embs = torch.nn.functional.normalize(torch.tensor(all_embs)).numpy()#归一化，可以理解为除以向量的模
    return all_embs


if __name__ == "__main__":
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    # device = torch.device("cpu")
    t2v_model = SentenceTransformer("/ai/ld/remote/rag/code/sentence-transformer/output/bi-encoder/stsb_augsbert_BM25_-ai-ld-pretrain-bge-base-zh--2024-01-22_12-57-19")#这里是使用使用词嵌入模型
    law_data = json.load(open("/ai/ld/remote/rag/code/retrieval/fatiao.json",encoding='utf-8'))#读取法条数据
    law_child_type=set([typ[1].strip() for typ in law_data])
    law_dict={}
    for i in law_data:
        if i[1].strip() in law_child_type:
            if i[1].strip() in law_dict:
                law_dict[i[1].strip()].append(i[2])
            else:
                law_dict[i[1].strip()] = [i[2]]

    for child_cls in law_dict.keys():
        all_embs = t2v_model.encode(law_dict[child_cls])#将法条文本变成向量
        pickle.dump(all_embs, open("/ai/ld/remote/rag/code/retrieval/fatiao_embeding/{child_cls}.pkl".format(child_cls=child_cls), 'wb'))#将向量写入pkl文件
        pickle.dump(law_dict[child_cls],open("/ai/ld/remote/rag/code/retrieval/fatiao_embeding/{child_cls}_list.pkl".format(child_cls=child_cls), 'wb'))