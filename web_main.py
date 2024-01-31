import gradio as gr
import copy
import re
import os
import torch
from torch import nn
import json
from transformers import (
        get_linear_schedule_with_warmup,BertTokenizer,
        AdamW,
        AutoModelForSequenceClassification,#bert[b,l,h]->cls[b,h]*[h,num_tag]->[b,num_tag]
        AutoConfig
        )
import faiss                   
import pickle
import argparse
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer,util
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig


prompt_retrieval="""{retrival_text}\n
请你作为一个专业的资深律师，严格根据以上法律条文，不要生成法律条文中没有提到的回复，可能存在一些不需要使用的法律条文，可以忽略。对以下问题{question}给出严谨的答案："""

prompt_noretrieval='请你作为一个专业的资深律师，对以下问题{question}给出严谨客观的答案：'
def load_LLM(LLM_path):
    tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(LLM_path, trust_remote_code=True)
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_path,
        device_map='auto',
        trust_remote_code=True,
        #use_safetensors=True,
        # bf16=True
    ).eval()
    return tokenizer,model


with open('/ai/ld/remote/rag/code/text_classification/law_data/old_dict.json','r',encoding='utf-8') as f:
    old_dict=json.load(f)
def load_cls_model(pretrained_weights):
    pretrained_weights = "/ai/ld/remote/rag/code/text_classification/myfinetun-bert_large_chinese1_child"#r'D:\liudan\pretrain\bert-base-chinese'#建立模型
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)#分词模型
    config = AutoConfig.from_pretrained(pretrained_weights)#加载配置文件
    #单独指定config，在config中指定分类个数
    nlp_classif = AutoModelForSequenceClassification.from_pretrained(pretrained_weights,
                                                            config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设备
    nlp_classif.to(device)
    return nlp_classif,tokenizer,device

def cls_infer( model,tokenizer,device,string):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode_plus((string), truncation=True, max_length=480, padding='max_length',
                                              return_tensors='pt')  # 没有return_tensors会返回list！！！！

        outputs = model(ids["input_ids"].to(device),
                            attention_mask=ids["attention_mask"].to(device))

        logits = outputs[0]
        predic = torch.argmax(logits, axis=1).data.cpu().numpy()
    
    return predic

def load_retrieval_model(pretrained_weights):
    emb_model =  SentenceTransformer(pretrained_weights)
    return emb_model

def retriver(emb_model,query,emb_path,list_path):
    law_embeds = pickle.load(open(emb_path, 'rb'))#读取向量
    q_emb = emb_model.encode([query])
    law_list=pickle.load(open(list_path, 'rb'))
    similarities = util.cos_sim(q_emb, law_embeds)

    cosine_similarities = np.dot(law_embeds, q_emb.T) / (np.linalg.norm(law_embeds, axis=1) * np.linalg.norm(q_emb))
    cosine_similarities=cosine_similarities[:,0]
    average_similarity=np.percentile(cosine_similarities, 80)
    binary_similarities = (cosine_similarities > average_similarity).astype(int).tolist()
    retriber_start2end=find_valid_sequences(binary_similarities,zero_threshold=5,near_zero_threshold=1,min_length=4)
    retriver_list=[]
    retriver_string=''
    for i in retriber_start2end:
        retriver_list.append(law_list[i[0]:i[1]+1])
        retriver_string+='\n'.join(law_list[i[0]:i[1]+1])+'\n\n'
    # index = faiss.IndexFlatIP(law_embeds.shape[-1])#建立一个索引库，其中向量维度为 law_embeds.shape[-1]
    # index.add(law_embeds)
    # D, I = index.search(q_emb,top_k)#找出最相似的top-k个向量的乘积和索引
    return retriver_list,retriver_string

def find_valid_sequences(A,zero_threshold=5,near_zero_threshold=1,min_length=3):
    results = []
    start = None
    zero_count = 0
    in_sequence = False

    for i, num in enumerate(A):
        if num == 1:
            if not in_sequence:
                in_sequence = True
                start = i
                zero_count = 0
            elif i == len(A) - 1 and in_sequence:
                # End of list, close the sequence
                if i - start >= 1:
                    results.append((start, i))
        elif num == 0 and in_sequence:
            zero_count += 1
            if zero_count > zero_threshold or (i < len(A) - 1 and A[i + 1] == 0):
                # End the current sequence if more than two zeros or next is also zero
                if i - start - zero_count >= 1 and i-1-start>=min_length-1:
                    results.append((start, i - 1))
                in_sequence = False
                zero_count = 0
    return results

def gradio_response(message, history): 
    global history_copy
    if message=='clear':
        history=[]
    #message是原始问题
    if history!=[]:
        history=copy.deepcopy(history_copy)
    print(history)

    try:
        query_type=str(cls_infer(cls_model,cls_tokenizer,device,message)[0])
        if query_type  in old_dict.keys():
            emb_path='/ai/ld/remote/rag/code/retrieval/fatiao_embeding/{}.pkl'.format(old_dict[query_type])
            list_path='/ai/ld/remote/rag/code/retrieval/fatiao_embeding/{}_list.pkl'.format(old_dict[query_type])
            retriver_list,retriver_string=retriver(emb_model,message,emb_path,list_path)
            print('引用法条:',retriver_string,'\n\n')
            response,history=llm.chat(tokenizer,prompt_retrieval.format(retrival_text=retriver_string,question=message),history=[])
            history_copy=copy.deepcopy(history)
            print(response)
            return '引用《{}》法条如下：\n\n'.format(old_dict[query_type])+retriver_string+'\n\n'+'Final Answer:\n'+response
        else:
            print('未找到相关法条，以下回复准确性可能降低！')
            response,history=llm.chat(tokenizer,prompt_noretrieval.format(question=message),history=[])
            history_copy=copy.deepcopy(history)
            print(response)
            return '没有该类型的知识库，回复准确率可能降低！'+'\n\n'+'Final Answer:\n'+response
    except Exception as e:
        response=e
        return response
    
    


if __name__=='__main__':
    cls_model,cls_tokenizer,device=load_cls_model(pretrained_weights = "/ai/ld/remote/rag/code/text_classification/myfinetun-bert_large_chinese1_child")
    emb_model=load_retrieval_model('/ai/ld/remote/rag/code/sentence-transformer/output/bi-encoder/stsb_augsbert_BM25_-ai-ld-remote-rag-code-sentence-transformer-output-bi-encoder-stsb_augsbert_BM25_-ai-ld-pretrain-bge-base-zh--2024-01-22_12-57-19-2024-01-23_16-51-19')
    tokenizer,llm=load_LLM('/ai/ld/pretrain/Qwen-14B-Chat/')
    demo = gr.ChatInterface(gradio_response,chatbot=gr.Chatbot(label="Chagent:\nchain of agent",height=500))
    demo.launch(share=True)