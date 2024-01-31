import os, json,random
from transformers import AutoTokenizer
import numpy as np
import math


tokenizer=AutoTokenizer.from_pretrained("/ai/ld/pretrain/bert-base-chinese/")
path = "/ai/ld/remote/rag/code/retrieval/fatiao.json"
with open(path, 'r') as f:
    data = json.load(f)

cls_count={}
dict = {} # KEY 分类 value count
c1_keys = []
c2_keys = []
my_dict = {} # parent class : sub class
for ls in data:
    c1_key = ls[0]
    c2_key = ls[1]
    combo_key = ls[0] + "-" + ls[1]
    if c1_key not in c1_keys:
        c1_keys.append(c1_key)
    if combo_key not in c2_keys:
        c2_keys.append(combo_key)

    if c1_key not in my_dict:
        my_dict[c1_key] = {}
    if c2_key not in my_dict[c1_key]:
        my_dict[c1_key][c2_key] = 0
print(my_dict)
# print(c1_keys)
# print(c2_keys)
with open(path,'r',encoding='utf-8') as f:
    fatiao_dict=json.load(f)
child_list=list({item[1] for item in fatiao_dict})
child_list.remove('附则')
child_list.remove('宪法修正案1988年')
child_dict={item:index for index,item in enumerate(child_list)}


with open('/ai/ld/remote/rag/code/text_classification/law_data/103_type_dict.json','w',encoding='utf-8') as f:
    json.dump(child_dict,f,ensure_ascii=False)

path = "/ai/ld/remote/ChatGLM-6B-main/ptuning/law/训练数据_带法律依据_92k.json"
with open(path, 'r') as f:
    data = json.load(f)
print(len(data))
exception_list = []
ex_list = []
new_list = []
multi_labels = []
except_child=[]
training_data = {}
def gen_ref_list(reference):
    ref_list = []
    for ref in reference:
        index = ref.find(": ")
        #print("index", index)
        rf = ref[index + 1:].lstrip().rstrip().replace("\n", "").replace("\"", "")
        if rf[-1] == ",":
            rf = rf[:-1]
        ref_list.append(rf)
    #print(ref_list)
    return ref_list

xiuzheng_list=[]

for obj in data:
    try:
        labels = set()#
        reference_list = obj["reference"]#获取
        ref_list = gen_ref_list(reference_list)
        question = obj["question"].replace("\n", "")
        # ref = reference_list[0].split(":")[0]
        # index = ref.find("-")
        # cls1, cls2 = ref[: index], ref[index + 1:]
        if len(reference_list) < 1:
            exception_list.append(obj)
            continue
        for ref in reference_list:
            law_name = ref.split(":")[0]
            if law_name in labels:
                continue
            index = law_name.find("-")
            cls1, cls2 = law_name[: index], law_name[index + 1:]
            if cls1 in my_dict and cls2 in my_dict[cls1]:
                if law_name not in labels:
                    # print(cls1, cls2)
                    labels.add(law_name)
                    my_dict[cls1][cls2] = my_dict[cls1][cls2] + 1
            else:
                if cls1 in my_dict:
                    labels.add(law_name)
                    my_dict[cls1][cls2] = 1
                else:
                    ex_list.append(obj)

        labels = list(labels)
        if len(labels) > 1:
            multi_labels.append(obj)
        else:
            index = labels[0].find("-")
            parent_cls, child_cls = labels[0][: index], labels[0][index + 1:]
            if child_cls in child_list:
                training_data[question] = {"parent_cls": parent_cls, "child_cls": child_cls, "pos": ref_list}
                if child_cls not in cls_count.keys():
                    cls_count[child_cls] = 1
                else:
                    cls_count[child_cls] = cls_count[child_cls] + 1
            elif '刑法修正' in child_cls:
                xiuzheng_list.append(ref_list)
                continue
                training_data[question] = {"parent_cls": parent_cls, "child_cls": '刑法', "pos": ref_list}
                if '刑法' not in cls_count.keys():
                    cls_count['刑法'] = 1
                else:
                    cls_count['刑法'] = cls_count['刑法'] + 1
            else:
                print(question,child_cls)
                except_child.append(child_cls)
    except Exception as e:
        print("error ", e, reference_list)
        exception_list.append(obj)
all_qustion=list(training_data.keys())

###分类数据量统计###
print('数据分布',cls_count)
cls_weight=[round(float(1/1+math.log(cls_count[child_cls]/33)),2) for child_cls in child_list if child_cls in cls_count.keys()]
print('权重',cls_weight)


###交集和并集###
print('真实出现，法条没有出现',set(cls_count.keys())-set(child_list))
print('法条出现，真实没有出现',set(child_list)-set(cls_count.keys()))


###长度统计###
lengths = [len(tokenizer(x)['input_ids']) for x in all_qustion]
percentiles = np.percentile(lengths, [10, 20, 30,40,50,60,70,80,90,95,98,99,100])
print('长度百分比',[10, 20, 30,40,50,60,70,80,90,95,98,99,99.5,99.9,100],percentiles)


random.shuffle(all_qustion)
tran_split=all_qustion[:int(len(all_qustion)*0.92)]
test_split=all_qustion[int(len(all_qustion)*0.92):]
with open('/ai/ld/remote/rag/code/text_classification/law_data/tran_103_type.txt','w',encoding='utf-8') as f:
    for question in tran_split:
        f.write(question+'****'+str(child_list.index(training_data[question]['child_cls']))+'\n')
with open('/ai/ld/remote/rag/code/text_classification/law_data/test_103_type.txt','w',encoding='utf-8') as f:
    for question in test_split:
        f.write(question+'****'+str(child_list.index(training_data[question]['child_cls']))+'\n')
child_dict={item:index for index,item in enumerate(child_list)}

with open('/ai/ld/remote/rag/code/text_classification/law_data/103_type_dict.json','w',encoding='utf-8') as f:
    json.dump(child_dict,f,ensure_ascii=False)

print(training_data["某企业的建设用地使用权即将到期，但该企业拟将该土地转让给其他企业。该企业需要根据哪些法规采取相应的措施？"])
# print(training_data)
print(len(multi_labels))
print(len(exception_list))
print(len(training_data))
cls1_list=set([dict['parent_cls'] for question,dict in training_data.items()])
cls2_list=set([dict['child_cls'] for question,dict in training_data.items()])
print(len(cls1_list),len(cls2_list))
# data_dist = "training_data.json"
# with open(data_dist, 'w') as fp:
#     json.dump( training_data, fp,ensure_ascii=False)
