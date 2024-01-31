import os, json

path = "fatiao.json"
with open(path, 'r') as f:
    data = json.load(f)

dict = {}  # KEY 分类 value count
c1_keys = []
c2_keys = []
my_dict = {}  # parent class : sub class
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

path = "/Users/junxiyin/Desktop/law/训练数据_带法律依据_92k.json"
with open(path, 'r') as f:
    data = json.load(f)
print(len(data))
exception_list = []
ex_list = []
new_list = []
multi_labels = []
training_data = {}


def gen_ref_list(reference):
    ref_list = []
    for ref in reference:
        index = ref.find(": ")
        rf = ref[index + 1:].lstrip().rstrip().replace("\n", "").replace("\"", "")
        if rf[-1] == ",":
            rf = rf[:-1]
        ref_list.append(rf)
    # print(ref_list)
    return ref_list


ref_dic_list = []  # {"诉讼与非诉讼程序法-人民调解法2010-08-28": []}

law_dict = {}  # law_name : 小条条

for obj in data:
    try:
        labels = set()
        reference_list = obj["reference"]
        law_name = reference_list[0].split(":")[0]
        if law_name not in law_dict:
            law_dict[law_name] = set()
        ref_list = gen_ref_list(reference_list)
        for sub_ref in ref_list:
            law_dict[law_name].add(sub_ref)
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
            training_data[question] = {"law_name": labels[0], "parent_cls": parent_cls, "child_cls": child_cls,
                                       "pos": ref_list}

    except Exception as e:
        print("error ", e, reference_list)
        exception_list.append(obj)


def get_neg(po, all_samples):
    return [sample for sample in all_samples if sample not in po]

from fastbm25 import fastbm25
"""
from fastbm25 import fastbm25

corpus = [
    "张三考上了清华",
    "李四考上了北大",
    "我烤上了地瓜.",
    "我们都有光明的未来."
]
model = fastbm25(corpus)
query = "我考上了大学"
result = model.top_k_sentence(query,k=1)
print(result)
"""

def gen_bm25_negs(query_laws, corpus):
    model = fastbm25(corpus)
    ret = {}
    for query in query_laws:
        result = model.top_k_sentence(query, k=30) # threshold value is 70.00
        result = [neg_tuple for neg_tuple in result if neg_tuple[2] < 70.0]
        result = result[-5:] if len(result) >= 5 else result
        ret[query] = result

    return ret

pos_neg_dict = {}
import tqdm
count = 0
for question, compact in tqdm.tqdm(training_data.items()):
    print("ques", question)
    count += 1
    pos = compact["pos"]
    ln = compact["law_name"]
    negs = get_neg(pos, law_dict[ln])
    pos_neg_dict[question] = {"pos": pos, "neg": gen_bm25_negs(pos, negs)}

print(pos_neg_dict["一家企业申请化妆品生产许可，需要满足哪些条件？"])

# print(training_data[
#           "某企业的建设用地使用权即将到期，但该企业拟将该土地转让给其他企业。该企业需要根据哪些法规采取相应的措施？"])
# print(law_dict["公安部-公安机关办理刑事案件程序规定2020-07-20"])
# print(sum([len(law_dict[key]) for key in law_dict.keys()]))
# print(len(multi_labels))
# print(len(exception_list))
# print(pos_neg_dict["某企业的建设用地使用权即将到期，但该企业拟将该土地转让给其他企业。该企业需要根据哪些法规采取相应的措施？"])

data_dist = "pos_neg_dict.json"
with open(data_dist, 'w') as fp:
    json.dump( pos_neg_dict, fp, ensure_ascii=False)
