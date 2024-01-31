import tqdm
import json, random, csv

path = "/ai/ld/remote/rag/code/sentence-transformer/pos_neg_dict.json"
with open(path , 'r') as f:
    data = json.load(f)
    
with open("/ai/ld/remote/rag/code/sentence-transformer/training_data.json", "r") as rf:
    raw_data = json.load(rf)
    

valid_set = ["民法典", "刑法"]
def gen_pairs(query, neg):
    pass
    # for pos, negs in neg.items():
    #     [query, pos, 1]
    #     for neg_list in negs:
    #         (query , neg_list[0], 0)
        

with open("training_data_sample.tsv","w") as csvfile: 
    writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n')

    #先写入columns_name
    writer.writerow(["split",  "score", "query","item"])
    #写入多行用writerows
    all_rows = []
    count = 0
    for query, obj in tqdm.tqdm(data.items()):
        count += 1
        if count <= 10000:
            split = "train"
            if random.random() <= 0.1:
                if random.random() <= 0.5:
                    split = "test"
                else:
                    split = "dev"
            neg = obj["neg"]
            for pos, negs in neg.items():
                all_rows.append([split, 1.0, query, pos])
                for neg_list in negs:
                    all_rows.append([split, 0.0, query, neg_list[0]])
    
    writer.writerows(all_rows)


    