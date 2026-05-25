# coding:utf-8
import json
import random
import re

def convert_data_format(annotated_data):
    outputs = []
    for data in annotated_data:
        if data["annotations"] == []:
            continue
        entities = []
        for annotations in data["annotations"]:
            for result in annotations["result"]:
                if result["type"] == "labels":
                    if result["value"]["labels"][0] == "役者名" \
                    or result["value"]["labels"][0] == "替名" \
                    or result["value"]["labels"][0] == "地名" \
                    or result["value"]["labels"][0] == "力士名" \
                    or result["value"]["labels"][0] == "演目" \
                    or result["value"]["labels"][0] == "座本" \
                    or result["value"]["labels"][0] == "絵師" \
                    or result["value"]["labels"][0] == "シリーズ名" \
                    or result["value"]["labels"][0] == "屋号" \
                    or result["value"]["labels"][0] == "武将":
                        entities.append(
                            {
                                "name": result["value"]["text"],
                                "span": [
                                    result["value"]["start"],
                                    result["value"]["end"],
                                ],
                                # "type_id": label2id[result["value"]["labels"][0]],
                                "type": result["value"]["labels"][0]
                            }
                        )
        if entities != []:
            entities = sorted(entities, key=lambda x: x["span"][0])
        if len(data["data"]["text"]) > 512:
            continue
        outputs.append(
            {
                "curid": data["id"],
                "text": data["data"]["text"].replace("　"," "),
                "id": data["data"]["id"],
                "entities": entities,
            }
        )
    return outputs

def find_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value]

def trans():
    # read file
    dataset_dir = str("/root/GMNER/data_processing/trans/res.json")
    with open(dataset_dir, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    dataset = convert_data_format(source_data)
    
    # read tag
    tag_dir = str("/root/GMNER/data_processing/trans/tag.json")
    with open(tag_dir, "r", encoding="utf-8") as f:
        tag_data = json.load(f)
        
    # build new dataset format
    x_data = []
    y_data = []
    ukiyoe_id_data = []
    gold_res = {}
    random.seed(22)
    random.shuffle(dataset)
    for data in dataset:
        x_data.append([i for i in data['text']])
        ukiyoe_id_data.append(data['id'])
        label_list = ["O" for i in range(0, len(data['text']))]
        gold_res[data['id']] = label_list
        for entity in data['entities']:
            if entity["type"] == "役者名":
                label = 1
            elif entity["type"] == "替名":
                label = 3
            elif entity["type"] == "地名":
                label = 5
            elif entity["type"] == "力士名":
                label = 7
            elif entity["type"] == "演目":
                label = 9
            elif entity["type"] == "座本":
                label = 11
            elif entity["type"] == "絵師":
                label = 13
            elif entity["type"] == "シリーズ名":
                label = 15
            elif entity["type"] == "屋号":
                label = 17
            elif entity["type"] == "武将":
                label = 19
            elif entity["type"] == "イベント名":
                label = 21
            else:
                continue

            label_list[entity['span'][0]] = str(find_keys_by_value(tag_data, label)[0])
            if entity['span'][1] == entity['span'][0]:
                continue
            # aviod enmoku
            else:
                for index in range(entity['span'][0] + 1, entity['span'][1]):
                    label_list[index] = str(find_keys_by_value(tag_data, label + 1)[0]) 
        y_data.append(label_list)
        
    return gold_res

def get_dict_from_file(file_path):
    dict = {}
    with open(file_path, "r") as f:
        for i in f.readlines():
            line = i.split("###")
            try:
                dict[line[0]] = line[1].replace("\n","")
            except Exception as e:
                print(line)
                
    return dict

def llm_pre_res(llm_res_path, gold_res_path):
    llm_res_dict = get_dict_from_file(file_path=llm_res_path)
    raw_res_dict = get_dict_from_file(file_path=gold_res_path)
    
    llm_pred = []
    gold_res = []
    gold_res_dict = trans()
    
    for i, (key, value) in enumerate(llm_res_dict.items()):
        true_res = raw_res_dict[key]
        llm_res_dict = value
        matches = re.findall(r'「(.*?)」', llm_res_dict)
        for m in matches:
            pre_res = ['O'] * len(true_res)
            m = m.replace("_", " ")
            m = m.split(" ")
            text = m[0]
            label = m[1]
            span = true_res.find(text)
            for i in range(len(text)):
                pre_res[span + i] = label
        llm_pred += pre_res
        gold_res += gold_res_dict[key]
            
            
        
    return llm_pred


        
def main():
    llm_res_path = "/root/GMNER/saved_model/gpt_ner_res_4_1_tag_without_hint_repair.txt"
    gold_res_path = "/root/GMNER/saved_model/raw_file.txt"
    res = llm_pre_res(llm_res_path=llm_res_path, gold_res_path=gold_res_path)
    print(res)
    return

if __name__ == "__main__":
    main()