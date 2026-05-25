### original metadata of ukiyoe
original_data = "./demo_data/original_data.json"
outputfile_path = "./demo_data/tmp.json"

picture_path = "./ukiyoe_picture/"
download_url = "./demo_data/download_url.txt"

figure_path = "./figure/"

# data_pipe.py
cur_iou_score_thresh = 0.5

# whether rag
switch_rag = True
raw_file = "/root/GMNER/saved_model/raw_file.txt"
rag_gpt_file = "/root/GMNER/saved_model/gpt_ner_res_4_1_tag_without_hint_repair.txt"
rag_gpt_dict_file = rag_gpt_file


# how to get rag infomation
# 0:Without RAG
# 1:from dict file
# 2:from gpt dirrectly
# 3:Get description from named entity
# 4:Get description from named entity with entity info
# 5:Get description from named entity by claude
rag_method = 1
