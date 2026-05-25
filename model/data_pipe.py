import re
from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from transformers import AutoTokenizer
from fastNLP.core.metrics import _bio_tag_to_spans
from fastNLP.io import DataBundle
from fastNLP.core.field import Padder

class IdentityPadder(Padder):
    """Returns the batch as a plain list without any padding."""
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        return contents

class RegionLabelPadder(Padder):
    """Pads a list-of-list-of-arrays to a 3D tensor (batch, max_entities, feat_dim)."""
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        import torch
        max_ents = max(len(c) for c in contents)
        feat_dim = contents[0][0].shape[0]
        batch = torch.zeros(len(contents), max_ents, feat_dim)
        for i, sample in enumerate(contents):
            for j, arr in enumerate(sample):
                batch[i, j] = torch.tensor(arr, dtype=torch.float)
        return batch

class CoverFlagPadder(Padder):
    """Pads a list-of-list-of-scalar-arrays to a 2D tensor (batch, max_entities)."""
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        import torch
        max_ents = max(len(c) for c in contents)
        batch = torch.full((len(contents), max_ents), fill_value=-1, dtype=torch.long)
        for i, sample in enumerate(contents):
            for j, arr in enumerate(sample):
                batch[i, j] = int(arr[0])
        return batch
import numpy as np
from itertools import chain
from fastNLP import Const
from functools import cmp_to_key
import json
from copy import deepcopy
from tqdm import tqdm
import os
import torch
import torchvision

import data_processing.config as config
import data_processing.gpt_request as gpt_request


class BartNERPipe(Pipe):
    def __init__(self,image_feature_path=None, 
                      image_annotation_path= None,
                      max_bbox =16,
                      normalize=False,
                      tokenizer='facebook/bart-base', 
                      target_type='word'):
        """

        :param tokenizer:
        :param dataset_name:
        :param target_type:
            word: 生成word的start; #仅支持
            bpe: 生成所有的bpe
            span: 每一段按照start end生成
            span_bpe: 每一段都是start的所有bpe，end的所有bpe
        """
        super().__init__()
        
        
        self.image_feature_path=image_feature_path  # vinvl 使用
        self.image_annotation_path = image_annotation_path
        
        self.max_bbox= max_bbox  
        self.max_aspect = 6
        self.region_dim=2048
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        assert target_type in ('word') #

        cur_num_tokens = self.tokenizer.vocab_size
        self.num_token_in_orig_tokenizer = cur_num_tokens
        self.target_type = target_type
        self.gpt_client = gpt_request.GPTClient()
        self.gpt_path = config.rag_gpt_file
        self.gpt_dict_path = config.rag_gpt_dict_file
        self.raw_file_path = config.raw_file
        
        self.not_cover = 0
        self.covered = 0

    def get_rag_dict_from_file(self):
        rag_dict = {}
        with open(self.gpt_dict_path, "r") as f:
            for i in f.readlines():
                line = i.split("###")
                try:
                    rag_dict[line[0]] = line[1].replace("\n","")
                except Exception as e:
                    print(line)
                    
        return rag_dict

    def add_tags_to_special_tokens(self, data_bundle):
        
        mapping ={}
         
        ## targt-region relation
        mapping['0'] ='<<which region>>'  ## 取自 region_label 的最后一个item,相关是0 不相关是1
        mapping['1'] ='<<no region>>'
        
        # mapping['loc'] = '<<location>>'
        # mapping['per'] = '<<person>>'
        # mapping['other'] = '<<others>>'
        # mapping['org'] = '<<organization>>'
        # self.mapping =mapping
        
        ### New Entity Type
        new_entity_type_1 = '<<地名>>'
        new_entity_type_2 = '<<役者>>'
        new_entity_type_3 = '<<替名>>'
        new_entity_type_4 = '<<演目>>'
        mapping['地名'] = new_entity_type_1
        mapping['役者'] = new_entity_type_2
        mapping['替名'] = new_entity_type_3
        mapping['演目'] = new_entity_type_4
        self.mapping = mapping
        

        sorted_add_tokens = list(mapping.values())
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id
        if hasattr(self.tokenizer, 'unique_no_split_tokens'):
            self.tokenizer.unique_no_split_tokens = self.tokenizer.unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}  # 给定转换后的tag，输出的是在tokenizer中的id，用来初始化表示
        self.mapping2targetid = {}  # 给定原始tag，输出对应的数字

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.num_token_in_orig_tokenizer
            self.mapping2id[value] = key_id[0]  
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def read_image_label(self,img_id):
        import xml.etree.ElementTree as ET
        fn=os.path.join(self.image_annotation_path,img_id+'.xml')
        tree=ET.parse(fn)
        root=tree.getroot()
        aspects = []
        boxes = []
        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_name=names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    xmin = int(box_container[0].findall('xmin')[0].text) 
                    ymin = int(box_container[0].findall('ymin')[0].text) 
                    xmax = int(box_container[0].findall('xmax')[0].text) 
                    ymax = int(box_container[0].findall('ymax')[0].text) 
                aspects.append(box_name)
                boxes.append([xmin,ymin,xmax,ymax])
        return aspects, boxes

    def get_tokens_from_words(self, words):
        # get tokens from words
        word_bpes = [[self.tokenizer.bos_token_id]]
        first = []  # 用来取每个word第一个bpe
        cur_bpe_len = 1
        for word in words:
            bpes = self.tokenizer.tokenize(word)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            first.append(cur_bpe_len)
            cur_bpe_len += len(bpes)
            word_bpes.append(bpes)
        assert first[-1] + len(bpes) == sum(map(len, word_bpes))
        word_bpes.append([self.tokenizer.eos_token_id])
        assert len(first) == len(words) == len(word_bpes) - 2   ## raw_word

        lens = list(map(len, word_bpes))
        cum_lens = np.cumsum(lens).tolist()
        first = list(range(cum_lens[-1]))  ## first 只掩码sentence内部的

        return word_bpes, first, cum_lens, lens
    
    def get_tokens_from_rag_words(self, words):
        # get tokens from words
        word_bpes = [[self.tokenizer.bos_token_id]]
        first = []  # 用来取每个word第一个bpe
        cur_bpe_len = 1
        matches = re.findall(r'「(.*?) (.*?)」', words)
        for word, ner_category in matches:
            bpes = self.tokenizer.tokenize(word)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)
            ner_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<<"+ner_category+">>"))
            first.append(cur_bpe_len)
            cur_bpe_len += len(ner_id)
            word_bpes.append(ner_id)
        # assert first[-1] + len(ner_id) == sum(map(len, word_bpes))
        word_bpes.append([self.tokenizer.eos_token_id])
        # assert len(first) == len(words) == len(word_bpes) - 2   ## raw_word

        lens = list(map(len, word_bpes))
        cum_lens = np.cumsum(lens).tolist()
        first = list(range(cum_lens[-1]))  ## first 只掩码sentence内部的

        return word_bpes, first, cum_lens, lens

    def process(self, data_bundle):
        
        self.add_tags_to_special_tokens(data_bundle)  

        # 转换tag
        target_shift = len(self.mapping) + 2  

        def prepare_target(ins):
            img_id = ins['img_id']
            
            image_num = 0
            image_tag =''
            image_boxes = np.zeros((self.max_bbox,4),dtype= np.float32)
            image_feature = np.zeros((self.max_bbox, self.region_dim),dtype= np.float32)
            
            if self.image_feature_path:
                ########### Vinvl image feature
                try:
                    img=np.load(os.path.join(self.image_feature_path,str(img_id)+'.jpg.npz'), allow_pickle=True)
                    image_num = img['num_boxes']
                    image_feature_ = img['box_features']
                    if self.normalize:
                        image_feature_ = (image_feature_/np.sqrt((image_feature_**2).sum()))  
                    final_num = min(image_num,self.max_bbox)
                    image_feature[:final_num] = image_feature_[:final_num]
                    image_boxes[:final_num] = img['bounding_boxes'][:final_num]
                    print("image feature loaded."+str(img_id)) 
                except Exception as e:
                    print("Exception:", e)
                    print("no image feature"+str(img_id)) 
            else:
                print("image feature error!")

            raw_words = ins['raw_words']
            for i in range(len(raw_words)):
                if raw_words[i] == ' ':
                    raw_words[i] = '_'
            word_bpes, first, cum_lens, lens = self.get_tokens_from_words(raw_words)
            
            with open(self.raw_file_path, "a") as gpt_file:
                gpt_file.write(img_id + "###" + "".join(raw_words) + "\n")
            
            # Get rag_words
            if config.rag_method == 0:
                 rag_words = ""
            elif config.rag_method == 1:
                # Get from dictionary
                # ------------------
                rag_dict = self.get_rag_dict_from_file()
                rag_words = rag_dict[img_id]
                # ------------------
            elif config.rag_method == 2:
                # Get directly
                # ------------------
                rag_words = self.gpt_client.getNERResultStrResponseFomGPT("".join(raw_words))
                with open(self.gpt_path, "a") as gpt_file:
                    gpt_file.write(img_id + "###" + rag_words + "\n")
                # ------------------
            elif config.rag_method == 3:
                # Get description from named entity
                # ------------------
                rag_words = self.gpt_client.getNERTagResultResponseFomGPT("".join(raw_words), entity=entities_list)
                with open(self.gpt_path, "a") as gpt_file:
                    gpt_file.write(img_id + "###" + rag_words + "\n")
            
            elif config.rag_method == 4:
                # Get description from named entity with entity info
                # ------------------
                entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]
                entities_list = []
                for i,e in enumerate(entities):
                    entities_list.append(''.join(e))
                with open(self.gpt_path, "a") as gpt_file:
                    for rag_word, entity in zip(rag_words, entities_list):
                        gpt_file.write(img_id + "###" + entity + "###" + rag_word + "\n")
                rag_words = "".join(rag_words)
                # -------------------------
            elif config.rag_method == 5:
                # Get description from named entity with entity info by claude
                # ------------------
                rag_words = self.gpt_client.getNERResultStrResponseFomClaude("".join(raw_words))
                with open(self.gpt_path, "a") as gpt_file:
                    gpt_file.write(img_id + "###" + rag_words + "\n")
            
            rag_word_bpes, rag_first, rag_cum_lens, rag_lens = self.get_tokens_from_rag_words(rag_words)

            ###### image_label #######
            aspect_ious_dic ={}   ## {aspect:[iou1,iou2,...]}
            xml_file_path = os.path.join(self.image_annotation_path,img_id+'.xml')
            if os.path.exists(xml_file_path):
                names, gt_boxes = self.read_image_label(img_id)
                assert len(names) > 0, img_id
                IoUs=(torchvision.ops.box_iou(torch.tensor(gt_boxes),torch.tensor(image_boxes))).numpy() #[x,4],[16,4]  ->[x,16]
                
                for i,nn in enumerate(names):  ## 对于每个标注框
                    cur_iou = IoUs[i]
                    # if max(cur_iou) < data_processing.config.cur_iou_score_thresh: ## object detector 没有检测到
                    if max(cur_iou) < config.cur_iou_score_thresh: ## object detector 没有检测到
                        self.not_cover +=1  ## ps: not_cover 这个数量是针对每个标注框的，不是每个实体的。
                        if nn not in aspect_ious_dic: ## 首次出现这个name，赋 -1；如果之前已经有这个name的记录，此处都不再关注
                            aspect_ious_dic[nn] = np.array([-1])  
                    else:
                        self.covered +=1
                        if nn in aspect_ious_dic:  ## 如果一个aspect对应多个标注框，更新iou
                            last_iou = aspect_ious_dic[nn]
                            if last_iou[0] == -1:  
                                aspect_ious_dic[nn] = cur_iou ## 直接赋当前iou
                            else: ## 该aspect有多个标注框，且多个框被检测到
                                final_iou = np.array([max(last_iou[i],cur_iou[i]) for i in range(len(last_iou))])
                                aspect_ious_dic[nn] = final_iou
                        else:
                            aspect_ious_dic[nn] = cur_iou
            else:
                print(img_id + ".xml file is not existed.")
            
            
           
            region_label = []
            cover_flag = []  ## 0:entity-region 相关，但 detector 没有检测到 ；1: entity-region 相关，且检测到；2:entity-region 不相关
            entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]
            for i,e in enumerate(entities):
                e = ''.join(e)
                if e in aspect_ious_dic:
                    ori_ious = aspect_ious_dic[e]
                    ### 处理notcover
                    if ori_ious[0] == -1:
                        average_iou = 0.
                        region_label.append(np.array([average_iou]*self.max_bbox + [1.]))   ## 按照不相关训练
                        cover_flag.append(np.array([0])) ## 按照相关评估 ## 
                    else:
                        keeped_ious = np.array([iou if iou >0.5 else 0 for iou in ori_ious])
                        norm_iou = keeped_ious / float(sum(keeped_ious))
                        region_label.append(np.append(norm_iou,[0.]))
                        cover_flag.append(np.array([1]))
                    
                else:
                    average_iou = 0.  # 0. # 1 / self.max_bbox
                    region_label.append(np.array([average_iou]*self.max_bbox + [1.]))  
                    cover_flag.append(np.array([2]))
            
            ## 全 O 是 [], 会报错，先pad一个
            if len(region_label) ==0:
                region_label.append(np.array([0.]*(self.max_bbox +1)))
            if len(cover_flag) ==0:
                cover_flag.append(np.array([2]))

            
            entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]
            entity_tags = ins['entity_tags']  # [tag1, tag2...]
            target = [0]  
            pairs = []
            
            assert len(entity_spans) == len(entity_tags)
            _word_bpes = list(chain(*word_bpes))
            _rag_word_bpes = list(chain(*rag_word_bpes))
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                for i in range(num_ent):
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    if self.target_type == 'word':
                        cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    elif self.target_type == 'span':
                        cur_pair_.append(cum_lens[start])
                        cur_pair_.append(cum_lens[end]-1)  # it is more reasonable to use ``cur_pair_.append(cum_lens[end-1])``
                    elif self.target_type == 'span_bpe':
                        cur_pair_.extend(
                            list(range(cum_lens[start], cum_lens[start + 1])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                        cur_pair_.extend(
                            list(range(cum_lens[end - 1], cum_lens[end])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                    elif self.target_type == 'bpe':
                        cur_pair_.extend(list(range(cum_lens[start], cum_lens[end])))
                    else:
                        raise RuntimeError("Not support other tagging")
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                    if 'word' == self.target_type or word_idx != -1:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx])[:1])[0]
                    else:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx])[-1:])[0]
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])
                               
                cur_pair.append(self.mapping2targetid[str(int(region_label[idx][-1]))] +2)  ##  entity-region relation
                cur_pair.append(self.mapping2targetid[tag] + 2)  # 加2是由于有shift 
                ### ↑ [span, <<which region>>(+Linear), type] / [span, <<no region>>, type]                
                
                pairs.append([p for p in cur_pair])
                
            target.extend(list(chain(*pairs)))
            target.append(1)  # 特殊的eos

            word_bpes = list(chain(*word_bpes))
            rag_word_bpes = list(chain(*rag_word_bpes))
            if config.switch_rag:
                pass
            else:
                pass
                # rag_word_bpes = [1]
            
            
            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first,'image_tag':image_tag, 'image_feature':image_feature,'region_label':region_label,'cover_flag':cover_flag,'rag_tokens': rag_word_bpes}
            return dict

        data_bundle.apply_more(prepare_target, use_tqdm=False, tqdm_desc='pre. tgt.')  

        data_bundle.set_ignore_type('target_span', 'entities')
        data_bundle.set_ignore_type('image_tag')
        data_bundle.set_ignore_type('region_label', 'cover_flag')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='rag_tokens', new_field_name='rag_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'rag_seq_len', 'first','image_feature','image_tag','rag_tokens')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities','region_label','cover_flag')
        # Use custom padders for variable-length fields
        identity_padder = IdentityPadder()
        region_padder = RegionLabelPadder()
        cover_padder = CoverFlagPadder()
        for _, dataset in data_bundle.iter_datasets():
            for fname in ('target_span', 'entities'):
                if dataset.has_field(fname):
                    dataset.set_padder(fname, identity_padder)
            if dataset.has_field('region_label'):
                dataset.set_padder('region_label', region_padder)
            if dataset.has_field('cover_flag'):
                dataset.set_padder('cover_flag', cover_padder)
        print("not_cover: %d"%(self.not_cover))
        print("covered: %d"%(self.covered))
        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        
        # 读取数据
        if isinstance(paths, str):
            path = paths
        else:
            path = paths['train']
        
        data_bundle = TwitterNer(demo=demo).load(paths)
        
        data_bundle = self.process(data_bundle)
        
        return data_bundle



class TwitterNer(ConllLoader):
   

    def __init__(self, demo=False):
        headers = [
            'raw_words', 'target',
        ]
        # most of the data should put the label in the last column.
        super().__init__(headers=headers, indexes=[0, -1])
        self.demo = demo
    def read_file(self,filename):
        
        f=open(filename)
        data=[]
        raw_data=[]
        target=[]
        coarse_target=[]
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(raw_data) > 0:
                    #import pdb;pdb.set_trace()
                    data.append((raw_data,target, coarse_target))
                    raw_data = []
                    target = []
                    coarse_target=[]
                continue
            splits = line.split('\t')
            if len(splits) == 1: ## Imageid
                raw_data.append(splits[0][:-1])
            else:
                raw_data.append(splits[0])
                target.append(splits[-1][:-1]) 
                coarse_target.append("O")
        if len(raw_data) >0:
            data.append((raw_data,target, coarse_target))
            raw_data = []
            target = []
            coarse_target=[]
        print("The number of samples: "+ str(len(data)))
        return data


    def _load(self, path):
        
        ds = DataSet()
        data = self.read_file(path)
        for raw_words, target, coarse_target in data:
            img_id = raw_words[0][6:]
            raw_words = raw_words[1:]  #去除第一个token raw_words[0]='IMGID:XXX
            target = iob2(target)      #同上
            spans = _bio_tag_to_spans(target)  #Example:('person_other', (8, 10))  #从0开始
            coarse_target = iob2(coarse_target)
            entities = []
            entity_tags = []
            entity_spans = []
            for tag, (start, end) in spans:
                entities.append(raw_words[start:end])
                entity_tags.append(tag.lower())
                entity_spans.append([start, end])

            ds.append(Instance(img_id=img_id, raw_words=raw_words, entities=entities, entity_tags=entity_tags,  
                               entity_spans=entity_spans, raw_target=target, coarse_target=coarse_target))
            if self.demo and len(ds) > 30:
                break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds





if __name__ == '__main__':
    data_bundle = TwitterNer(demo=False).load('data/twitter')
    BartNERPipe(target_type='word', dataset_name='twitter').process(data_bundle)
