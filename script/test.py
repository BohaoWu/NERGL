import sys
import os
# the repository root, where model/, data_processing/ and data/ live; run from it (python3 script/train.py ...)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_processing import config as cfg     # every path of the project
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']

import warnings
                                                 
warnings.filterwarnings('ignore')


from model.data_pipe import BartNERPipe
from model.bart_multi_concat import BartSeq2SeqModel  
from model.generator_multi_concat import SequenceGeneratorModel     
from model.metrics import Seq2SeqSpanMetric
from model.span_text import word_cum_lens, span_text 

import fitlog

import argparse

from fastNLP import SequentialSampler

from fastNLP import DataSetIter
from fastNLP.core.utils import _move_dict_value_to_device
import torch

from data_processing.eval import evaluate_file
from fastNLP.core.field import Padder

class ListPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        return list(contents)

fitlog.debug()
fitlog.set_log_dir(os.path.join(cfg.LOG_DIR, 'fitlog/default'))

def get_data(args, dataset_name, demo):

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type)
    if dataset_name == 'twitter-ner': 
        paths ={
            'train': os.path.join(args.datapath,'train.txt'),
            'dev': os.path.join(args.datapath,'dev.txt'),
            'test': os.path.join(args.datapath,'test.txt') }

        data_bundle = pipe.process_from_file(paths, demo=demo)
        
    return data_bundle, pipe.tokenizer, pipe.mapping2id

def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    with open (args.pred_output_file,'w') as fw:
        for (batch_x, batch_y) in data_iterator:
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            image_feature = batch_x['image_feature']
            tgt_tokens = batch_x['tgt_tokens']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            rag_seq_len = batch_x['rag_seq_len']
            first = batch_x['first']
            region_label = batch_y['region_label']
            target_span = batch_y['target_span']
            cover_flag = batch_y['cover_flag']
            rag_tokens = batch_x['rag_tokens']
            

            results = model.predict(src_tokens,image_feature,rag_tokens, src_seq_len=src_seq_len, rag_seq_len=rag_seq_len, first=first)
            
            pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
            
            pred_pairs, target_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,src_seq_len,predict_mode=True)
            
            raw_words = batch_y['raw_words']
            word_start_index = 8 ## 2 + 2 +4
            assert len(pred_pairs) == len(target_pairs)
            for i in range(len(pred_pairs)):
                cum = word_cum_lens(raw_words[i], tokenizer)
                fw.write(''.join(raw_words[i])+'\n')
                fw.write('Pred: ')
                for k,v in pred_pairs[i].items():
                    entity_span = span_text(k, raw_words[i], cum, word_start_index)
                    
                    region_pred, entity_type_ind = v
                    if entity_type_ind[0] < 10:
                        entity_type = ids2label[entity_type_ind[0]]
                    else:
                        print("Entity type error! entity_type_ind[0]:", entity_type_ind[0])
                        print("ids2label:", ids2label)
                        entity_type = ids2label[3]
                    
                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                fw.write('  GT: ')
                for k,v in target_pairs[i].items():
                    entity_span = span_text(k, raw_words[i], cum, word_start_index)
        
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]

                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n\n')
        res = metric.get_metric()  
        fw.write(str(res))
    return res

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
    parser.add_argument('--datapath', default=cfg.TXT_DIR + '/', type=str)
    parser.add_argument('--image_feature_path',default=cfg.VINVL_DIR, type=str)
    parser.add_argument('--image_annotation_path',default=cfg.XML_DIR + '/', type=str)
    parser.add_argument('--box_num',default='16', type=int)
    parser.add_argument('--model_weight',default= None, type = str)
    parser.add_argument('--normalize',default=False, action = "store_true")
    parser.add_argument('--max_len', default=30, type=int)
    parser.add_argument('--batch_size',default=16,type=int)
    parser.add_argument("--log",default=os.path.join(cfg.LOG_DIR, 'fitlog'),type=str)
    args= parser.parse_args()


    model_path = args.model_weight.rsplit('/')
    args.pred_output_file = '/'.join(model_path[:-1])+'/pred_test_'+model_path[-1]+'.txt'


    dataset_name = 'twitter-ner'
    args.length_penalty = 1
    args.target_type = 'word'
    args.schedule = 'linear'
    args.decoder_type = 'avg_feature'
    args.num_beams = 1   
    args.use_encoder_mlp = 1
    args.warmup_ratio = 0.01
    eval_start_epoch = 0


    if 'twitter' in dataset_name:  
        max_len, max_len_a = args.max_len, 0.6
    else:
        print("Error dataset_name!")


    if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
        args.decoder_type = None
    demo = False

    data_bundle, tokenizer, mapping2id = get_data(args, dataset_name, demo)

    print(f'max_len_a:{max_len_a}, max_len:{max_len}')

    print(data_bundle)

    bos_token_id = 0
    eos_token_id = 1
    label_ids = list(mapping2id.values())


    model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                        use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num)


    vocab_size = len(tokenizer)

    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id, 
                                max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                                restricter=None, top_k = 1
                                )

    # one checkpoint file, or the per-epoch files <model_weight>_<i> that train.py saves
    if os.path.isfile(args.model_weight):
        weights = [args.model_weight]
    else:
        weights = [args.model_weight + "_" + str(i) for i in range(40) if os.path.isfile(args.model_weight + "_" + str(i))]
    for i, weight in enumerate(weights):
        # mmap reads only the tensors that are used, so the optimizer state in a
        # bundled checkpoint is never pulled off the disk.
        blob = torch.load(weight, map_location="cpu", mmap=True)
        model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)



        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'


        metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), box_num =args.box_num, target_type=args.target_type,print_mode = False )


        test_dataset = data_bundle.get_dataset('test')

        test_dataset.set_target('raw_words', 'raw_target')
        test_dataset.set_padder('raw_words', ListPadder())
        test_dataset.set_padder('raw_target', ListPadder())

        device = torch.device(device)
        model.to(device)

        ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}
        model.eval()
        test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
        test_f = test_res['f']
        print(os.path.basename(weight) + " test: "+str(test_res))
        
        # eval NER
        evaluate_file(args.pred_output_file)
        
    
    
if __name__ == "__main__":
    main()