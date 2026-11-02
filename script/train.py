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
from model.metrics import Seq2SeqSpanMetric, _compute_tp_fn_fp
from model.losses import get_loss
from model.span_text import word_cum_lens, span_text

import fitlog

from torch import optim
from fastNLP import BucketSampler, SequentialSampler

from fastNLP import DataSetIter
from fastNLP.core.utils import _move_dict_value_to_device

from data_processing.get_figure import get_figure_one, get_figure
from fastNLP.core.field import Padder

class ListPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        return list(contents)

fitlog.debug()
fitlog.set_log_dir(os.path.join(cfg.LOG_DIR, 'fitlog/default'))




import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default=cfg.BART_DIR, type=str)
parser.add_argument('--datapath', default=cfg.TXT_DIR + '/', type=str)
parser.add_argument('--image_feature_path',default=cfg.VINVL_DIR + '/', type=str)
parser.add_argument('--image_annotation_path',default=cfg.XML_DIR + '/', type=str)
parser.add_argument('--region_loss_ratio',default='1.0', type=float)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--use_kl',default=False,action ="store_true")
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument("--save_model",default=1,type=int)
parser.add_argument("--save_path",default='save_models/best',type=str)
parser.add_argument("--log",default=os.path.join(cfg.LOG_DIR, 'fitlog'),type=str)
args= parser.parse_args()


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

def get_data():

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

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print(data_bundle)
print("The number of tokens in tokenizer", tokenizer.vocab_size)
print(tokenizer.convert_ids_to_tokens([0, 1, 2]))

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

## parameter scale
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
##

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(args.seed)


parameters =[]
params = {'lr':args.lr}
params['params'] = [param for name, param in model.named_parameters() ]
parameters.append(params)

optimizer = optim.AdamW(parameters)


metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), box_num =args.box_num, target_type=args.target_type,print_mode = False )

train_dataset = data_bundle.get_dataset('train')
eval_dataset = data_bundle.get_dataset('dev')
test_dataset = data_bundle.get_dataset('test')
print(train_dataset[:3])

device = torch.device(device)
model.to(device)


def load_checkpoint(path, model, optimizer=None):
    """Read a checkpoint written either as a bundle or as a bare state dict."""
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "model" in blob:
        model.load_state_dict(blob["model"])
        if optimizer is not None and blob.get("optimizer") is not None:
            optimizer.load_state_dict(blob["optimizer"])
        return int(blob.get("epoch", -1))
    model.load_state_dict(blob)
    return -1


def Training(args, train_idx, train_data, model, device, optimizer):
    
    train_sampler = BucketSampler(seq_len_field_name='src_seq_len',batch_size=args.batch_size)   # 带Bucket的 Random Sampler. 可以随机地取出长度相似的元素
    train_data_iterator = DataSetIter(train_data, batch_size=args.batch_size, sampler=train_sampler)
    
    train_loss = 0.
    train_region_loss = 0.
    for batch_x, batch_y in (train_data_iterator):
        _move_dict_value_to_device(batch_x, batch_y, device=device)
        src_tokens = batch_x['src_tokens']
        image_feature = batch_x['image_feature']
        tgt_tokens = batch_x['tgt_tokens']
        src_seq_len = batch_x['src_seq_len']
        tgt_seq_len = batch_x['tgt_seq_len']
        rag_seq_len = batch_x['rag_seq_len']
        first = batch_x['first']
        region_label = batch_y['region_label']
        rag_tokens = batch_x['rag_tokens']


        results = model(src_tokens=src_tokens,image_feature=image_feature, tgt_tokens=tgt_tokens, rag_tokens=rag_tokens, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len,rag_seq_len=rag_seq_len, first=first)
        pred, region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        loss, region_loss = get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl=args.use_kl)
        
        train_loss += loss.item()
        train_region_loss += region_loss.item()

        all_loss = loss + args.region_loss_ratio * region_loss
    
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("train_loss: %f"%(train_loss))
    print("train_region_loss: %f"%(train_region_loss))
    return train_loss, train_region_loss

def Inference(args,eval_data, model, device, metric):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    for batch_x, batch_y in (data_iterator):
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

        results = model.predict(src_tokens,image_feature,rag_tokens,src_seq_len=src_seq_len, rag_seq_len=rag_seq_len, first=first)
        
        pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,src_seq_len)
    res = metric.get_metric()  ## {'f': 20.0, 'rec': 16.39, 'pre': 25.64, 'em': 0.125, 'uc': 0}
    return res


def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    with open (args.pred_output_file,'w') as fw:
        for batch_x, batch_y in (data_iterator):
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

            results = model.predict(src_tokens,image_feature,rag_tokens, src_seq_len=src_seq_len, rag_seq_len=rag_seq_len,first=first)
            
            pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
            
            pred_pairs, target_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,src_seq_len=src_seq_len,predict_mode=True)
            
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
                    if entity_type_ind[0] <= 7:
                        entity_type = ids2label[entity_type_ind[0]]
                    else:
                        print("Entity type error.")
                        entity_type = '6'
                    
                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                fw.write(' GT : ')
                for k,v in target_pairs[i].items():
                    entity_span = span_text(k, raw_words[i], cum, word_start_index)
        
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]

                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                # GMNER hits of this print under the metric's own position matching, so
                # the counts can be split by subset (e.g. genre) without re-matching text
                hit = _compute_tp_fn_fp(pred_pairs[i], target_pairs[i], metric.box_num)
                fw.write(' Hit: %d\n\n' % (hit[3] + hit[4]))
        res = metric.get_metric()  
        fw.write(str(res))
    return res



max_dev_f = 0.
max_test_f = 0.
best_dev = {}
best_test = {}
best_dev_corresponding_test = {}

train_loss_list = []
train_region_loss_list = []

train_f_value_list = []
dev_f_value_list = []
test_f_value_list = []

train_useful_correct_list = []
dev_useful_correct_list = []
test_useful_correct_list = []

# EARLY_STOP_PATIENCE=0 disables early stopping, reproducing the published
# protocol (fixed epoch budget, no early exit).
early_stop_patience = int(os.environ.get("EARLY_STOP_PATIENCE", "5"))
early_stop_min_delta = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.5"))
early_stop_best_f = -1.0
early_stop_counter = 0

# RESUME_FROM=<checkpoint> continues an earlier run: the weights and the
# optimiser state are restored and training picks up at the next epoch.
start_epoch = 0
resume_from = os.environ.get("RESUME_FROM", "")
if resume_from:
    done = load_checkpoint(resume_from, model, optimizer)
    model.to(device)
    start_epoch = done + 1
    print("resumed from %s, continuing at epoch %d" % (resume_from, start_epoch))

# The written predictions behind Tables 3 and 4 are taken at the epoch that the
# validation set selects, separately for each metric: NER F1 for the NER dump and
# GMNER F1 for the GMNER dump. Checkpoints are kept only every SAVE_EVERY epochs,
# so the dumps are written as soon as a new best appears rather than reloaded later.
dump_ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}
test_dataset.set_target('raw_words', 'raw_target')
test_dataset.set_padder('raw_words', ListPadder())
test_dataset.set_padder('raw_target', ListPadder())
best_sel = {'ner': -1.0, 'gmner': -1.0}


def dev_ner_f(d):
    """NER F1 recovered from the GMNER counts that the metric reports."""
    tp_g = d['useful_correct'] + d['noregion_correct']
    if d['pre'] == 0 or d['rec'] == 0:
        return 0.0
    npred, gold = tp_g / (d['pre'] / 100.0), tp_g / (d['rec'] / 100.0)
    p, r = d['type_correct'] / npred, d['type_correct'] / gold
    return 200 * p * r / (p + r) if p + r else 0.0


for train_idx in range(start_epoch, args.n_epochs):
    print("-"*12+"Epoch: "+str(train_idx)+"-"*12)

    model.train()
    train_loss, train_region_loss = Training(args,train_idx=train_idx,train_data=train_dataset, model=model, device=device,
                                                optimizer=optimizer)
    
    train_loss_list.append(train_loss)
    train_region_loss_list.append(train_region_loss)

    model.eval()
    dev_res = Inference(args,eval_data=eval_dataset, model=model, device=device, metric = metric)
    dev_f = dev_res['f']
    
    dev_f_value_list.append(dev_f)
    dev_useful_correct_list.append(dev_res['useful_correct'])
    print("dev: "+str(dev_res))

   
    test_res = Inference(args,eval_data=test_dataset, model=model, device=device, metric = metric)
    
    
    test_f = test_res['f']
    test_f_value_list.append(test_f)

    if args.save_path:
        for sel, score in (('ner', dev_ner_f(dev_res)), ('gmner', dev_f)):
            if score > best_sel[sel]:
                best_sel[sel] = score
                head, tail = os.path.split(args.save_path)
                args.pred_output_file = os.path.join(head, 'pred_%s_best%s.txt' % (tail, sel))
                Predict(args, eval_data=test_dataset, model=model, device=device, metric=metric,
                        tokenizer=tokenizer, ids2label=dump_ids2label)
                print("best-%s dump: epoch %d, dev %.2f -> %s" % (sel, train_idx, score, args.pred_output_file))
                # SAVE_BEST=1 also keeps the weights of the GMNER-selected epoch,
                # the model the article reports, e.g. for the demo
                if sel == 'gmner' and os.environ.get("SAVE_BEST") == "1":
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), args.save_path + "_bestgmner")
    test_useful_correct_list.append(test_res['useful_correct'])
    print("test: "+str(test_res))

    train_res = Inference(args,eval_data=train_dataset, model=model, device=device, metric = metric)
    train_f = train_res['f']
    train_f_value_list.append(train_f)
    train_useful_correct_list.append(train_res['useful_correct'])
    print("train: "+str(train_res))


    save_f = 0.0
    if dev_f >= save_f:
        max_dev_f = dev_f 
        # Keep a checkpoint every tenth epoch and the last one, instead of all
        # fifty. That is enough to carry on training from, and enough to average
        # the final weights over, while using a tenth of the disk.
        save_every = int(os.environ.get("SAVE_EVERY", "10"))
        if args.save_model and (train_idx % save_every == save_every - 1
                                or train_idx == args.n_epochs - 1):
            model_to_save = model.module if hasattr(model, 'module') else model  
            # The optimiser state goes in alongside the weights, so training can
            # carry on from here rather than restart. There is no learning-rate
            # scheduler in this loop, so there is nothing further to record.
            torch.save({"model": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": train_idx},
                       args.save_path + "_" + str(train_idx))
        best_dev = dev_res
        best_dev['epoch'] = train_idx
        best_dev_corresponding_test = test_res
        best_dev_corresponding_test['epoch'] = train_idx
    
        
   
    if test_f >= max_test_f:
        max_test_f = test_f
        best_test = test_res
        best_test['epoch'] = train_idx

    if dev_f > early_stop_best_f + early_stop_min_delta:
        early_stop_best_f = dev_f
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Early stop counter: {early_stop_counter}/{early_stop_patience} (best_dev_f={early_stop_best_f:.2f}, current={dev_f:.2f})")
    if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
        print(f"Early stopping at epoch {train_idx}: no improvement > {early_stop_min_delta} for {early_stop_patience} epochs")
        break

f_list = [train_f_value_list, test_f_value_list, dev_f_value_list]
f_list_label = ["Train Set F Value", "Test Set F Value", "Validation Set F Value"]

useful_correct_list = [train_useful_correct_list, test_useful_correct_list, dev_useful_correct_list]
useful_correct_list_label = ["Train Useful Correct", "Test Useful Correct", "Validation Useful Correct"]

# get figure of lossing; each run writes to figure/train/<save_path name>/ so runs do not overwrite each other
fig_dir = os.path.join("train", os.path.basename(args.save_path) or "default")
get_figure_one(train_loss_list, "Train Lossing", "Training Lossing", sub_dir=fig_dir)
get_figure_one(train_region_loss_list, "Train Region Lossing", "Region Lossing", sub_dir=fig_dir)
get_figure(f_list, f_list_label, title="F Value" ,ylabel="F Value", sub_dir=fig_dir)
get_figure(useful_correct_list, useful_correct_list_label, title="Useful Correct", ylabel="Number of Useful Correct", sub_dir=fig_dir)

print("                   best_dev: "+str(best_dev))
print("best_dev_corresponding_test: "+str(best_dev_corresponding_test))
print("                  best_test: "+str(best_test))

if args.save_path and args.save_model:
    print("-"*12+'Predict'+'-'*12)
    ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}

    model_path = args.save_path.rsplit('/')
    args.pred_output_file = '/'.join(model_path[:-1])+'/pred_'+model_path[-1]+'.txt'

    best_epoch = best_dev['epoch']
    load_checkpoint(args.save_path + "_" + str(best_epoch), model)
    model.to(device)

    print(test_dataset[:3])
    test_dataset.set_target('raw_words', 'raw_target')
    test_dataset.set_padder('raw_words', ListPadder())
    test_dataset.set_padder('raw_target', ListPadder())


    model.eval()
    test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
    test_f = test_res['f']
    print("test: "+str(test_res))