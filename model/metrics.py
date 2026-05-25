
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
import numpy as np


class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, box_num,target_type='bpe',print_mode = False):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2  # +2是由于有前面有两个特殊符号，sos和eos
        self.box_num = box_num

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0
        self.uc = 0
        self.nc = 0
        self.tc = 0
        self.sc = 0
        self.sp = 0
        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的
        self.print_mode = print_mode
        
        self.pre_labels = []
        self.true_labels = []
        
        # NER labels
        self.labels = []
    
    def create_label_seq(self, true_seq, pre_seq, seq_len):
        """文字ベースでラベルのlistを作成"""
        # "O"のラベルで初期化したラベルのlistを作成する
        labels = ["O"] * seq_len
        for i in true_seq: # 各固有表現を処理する
            for j in range(len(i[:-2])):
                labels[j] = i[-1]
        self.true_labels += labels
        return

    def evaluate(self, target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,src_seq_len,predict_mode = False):
       
        region_pred = region_pred[:,1:,:].tolist()
        bbox_num = region_label.size(-1) -1  ## -1维度的最后一个item 0/1 表示 是否有region

        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        # pred_spans = []
        batch_pred_pairs =[]
        batch_target_pairs =[]
        for i, (true_seq, pred_seq, seq_len) in enumerate(zip(target_span, pred.tolist(), src_seq_len)):
            if not isinstance(true_seq,list):  ####!!! 有的过来是array 有的过来是list
                true_seq = true_seq.tolist()
            em = 0
            pred_seq = pred_seq[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            all_pairs = {}
            cur_pair = []
            # self.create_label_seq(true_seq, pred_seq, seq_len)
            if len(pred_seq):
                k = 0
                while k < len(pred_seq)-2:
                    if pred_seq[k]<self.word_start_index: # 是类别预测
                        if len(cur_pair) > 0:  # 之前有index 预测，且为升序，则添加pair
                            if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                
                                if pred_seq[k] == 2: ## 相关
                                    all_pairs[tuple(cur_pair)] = [region_pred[i][k],[pred_seq[k+1]]] 
                                elif pred_seq[k] == 3: ## 不相关
                                    all_pairs[tuple(cur_pair)] = [[bbox_num],[pred_seq[k+1]]] 
                                else:
                                    # print("region relation error!")
                                    pass
                        cur_pair = []
                        k = k+2
                    else: # 记录当前 pair 的index 预测
                        cur_pair.append(pred_seq[k])
                        k= k+1
                if len(cur_pair) > 0:
                    if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                        
                        if pred_seq[k] == 2:
                            all_pairs[tuple(cur_pair)] = [region_pred[i][k],[pred_seq[k+1]]]  ## 相关
                        elif pred_seq[k] == 3:
                            all_pairs[tuple(cur_pair)] = [[bbox_num],[pred_seq[k+1]]]   ## 不相关
                        else:
                            print("region relation error!")

           
            all_ts = {}
           
            for e in range(len(true_seq)):  ## i -> sample ,e -> entity
               
                if cover_flag[i][e] == 0: ## not cover
                    true_region =[bbox_num+1]
                elif cover_flag[i][e] == 2: ## 不相关
                    if region_label[i][e][-1] == 1 : ## no region
                        true_region = [bbox_num]
                    else:
                        import pdb;pdb.set_trace()
                elif cover_flag[i][e] == 1:  ## 相关
                    if region_label[i][e][-1] == 0 :
                        true_region = region_label[i][e].nonzero().squeeze(1).tolist()
                    else:
                        import pdb;pdb.set_trace()
                
                text_span = true_seq[e][:-2]
                entity_type = true_seq[e][-1]
               
                all_ts[tuple(text_span)] = [true_region,[entity_type]]

            
         

            tp,fp,fn,uc, nc, tc, sc, sp = _compute_tp_fn_fp(all_pairs, all_ts,self.box_num)
            if self.print_mode:
                print("all_pairs: "+str(all_pairs))
                print("all_ts: "+str(all_ts))
                print('tp: %d fp: %d  fn: %d'%(tp,fp,fn))
            
            
            
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.uc += uc
            self.nc += nc
            self.tc += tc
            self.sc += sc
            self.sp = sp
            
            batch_pred_pairs.append(all_pairs)
            batch_target_pairs.append(all_ts)
        
        if predict_mode:
            return batch_pred_pairs,batch_target_pairs
            

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        res['f'] = round(f*100, 2)
        res['rec'] = round(rec*100, 2)
        res['pre'] = round(pre*100, 2)
        res['em'] = round(self.em/self.total, 4)
        res['useful_correct'] =round(self.uc)
        res['noregion_correct'] =round(self.nc)
        res['type_correct'] =round(self.tc)
        res['span_correct'] =round(self.sc)
        res['supports'] =round(self.sp)
        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0
            self.uc =0
            self.nc =0
            self.tc =0
            self.sc =0
        return res


def _compute_tp_fn_fp(pred_seq, true_seq,box_num):
    
    supports = len(true_seq)
    pred_sum = len(pred_seq)
    correct_num = 0
    useful_correct = 0
    noregion_correct = 0
    span_correct = 0
    type_correct = 0
    
    tp_mner = 0
    fp_mner = 0
    fn_mner = 0
    for k,v in pred_seq.items():
        span = k
        region_pred, entity_type = v
        if span in true_seq:
            r,e = true_seq[span]
            if set(e) == set(entity_type) and len(set(region_pred) & set(r)) != 0:
               
                if box_num not in set(r):
                    useful_correct +=1 
                else:
                    noregion_correct +=1
                correct_num += 1
            if set(e) == set(entity_type):
                type_correct +=1
            span_correct += 1 
        
    
    tp = correct_num
    fp = pred_sum - correct_num
    fn = supports - correct_num
    
    return tp,fp,fn,useful_correct,noregion_correct,type_correct,span_correct,supports




