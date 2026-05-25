from typing import List, Tuple, Set
import re
from sklearn.metrics import classification_report

Entity = Tuple[str, Tuple[int], str]

def create_character_labels(len, one):
    labels = ['O'] * len
    for i in one:
        for j in range(i[1][0] + 1, i[1][1]):
            labels[j] = i[2]
    return labels


def compute_classification_report(preds: List[List[Entity]], gts: List[List[Entity]], title_lens):
    y_true = []
    y_pred = []

    for pred_list, gold_list, title_len in zip(preds, gts, title_lens):
        gold_set = {(text, span, label) for (text, span, label) in gold_list}
        pred_set = {(text, span, label) for (text, span, label) in pred_list}
        

        true = create_character_labels(title_len, gold_set)
        y_true += true
        
        pre = create_character_labels(title_len, pred_set)
        y_pred += pre
                
    labels = sorted(set(y_true + y_pred) - {"O"})

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        digits=4,
        zero_division=0  # 防止除以零报错
    )
    print(report)
    
def extract_spans_and_labels(pred_line: str, text_line: str) -> List[Entity]:
    entity_pattern = r'\(\s*([^,]+?)\s*,\s*\[[^\]]*\]\s*,\s*<<([^>]+)>>\s*\)'
    matches = re.findall(entity_pattern, pred_line)

    results = []
    for entity_text, label in matches:
        start = text_line.find(entity_text)
        if start == -1:
            print(f"⚠️ 实体未找到: {entity_text}")
            continue
        end = start + len(entity_text)
        results.append((entity_text, (start, end), label))  # ✅ tuple, not dict
    return results

def parse_entities(line: str, text: str) -> List[Entity]:
    """
    Parse entity line like:
    Pred: (いかみの権太 , [18] , <<kaena>> ) (梶原景時 , [18] , <<kaena>> )
    """
    entities = []
    entities = extract_spans_and_labels(line, text)
    return entities

def compute_micro_f1(preds: List[List[Entity]], gts: List[List[Entity]]):
    pred_set: Set[Entity] = set()
    gold_set: Set[Entity] = set()

    for pred_list in preds:
        pred_set.update(pred_list)
    for gold_list in gts:
        gold_set.update(gold_list)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_file(filepath: str):
    preds_all = []
    gts_all = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pred = []
    gt = []
    title_len =[]

    for line in lines:
        line = line.strip()
        if "useful_correct" in line:
            continue
        if line.startswith("Pred:"):
            pred = parse_entities(line, text)
            preds_all.append(pred)
        elif line.startswith("GT") or line.startswith("GT:"):
            gt = parse_entities(line, text)
            gts_all.append(gt)
        else:
            text = line
            if len(text) > 0:
                title_len.append(len(text))
            

    compute_classification_report(preds_all, gts_all, title_len)

def main():
    # 示例调用
    evaluate_file('/root/GMNER/saved_model/pred_best_model.txt')
    
if __name__ == "__main__":
    main()