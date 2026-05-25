import sys
import os
sys.path.insert(0, '/root/GMNER')
os.chdir('/root/GMNER')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import base64
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from flask import Flask, render_template, jsonify, request

from model.data_pipe import BartNERPipe, IdentityPadder
from model.bart_multi_concat import BartSeq2SeqModel
from model.generater_multi_concat import SequenceGeneratorModel
from model.metrics import Seq2SeqSpanMetric
from fastNLP import DataSetIter, SequentialSampler
from fastNLP.core.utils import _move_dict_value_to_device
import data_processing.config as config

app = Flask(__name__)

# ─── Global state ───────────────────────────────────────────────
MODEL_WEIGHT = './saved_model/best_model_18'
BART_NAME    = '/root/GMNER/download_model/bart-large-japanese'
DATAPATH     = '/root/GMNER/Ukiyoe1000/txt/'
IMG_FEATURE  = '/root/GMNER/Ukiyoe1000_VinVL/'
IMG_ANNOT    = '/root/GMNER/Ukiyoe1000/xml/'
IMG_DIR      = '/root/GMNER/ukiyoe_picture/'
BOX_NUM      = 18
MAX_LEN      = 50

ENTITY_COLORS = {
    '地名': '#3B82F6',   # blue
    '役者': '#EF4444',   # red
    '替名': '#10B981',   # green
    '演目': '#F59E0B',   # amber
}

model_obj   = None
tokenizer   = None
mapping2id  = None
ids2label   = None
device      = None
test_samples = []   # list of dicts with cached predictions

# ─── Helpers ────────────────────────────────────────────────────

def load_model():
    global model_obj, tokenizer, mapping2id, ids2label, device

    pipe = BartNERPipe(
        image_feature_path=IMG_FEATURE,
        image_annotation_path=IMG_ANNOT,
        max_bbox=BOX_NUM,
        normalize=True,
        tokenizer=BART_NAME,
        target_type='word'
    )
    paths = {
        'train': os.path.join(DATAPATH, 'train.txt'),
        'dev':   os.path.join(DATAPATH, 'dev.txt'),
        'test':  os.path.join(DATAPATH, 'test.txt'),
    }
    data_bundle = pipe.process_from_file(paths, demo=False)
    tokenizer  = pipe.tokenizer
    mapping2id = pipe.mapping2id
    ids2label  = {2 + i: l.strip('<>') for i, l in enumerate(mapping2id.keys())}

    label_ids = list(mapping2id.values())
    bart = BartSeq2SeqModel.build_model(
        BART_NAME, tokenizer,
        label_ids=label_ids,
        decoder_type='avg_feature',
        use_encoder_mlp=1,
        box_num=BOX_NUM
    )
    seq_model = SequenceGeneratorModel(
        bart,
        bos_token_id=0, eos_token_id=1,
        max_length=MAX_LEN, max_len_a=0.6,
        num_beams=1, do_sample=False,
        repetition_penalty=1, length_penalty=1,
        pad_token_id=1, restricter=None, top_k=1
    )
    seq_model.load_state_dict(torch.load(MODEL_WEIGHT, map_location='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_model.to(device)
    seq_model.eval()
    model_obj = seq_model

    return data_bundle


def run_inference(data_bundle):
    """Run inference on the test set and cache results."""
    test_ds = data_bundle.get_dataset('test')
    test_ds.set_target('raw_words', 'raw_target')
    # Apply IdentityPadder so fastNLP doesn't try to pad variable-length string fields
    identity_padder = IdentityPadder()
    for fname in ('raw_words', 'raw_target'):
        if test_ds.has_field(fname):
            test_ds.set_padder(fname, identity_padder)

    metric = Seq2SeqSpanMetric(
        1, num_labels=len(list(mapping2id.values())),
        box_num=BOX_NUM, target_type='word', print_mode=False
    )

    word_start_index = 8
    results = []

    data_iter = DataSetIter(test_ds, batch_size=8, sampler=SequentialSampler())
    with torch.no_grad():
        for batch_x, batch_y in data_iter:
            _move_dict_value_to_device(batch_x, batch_y, device=device)

            src_tokens  = batch_x['src_tokens']
            img_feat    = batch_x['image_feature']
            rag_tokens  = batch_x['rag_tokens']
            src_seq_len = batch_x['src_seq_len']
            rag_seq_len = batch_x['rag_seq_len']
            first       = batch_x['first']
            tgt_tokens  = batch_x['tgt_tokens']
            region_label = batch_y['region_label']
            target_span  = batch_y['target_span']
            cover_flag   = batch_y['cover_flag']
            tgt_seq_len  = batch_x['tgt_seq_len']

            out = model_obj.predict(
                src_tokens, img_feat, rag_tokens,
                src_seq_len=src_seq_len,
                rag_seq_len=rag_seq_len,
                first=first
            )
            pred        = out['pred']
            region_pred = out['region_pred']

            pred_pairs, target_pairs = metric.evaluate(
                target_span, pred, tgt_tokens,
                region_pred, region_label, cover_flag,
                src_seq_len, predict_mode=True
            )

            raw_words_batch = batch_y['raw_words']
            img_ids = [test_ds[i]['img_id'] for i in range(
                len(results), len(results) + len(raw_words_batch)
            )]

            for i in range(len(pred_pairs)):
                cur_src = src_tokens[i].cpu().numpy().tolist()
                text    = ''.join(raw_words_batch[i])

                pred_ents = []
                for k, v in pred_pairs[i].items():
                    tok_ids = [cur_src[kk - word_start_index] for kk in k]
                    span    = tokenizer.decode(tok_ids).replace(' ', '')
                    reg, etype_ind = v
                    etype = ids2label.get(etype_ind[0], '?') if etype_ind[0] < 10 else '?'
                    pred_ents.append({'span': span, 'type': etype, 'region': reg})

                gt_ents = []
                for k, v in target_pairs[i].items():
                    tok_ids = [cur_src[kk - word_start_index] for kk in k]
                    span    = tokenizer.decode(tok_ids).replace(' ', '')
                    reg, etype_ind = v
                    etype = ids2label.get(etype_ind[0], '?')
                    gt_ents.append({'span': span, 'type': etype, 'region': reg})

                results.append({
                    'img_id':    img_ids[i],
                    'text':      text,
                    'pred_ents': pred_ents,
                    'gt_ents':   gt_ents,
                })

    return results


def img_to_b64(img_id):
    """Load ukiyo-e image → base64 string."""
    path = os.path.join(IMG_DIR, img_id + '.jpg')
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    # Resize to reasonable preview size
    w, h = img.size
    max_dim = 600
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def annotate_image(img_id, pred_ents, region_pred_list):
    """Draw colored entity-label overlay on image, return base64."""
    path = os.path.join(IMG_DIR, img_id + '.jpg')
    if not os.path.exists(path):
        return None

    img = Image.open(path).convert('RGB')
    w, h = img.size
    max_dim = 700
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size

    # Load XML bounding boxes
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(IMG_ANNOT, img_id + '.xml')
    boxes = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        orig_w = orig_h = None
        size_el = tree.getroot().find('size')
        if size_el is not None:
            orig_w = int(size_el.find('width').text)
            orig_h = int(size_el.find('height').text)
        for obj in tree.getroot().findall('object'):
            name_el = obj.find('name')
            bb = obj.find('bndbox')
            if name_el is None or bb is None:
                continue
            x1 = int(bb.find('xmin').text)
            y1 = int(bb.find('ymin').text)
            x2 = int(bb.find('xmax').text)
            y2 = int(bb.find('ymax').text)
            # Scale to display size
            if orig_w and orig_h:
                x1 = int(x1 * w / orig_w)
                y1 = int(y1 * h / orig_h)
                x2 = int(x2 * w / orig_w)
                y2 = int(y2 * h / orig_h)
            boxes.append({'name': name_el.text, 'box': (x1, y1, x2, y2)})

    draw = ImageDraw.Draw(img, 'RGBA')
    colors = list(ENTITY_COLORS.values())

    for idx, ent in enumerate(pred_ents):
        color = ENTITY_COLORS.get(ent['type'], '#9CA3AF')
        # highlight all GT boxes whose name matches the entity span
        for box_info in boxes:
            if ent['span'] in box_info['name'] or box_info['name'] in ent['span']:
                x1, y1, x2, y2 = box_info['box']
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3,
                               fill=(r, g, b, 60))
                draw.text((x1 + 3, y1 + 2), ent['span'], fill=color)

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


ETYPE_ABBREV = {'地名': 'PLACE', '役者': 'ACTOR', '替名': 'STAGE', '演目': 'PLAY'}
FONT_PATH    = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def process_title_to_tensors(title):
    """Convert a raw Japanese title string into model input tensors."""
    from itertools import chain
    raw_words = ['_' if c == ' ' else c for c in list(title)]

    word_bpes = [[tokenizer.bos_token_id]]
    for word in raw_words:
        bpes = tokenizer.tokenize(word)
        bpes = tokenizer.convert_tokens_to_ids(bpes)
        if bpes:
            word_bpes.append(bpes)
    word_bpes.append([tokenizer.eos_token_id])

    import numpy as _np
    lens      = list(map(len, word_bpes))
    cum_lens  = _np.cumsum(lens).tolist()
    first     = list(range(int(cum_lens[-1])))
    src_tokens = list(chain(*word_bpes))
    rag_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id]

    src_t     = torch.tensor([src_tokens], dtype=torch.long)
    rag_t     = torch.tensor([rag_tokens], dtype=torch.long)
    first_t   = torch.tensor([first],      dtype=torch.long)
    img_feat  = torch.zeros(1, BOX_NUM, 2048, dtype=torch.float)
    src_len_t = torch.tensor([len(src_tokens)], dtype=torch.long)
    rag_len_t = torch.tensor([len(rag_tokens)],  dtype=torch.long)

    return src_t, rag_t, first_t, img_feat, src_len_t, rag_len_t, raw_words


def parse_pred_sequence(pred_tensor, region_pred_tensor, src_tensor):
    """Parse the model's generated token sequence into entity dicts."""
    word_start_index = len(mapping2id) + 2   # 8
    target_shift     = word_start_index

    pred_list        = pred_tensor[0, 1:].cpu().tolist()          # drop leading token
    region_pred_list = region_pred_tensor[0, 1:, :].cpu().tolist()
    src_list         = src_tensor[0].cpu().tolist()

    # trim at first EOS (token id = 1)
    try:
        pred_list = pred_list[:pred_list.index(1)]
    except ValueError:
        pass

    all_pairs = {}
    cur_pair  = []
    k = 0
    while k < len(pred_list) - 2:
        token = pred_list[k]
        if token < word_start_index:                           # relation or type token
            if cur_pair and all(cur_pair[ii] < cur_pair[ii+1] for ii in range(len(cur_pair)-1)):
                region_rel = token
                type_id    = pred_list[k+1] if k+1 < len(pred_list) else 0
                region     = region_pred_list[k] if (region_rel == 2 and k < len(region_pred_list)) else [BOX_NUM]
                all_pairs[tuple(cur_pair)] = [region, [type_id]]
            cur_pair = []
            k += 2
        else:
            cur_pair.append(token)
            k += 1

    # boundary: last entity may not be followed by -2 tokens
    if cur_pair and all(cur_pair[ii] < cur_pair[ii+1] for ii in range(len(cur_pair)-1)):
        region_rel = pred_list[k]   if k   < len(pred_list) else 3
        type_id    = pred_list[k+1] if k+1 < len(pred_list) else 0
        region     = region_pred_list[k] if (region_rel == 2 and k < len(region_pred_list)) else [BOX_NUM]
        all_pairs[tuple(cur_pair)] = [region, [type_id]]

    entities = []
    for span_key, (region, type_id_list) in all_pairs.items():
        actual_idx = [idx - target_shift for idx in span_key]
        try:
            tok_ids   = [src_list[i] for i in actual_idx if 0 <= i < len(src_list)]
            span_text = tokenizer.decode(tok_ids).replace(' ', '')
        except Exception:
            span_text = '?'
        etype = ids2label.get(type_id_list[0], '?')
        entities.append({'span': span_text, 'type': etype, 'region': region})
    return entities


def get_ent_boxes(img_id, pred_ents):
    """Return per-entity XML bounding-box centers as fractions [0,1] of the original image size."""
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(IMG_ANNOT, img_id + '.xml')
    empty = [{'span': e['span'], 'type': e['type'],
               'color': ENTITY_COLORS.get(e['type'], '#9CA3AF'), 'boxes': []}
             for e in pred_ents]
    if not os.path.exists(xml_path):
        return empty

    tree   = ET.parse(xml_path)
    size_el = tree.getroot().find('size')
    orig_w = orig_h = 1
    if size_el is not None:
        orig_w = max(1, int(size_el.find('width').text))
        orig_h = max(1, int(size_el.find('height').text))

    name_boxes = {}
    for obj in tree.getroot().findall('object'):
        ne = obj.find('name');  bb = obj.find('bndbox')
        if ne is None or bb is None: continue
        name = ne.text or ''
        x1 = int(bb.find('xmin').text) / orig_w;  y1 = int(bb.find('ymin').text) / orig_h
        x2 = int(bb.find('xmax').text) / orig_w;  y2 = int(bb.find('ymax').text) / orig_h
        name_boxes.setdefault(name, []).append(
            {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})

    result = []
    for ent in pred_ents:
        matched = []
        for name, boxes in name_boxes.items():
            if ent['span'] in name or name in ent['span']:
                matched.extend(boxes)
        result.append({'span': ent['span'], 'type': ent['type'],
                        'color': ENTITY_COLORS.get(ent['type'], '#9CA3AF'), 'boxes': matched})
    return result


def load_vinvl_features(img_id):
    """Load VinVL image features for a given img_id; returns zero tensor on failure."""
    try:
        npz  = np.load(os.path.join(IMG_FEATURE, img_id + '.jpg.npz'), allow_pickle=True)
        feat = npz['box_features'].astype(np.float32)
        feat = feat / np.sqrt((feat ** 2).sum())          # normalize (same as training)
        final_num = min(int(npz['num_boxes']), BOX_NUM)
        image_feature = np.zeros((BOX_NUM, 2048), dtype=np.float32)
        image_feature[:final_num] = feat[:final_num]
        return torch.tensor(image_feature, dtype=torch.float).unsqueeze(0)
    except Exception:
        return torch.zeros(1, BOX_NUM, 2048, dtype=torch.float)


def annotate_upload_image(img_bytes, img_filename, pred_ents):
    """Draw colored entity boxes on an uploaded image; falls back to text strip if no XML."""
    from PIL import ImageFont
    import xml.etree.ElementTree as ET

    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    w, h = img.size
    if max(w, h) > 700:
        sc = 700 / max(w, h)
        img = img.resize((int(w*sc), int(h*sc)), Image.LANCZOS)
    w, h = img.size

    xml_path = os.path.join(IMG_ANNOT, img_filename + '.xml')
    boxes = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        size_el  = tree.getroot().find('size')
        orig_w = orig_h = None
        if size_el is not None:
            orig_w = int(size_el.find('width').text)
            orig_h = int(size_el.find('height').text)
        for obj in tree.getroot().findall('object'):
            ne = obj.find('name');  bb = obj.find('bndbox')
            if ne is None or bb is None: continue
            x1, y1 = int(bb.find('xmin').text), int(bb.find('ymin').text)
            x2, y2 = int(bb.find('xmax').text), int(bb.find('ymax').text)
            if orig_w and orig_h:
                x1, y1 = int(x1*w/orig_w), int(y1*h/orig_h)
                x2, y2 = int(x2*w/orig_w), int(y2*h/orig_h)
            boxes.append({'name': ne.text, 'box': (x1, y1, x2, y2)})

    try:
        font = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img, 'RGBA')

    if boxes and pred_ents:
        for ent in pred_ents:
            color = ENTITY_COLORS.get(ent['type'], '#9CA3AF')
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            for bi in boxes:
                if ent['span'] in bi['name'] or bi['name'] in ent['span']:
                    x1, y1, x2, y2 = bi['box']
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3,
                                   fill=(r, g, b, 60))
                    abbr = ETYPE_ABBREV.get(ent['type'], ent['type'])
                    lw   = len(abbr) * 8 + 6
                    draw.rectangle([x1, y1, x1+lw, y1+18], fill=(r, g, b, 220))
                    draw.text((x1+3, y1+2), abbr, fill='white', font=font)
    else:
        # No XML matched — draw colored label strip in top-left corner
        y_pos = 10
        for ent in pred_ents[:12]:
            color = ENTITY_COLORS.get(ent['type'], '#9CA3AF')
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            abbr = ETYPE_ABBREV.get(ent['type'], ent['type'])
            lw   = len(abbr) * 8 + 8
            draw.rectangle([6, y_pos, 6+lw, y_pos+22], fill=(r, g, b, 210))
            draw.text((10, y_pos+4), abbr, fill='white', font=font)
            y_pos += 28
            if y_pos > h - 30:
                break

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    samples_meta = [
        {'img_id': s['img_id'], 'text': s['text'][:30]}
        for s in test_samples
    ]
    return render_template('index.html',
                           samples=samples_meta,
                           entity_colors=ENTITY_COLORS)


@app.route('/samples_meta')
def samples_meta():
    """Return lightweight list of all test samples (img_id + text) for the picker."""
    return jsonify([
        {'idx': i, 'img_id': s['img_id'], 'text': s['text']}
        for i, s in enumerate(test_samples)
    ])


@app.route('/sample/<int:idx>')
def get_sample(idx):
    if idx < 0 or idx >= len(test_samples):
        return jsonify({'error': 'index out of range'}), 404

    s = test_samples[idx]
    img_b64     = img_to_b64(s['img_id'])
    anno_b64    = annotate_image(s['img_id'], s['pred_ents'], [])

    return jsonify({
        'img_id':    s['img_id'],
        'text':      s['text'],
        'pred_ents': s['pred_ents'],
        'gt_ents':   s['gt_ents'],
        'image_b64': img_b64,
        'anno_b64':  anno_b64,
        'ent_boxes': get_ent_boxes(s['img_id'], s['pred_ents']),
    })


@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    """Run inference on a chosen test image with a custom (potentially modified) title."""
    data   = request.get_json()
    img_id = (data.get('img_id') or '').strip()
    title  = (data.get('title')  or '').strip()
    if not img_id or not title:
        return jsonify({'error': '画像IDとタイトルが必要です'}), 400

    # Build text tensors from custom title
    src_t, rag_t, first_t, _, src_len_t, rag_len_t, raw_words = \
        process_title_to_tensors(title)

    # Load real VinVL features for the chosen image
    img_feat = load_vinvl_features(img_id)

    src_t     = src_t.to(device);    rag_t     = rag_t.to(device)
    first_t   = first_t.to(device);  img_feat  = img_feat.to(device)
    src_len_t = src_len_t.to(device); rag_len_t = rag_len_t.to(device)

    with torch.no_grad():
        out = model_obj.predict(
            src_t, img_feat, rag_t,
            src_seq_len=src_len_t, rag_seq_len=rag_len_t, first=first_t
        )

    entities = parse_pred_sequence(out['pred'], out['region_pred'], src_t)
    img_b64  = img_to_b64(img_id)
    anno_b64 = annotate_image(img_id, entities, [])

    return jsonify({
        'img_id':    img_id,
        'text':      ''.join(raw_words).replace('_', ' '),
        'pred_ents': entities,
        'image_b64': img_b64,
        'anno_b64':  anno_b64 or img_b64,
        'ent_boxes': get_ent_boxes(img_id, entities),
    })


# ─── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Demo started, PID={os.getpid()}')
    print('Loading model and data …')
    data_bundle = load_model()
    print('Running inference on test set …')
    test_samples.extend(run_inference(data_bundle))
    print(f'Ready — {len(test_samples)} test samples cached.')
    app.run(host='0.0.0.0', port=8888, debug=False)
