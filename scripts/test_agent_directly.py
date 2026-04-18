#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import random
import argparse
import re
from core.model_loader import model_manager
from core.qa_datasets import QA_Dataset_TempoQR
from core.utils import getAllDicts
from core.config import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Số câu hỏi lấy từ dataset")
    parser.add_argument("--split", type=str, default="valid", help="Split: valid, test")
    args = parser.parse_args()
    
    # Tạo args giả lập cho dataset
    import argparse as ap
    model_args = ap.Namespace()
    model_args.dataset_name = settings.DATASET_NAME
    model_args.model = 'tempoqr'
    model_args.supervision = 'soft'
    model_args.fuse = 'add'
    model_args.extra_entities = False
    model_args.frozen = 1
    model_args.lm_frozen = 1
    model_args.corrupt_hard = 0.0
    model_args.tkg_file = f'data/data/{settings.DATASET_NAME}/kg/full.txt'
    
    # Load dataset
    dataset = QA_Dataset_TempoQR(split=args.split, dataset_name=settings.DATASET_NAME, args=model_args)
    all_dicts = getAllDicts(settings.DATASET_NAME)
    wd_id_to_text = all_dicts['wd_id_to_text']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    ent2id = all_dicts['ent2id']
    num_entities = len(ent2id)
    
    # Lấy ngẫu nhiên num_samples câu hỏi
    indices = random.sample(range(len(dataset.data)), min(args.num_samples, len(dataset.data)))
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for idx in indices:
        q_obj = dataset.data[idx]
        question_raw = q_obj['question']
        # Thay Q‑id bằng tên (để câu hỏi dễ đọc, nhưng model dùng tokenization riêng)
        question = question_raw
        for qid in re.findall(r'Q[0-9]+', question):
            if qid in wd_id_to_text:
                question = question.replace(qid, wd_id_to_text[qid])
        
        # Lấy annotation từ dataset
        annotation = q_obj.get('annotation', {})
        head_qid = annotation.get('head')
        tail_qid = annotation.get('tail')
        time_str = annotation.get('time')
        
        # Chuyển head, tail sang internal ID
        head_id = 0
        if head_qid:
            if head_qid in ent2id:
                head_id = ent2id[head_qid]
            else:
                # Tìm trong wd_id_to_text
                for qid, name in wd_id_to_text.items():
                    if name == head_qid and qid in ent2id:
                        head_id = ent2id[qid]
                        break
        tail_id = 0
        if tail_qid:
            if tail_qid in ent2id:
                tail_id = ent2id[tail_qid]
            else:
                for qid, name in wd_id_to_text.items():
                    if name == tail_qid and qid in ent2id:
                        tail_id = ent2id[qid]
                        break
        if tail_id == 0:
            tail_id = head_id
        
        # Chuyển time sang start_time, end_time
        start_time = None
        end_time = None
        if time_str and time_str.isdigit():
            start_time = int(time_str)
            end_time = start_time
        
        # Dự đoán
        preds = model_manager.predict_with_time(question, k=5, head_id=head_id, tail_id=tail_id,
                                                 start_time=start_time, end_time=end_time)
        # Ground truth answers
        answers = q_obj.get('answers', [])
        expected_texts = []
        for ans in answers:
            if isinstance(ans, str) and ans.startswith('Q'):
                text = wd_id_to_text.get(ans, ans)
                expected_texts.append(text)
            elif isinstance(ans, int):
                if ans < num_entities:
                    wd = id2ent.get(ans)
                    text = wd_id_to_text.get(wd, f"entity_{ans}") if wd else f"entity_{ans}"
                    expected_texts.append(text)
                else:
                    time_idx = ans - num_entities
                    time_val = id2ts.get(time_idx, (0,0,0))[0]
                    expected_texts.append(str(time_val))
            else:
                expected_texts.append(str(ans))
        
        top1 = preds[0] if preds else ""
        if top1 in expected_texts:
            correct_top1 += 1
        if any(p in expected_texts for p in preds[:5]):
            correct_top5 += 1
        total += 1
        
        print(f"\n📝 Original: {question_raw}")
        print(f"   Converted: {question}")
        print(f"   Head: {head_qid} -> ID {head_id}")
        print(f"   Tail: {tail_qid} -> ID {tail_id}")
        print(f"   Time: {time_str}")
        print(f"   Expected: {expected_texts[:5]}{'...' if len(expected_texts)>5 else ''}")
        print(f"   Top-5 Predictions: {preds}")
        print(f"   ✅ Top-1 correct: {top1 in expected_texts}")
        print(f"   ✅ Top-5 correct: {any(p in expected_texts for p in preds[:5])}")
    
    print(f"\n{'='*60}")
    print(f"📊 Accuracy on {total} questions from {args.split} split (using dataset annotations):")
    print(f"   Top-1: {correct_top1}/{total} = {correct_top1/total*100:.2f}%")
    print(f"   Top-5: {correct_top5}/{total} = {correct_top5/total*100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()