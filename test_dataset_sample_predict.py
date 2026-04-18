#!/usr/bin/env python3
"""
Test Dataset Sample Predict - Sử dụng đúng pipeline của TempoQR dataset
"""

import sys
import os
import torch
import pickle
import argparse
import random
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dataset_sample_predict():
    """Test lấy câu hỏi từ dataset và predict sử dụng collate_fn chuẩn"""
    print("Dataset Sample Predict Test (using correct pipeline)")
    print("=" * 60)
    
    # Load dataset
    dataset_name = 'wikidata_big'
    split = 'valid'
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Setup arguments (giống như trong dataset)
    args = argparse.Namespace()
    args.dataset_name = dataset_name
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.tkg_file = f'data/data/{args.dataset_name}/kg/full.txt'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    from core.utils import loadTkbcModel
    from core.qa_tempoqr import QA_TempoQR
    
    tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
    tkbc_model = loadTkbcModel(tkbc_model_file)
    tkbc_model = tkbc_model.to(device)
    
    qa_model = QA_TempoQR(tkbc_model, args)
    model_path = f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        qa_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        qa_model.load_state_dict(checkpoint)
    qa_model = qa_model.to(device)
    qa_model.eval()
    
    # Load dataset (sẽ tự động xử lý tokenization và annotation)
    from core.qa_datasets import QA_Dataset_TempoQR
    dataset = QA_Dataset_TempoQR(split=split, dataset_name=dataset_name, args=args)
    print(f"Loaded {len(dataset.data)} questions from {split} split")
    
    # Lấy 10 câu hỏi ngẫu nhiên
    num_samples = min(10, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    from torch.utils.data import DataLoader
    success_count = 0
    total_count = 0
    
    for idx_in_dataset in sample_indices:
        total_count += 1
        # Lấy item thô (chưa qua collate)
        item = dataset[idx_in_dataset]  # tuple gồm 11 phần tử
        question_text = item[0]
        answers_single = item[10]  # ground truth answer ID (hoặc list)
        
        # Tạo DataLoader batch_size=1 để dùng collate_fn
        loader = DataLoader([item], batch_size=1, collate_fn=dataset._collate_fn)
        batch = next(iter(loader))
        
        # Chuyển batch sang device
        batch = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in batch]
        
        # Forward
        with torch.no_grad():
            scores = qa_model.forward(batch)  # scores shape (1, num_entities+num_times)
        
        # Lấy top-10 predictions
        import numpy as np
        scores_np = scores[0].cpu().numpy()
        top_indices = np.argsort(scores_np)[-10:][::-1]
        
        # Lấy ground truth answers từ dataset.data
        question_obj = dataset.data[idx_in_dataset]
        actual_answers = question_obj.get('answers', [])
        
        # Map actual answers (có thể là Wikidata ID hoặc time int) sang text để so sánh
        from core.utils import getAllDicts
        all_dicts = getAllDicts(dataset_name)
        wd_id_to_text = all_dicts['wd_id_to_text']
        id2ent = all_dicts['id2ent']
        id2ts = all_dicts['id2ts']
        ent2id = all_dicts['ent2id']
        num_entities = len(ent2id)
        
        actual_texts = []
        for ans in actual_answers:
            if isinstance(ans, str):
                # Wikidata ID (Q...)
                if ans in wd_id_to_text:
                    actual_texts.append(wd_id_to_text[ans])
                else:
                    actual_texts.append(ans)
            elif isinstance(ans, int):
                if ans < num_entities:
                    # entity ID
                    wd = id2ent.get(ans)
                    if wd:
                        actual_texts.append(wd_id_to_text.get(wd, f"entity_{ans}"))
                    else:
                        actual_texts.append(f"entity_{ans}")
                else:
                    # time ID
                    time_idx = ans - num_entities
                    time_val = id2ts.get(time_idx, (0,0,0))[0]
                    actual_texts.append(str(time_val))
            else:
                actual_texts.append(str(ans))
        
        # In kết quả
        print(f"\n{'='*80}")
        print(f"Question: {question_text}")
        print(f"Expected answers: {actual_texts}")
        print(f"Top-10 predictions:")
        found = False
        for i, idx_pred in enumerate(top_indices, 1):
            if idx_pred < num_entities:
                wd = id2ent.get(idx_pred)
                pred_text = wd_id_to_text.get(wd, f"entity_{idx_pred}") if wd else f"entity_{idx_pred}"
                print(f"  {i}. {pred_text}")
                # Kiểm tra xem có trong expected không
                for exp in actual_texts:
                    if exp.lower() in pred_text.lower() or pred_text.lower() in exp.lower():
                        found = True
                        print(f"     *** MATCH FOUND! ***")
                        break
            else:
                time_idx = idx_pred - num_entities
                time_val = id2ts.get(time_idx, (0,0,0))[0]
                pred_text = str(time_val)
                print(f"  {i}. {pred_text}")
                if pred_text in actual_texts:
                    found = True
                    print(f"     *** MATCH FOUND! ***")
        if found:
            success_count += 1
            print("✅ SUCCESS: Expected answer in top-10")
        else:
            print("❌ FAILED: Expected answer not found")
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY: {success_count}/{total_count} correct ({success_count/total_count*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    test_dataset_sample_predict()