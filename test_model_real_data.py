import torch

# Tắt cuDNN để tránh lỗi VRAM
torch.backends.cudnn.enabled = False

import re
import argparse
import os
import random
import numpy as np
from qa_tempoqr import QA_TempoQR
from qa_datasets import QA_Dataset_TempoQR
from utils import loadTkbcModel, getAllDicts
from transformers import DistilBertTokenizer

# 🧠 BỘ VÁ LỖI ĐỘNG (MONKEY PATCH): Gọt phần thừa của DistilBERT
def patch_tempoqr_bugs():
    for method_name in ['infer_time', 'infer_head', 'infer_tail']:
        if hasattr(QA_TempoQR, method_name):
            original_method = getattr(QA_TempoQR, method_name)
            def make_patched(orig_method):
                def patched_method(self, *args, **kwargs):
                    new_args = list(args)
                    try:
                        target_dim = self.tkbc_model.embeddings[0].weight.size(1)
                    except:
                        target_dim = 512
                        
                    for idx in range(len(new_args)):
                        if isinstance(new_args[idx], torch.Tensor) and new_args[idx].size(-1) > target_dim:
                            new_args[idx] = new_args[idx][..., :target_dim]
                            
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.size(-1) > target_dim:
                            kwargs[k] = v[..., :target_dim]
                            
                    return orig_method(self, *new_args, **kwargs)
                return patched_method
            setattr(QA_TempoQR, method_name, make_patched(original_method))

def test_model_with_real_data():
    print("🧪 Testing TempoQR - The True Zenith Engine")
    print("=" * 60)
    
    patch_tempoqr_bugs()
    
    dataset_name = 'wikidata_big'
    tkbc_model = loadTkbcModel(f'models/models/{dataset_name}/kg_embeddings/tcomplex.ckpt')
    
    args = argparse.Namespace()
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.dataset_name = dataset_name
    
    args.tkg_file = f'data/data/{dataset_name}/kg/train'
    if not os.path.exists(args.tkg_file):
        alternatives = ['train_corr20.txt', 'train_corr33.txt', 'train_corr50.txt', 'full.txt']
        for alt in alternatives:
            alt_path = f'data/data/{dataset_name}/kg/{alt}'
            if os.path.exists(alt_path):
                args.tkg_file = alt_path
                break

    args.valid_freq = 1
    args.max_epochs = 20
    args.batch_size = 1
    args.lr = 2e-4
    
    device = 'cpu'
    print(f"🖥️ Loading model on device: {device}")
    
    qa_model = QA_TempoQR(tkbc_model, args)
    model_path = f'models/models/{dataset_name}/qa_models/tempoqr_full_export.pt'
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            qa_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            qa_model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        
    qa_model.eval()
    qa_model = qa_model.to(device)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("📊 Loading dataset...")
    dataset = QA_Dataset_TempoQR(split='valid', dataset_name=dataset_name, args=args)
    
    all_dicts = getAllDicts(dataset_name)
    ent2id = all_dicts['ent2id']
    id2ts = all_dicts['id2ts']
    id2ent = all_dicts['id2ent']
    
    NUM_ENT = len(ent2id)
    NUM_TS = len(id2ts)

    qnode2text = {}
    entity_text_path = f'data/data/{dataset_name}/kg/wd_id2entity_text.txt'
    if os.path.exists(entity_text_path):
        with open(entity_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qnode2text[parts[0]] = parts[1]
    
    test_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, orig_idx in enumerate(test_indices, 1):
        question_dict = dataset.data[orig_idx]
        
        q_text_readable = question_dict['question']
        q_nodes_in_q = re.findall(r'Q\d+', q_text_readable)
        for qn in set(q_nodes_in_q): 
            if qn in qnode2text:
                q_text_readable = q_text_readable.replace(qn, f"[{qnode2text[qn]}]")
                
        print(f"\n📝 Question {i}: {q_text_readable}")
        
        try:
            # 🧠 XÂY DỰNG LẠI HOÀN TOÀN CẤU TRÚC TENSOR CHUẨN (KHÔNG DÙNG HÀM ÉP DIM)
            sample = dataset[orig_idx]
            
            # 1. Tokenizer (Chuẩn 2D: [1, seq_len])
            inputs = tokenizer(sample[0], return_tensors="pt", padding=True, truncation=True, max_length=100)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            seq_len = input_ids.size(1)
            
            # 2. Lấy biến Thô từ sample
            ent_ids = sample[1] if len(sample) > 1 else []
            times_raw = sample[5] if len(sample) > 5 else []
            
            # Helper function để trích xuất list/array
            def extract_list(var):
                if isinstance(var, torch.Tensor): return var.flatten().tolist()
                elif isinstance(var, np.ndarray): return var.flatten().tolist()
                elif isinstance(var, (list, tuple)): return list(var)
                elif isinstance(var, (int, float)): return [int(var)]
                return []

            ent_list = extract_list(ent_ids)
            time_list = extract_list(times_raw)
            
            # 3. Tạo Entity/Time Mask (Chuẩn 2D: [1, seq_len])
            entity_time_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            entity_mask = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            
            all_kg_ids = ent_list + [(t + NUM_ENT) for t in time_list]
            for idx, kg_id in enumerate(all_kg_ids):
                pos = idx + 1
                if pos < seq_len:
                    kg_id = max(0, kg_id)
                    if kg_id < (NUM_ENT + NUM_TS):
                        entity_time_ids[0, pos] = kg_id
                        entity_mask[0, pos] = 1
            
            # 4. Tạo các biến Scalar (Chuẩn 1D: [1])
            def make_1d_tensor(val_list, max_val, is_time=False):
                val = val_list[0] if len(val_list) > 0 else 0
                val = max(0, int(val))
                if is_time:
                    if val >= NUM_ENT: val -= NUM_ENT
                    if val >= NUM_TS: val = 0
                else:
                    if val >= NUM_ENT: val = 0
                return torch.tensor([val], dtype=torch.long, device=device)

            heads = make_1d_tensor(ent_list, NUM_ENT)
            tails = make_1d_tensor(ent_list[1:] if len(ent_list) > 1 else ent_list, NUM_ENT)
            tails2 = make_1d_tensor(ent_list[2:] if len(ent_list) > 2 else ent_list, NUM_ENT)
            
            times = make_1d_tensor(time_list, NUM_TS, is_time=True)
            t1 = times.clone()
            t2 = times.clone()
            
            # Gói lại đúng 10 tham số
            batch_data = (input_ids, attention_mask, entity_time_ids, entity_mask, heads, tails, times, t1, t2, tails2)

            with torch.no_grad():
                scores = qa_model.forward(batch_data)
            
            top_k = 5
            _, top_indices = torch.topk(scores, k=top_k, dim=1)
            
            predicted_answers = []
            predicted_answers_raw = [] 
            for ans_idx in top_indices[0]:
                ans_idx = ans_idx.item()
                if ans_idx < NUM_ENT: 
                    q_node = id2ent.get(ans_idx, f"Entity_{ans_idx}")
                    real_name = qnode2text.get(q_node, q_node)
                    predicted_answers.append(real_name)
                    predicted_answers_raw.append(q_node)
                else: 
                    time_idx = ans_idx - NUM_ENT
                    if time_idx in id2ts:
                        t_val = str(id2ts[time_idx][0])
                        predicted_answers.append(t_val)
                        predicted_answers_raw.append(t_val)
                    else:
                        predicted_answers.append(f"Time_{time_idx}")
                        predicted_answers_raw.append(f"Time_{time_idx}")
                        
            print(f"   🎯 Top-{top_k} predictions: {predicted_answers}")
            
            actual_answers_raw = [str(ans) for ans in question_dict.get('answers', [])]
            actual_answers_readable = [qnode2text.get(ans, ans) for ans in actual_answers_raw]
            
            print(f"   ✅ Actual answers: {actual_answers_readable}")
            
            if any(pred in actual_answers_raw for pred in predicted_answers_raw[:1]):
                correct_predictions += 1
                print(f"   🎉 CORRECT! (Hit@1)")
            else:
                print(f"   ❌ INCORRECT!")
                
            total_predictions += 1
                
        except Exception as e:
            print(f"   ❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 60)
    print("📊 TESTING RESULTS")
    print("=" * 60)
    print(f"📝 Total questions tested: {total_predictions}")
    print(f"✅ Correct predictions: {correct_predictions}")
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"🎯 Accuracy (Hit@1): {accuracy:.2f}%")

if __name__ == "__main__":
    test_model_with_real_data()