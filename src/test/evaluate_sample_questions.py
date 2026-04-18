#!/usr/bin/env python3
"""
📊 TempoQR - Đánh giá nhanh trên 5 câu hỏi mẫu (chạy trên CPU, đã fix lỗi memory)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import re
from core.utils import loadTkbcModel, getAllDicts
from core.qa_tempoqr import QA_TempoQR
from core.qa_datasets import QA_Dataset_TempoQR
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikidata_big")
    parser.add_argument("--device", default="cpu")  # Chạy trên CPU để tránh lỗi GPU memory
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load model
    print("🧠 Loading TempoQR model (on CPU)...")
    tkbc_path = os.path.join(project_root, f'models/models/{args.dataset}/kg_embeddings/tcomplex.ckpt')
    model_path = os.path.join(project_root, f'models/models/{args.dataset}/qa_models/tempoqr_full_export.pt')
    
    tkbc_model = loadTkbcModel(tkbc_path)
    tkbc_model = tkbc_model.to(args.device)
    
    class DummyArgs:
        pass
    model_args = DummyArgs()
    model_args.model = 'tempoqr'
    model_args.supervision = 'soft'
    model_args.fuse = 'add'
    model_args.extra_entities = False
    model_args.frozen = 1
    model_args.lm_frozen = 1
    model_args.corrupt_hard = 0.0
    model_args.tkg_file = os.path.join(project_root, f'data/data/{args.dataset}/kg/full.txt')
    model_args.dataset_name = args.dataset
    
    qa_model = QA_TempoQR(tkbc_model, model_args)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        qa_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        qa_model.load_state_dict(checkpoint)
    qa_model = qa_model.to(args.device)
    qa_model.eval()
    
    # 2. Dataset để lấy tokenizer và mappings
    dataset = QA_Dataset_TempoQR(split='valid', dataset_name=args.dataset, args=model_args)
    tokenizer = dataset.tokenizer
    all_dicts = dataset.all_dicts
    ent2id = all_dicts['ent2id']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    wd_id_to_text = all_dicts['wd_id_to_text']
    ts2id = all_dicts['ts2id']
    num_entities = len(ent2id)
    padding_idx = dataset.padding_idx
    
    # 3. Câu hỏi mẫu
    sample_questions = [
        "Who was the president of the United States in 2020?",
        "What happened in 2015?",
        "Who won the Nobel Prize in 2021?",
        "When was Barack Obama born?",
        "What company did Steve Jobs found?"
    ]
    
    # Hàm chuẩn bị input (có giới hạn độ dài)
    def prepare_question(question_text, max_length=256):
        # Tìm entity (dựa trên tên)
        found_entities = []
        for ent in ent2id.keys():
            if ent.lower() in question_text.lower():
                found_entities.append(ent)
        # Tìm năm
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', question_text)
        ent_times_text = found_entities + years
        
        # Gọi hàm tokenization đặc biệt
        tokenized, entity_time_final, entity_mask = dataset.get_entity_aware_tokenization(
            question_text, ent_times_text, ent_times_text
        )
        
        # Cắt ngắn nếu vượt quá max_length
        if len(tokenized) > max_length:
            tokenized = tokenized[:max_length]
            entity_time_final = entity_time_final[:max_length]
            entity_mask = entity_mask[:max_length]
        
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized)]).long()
        attention_mask = torch.ones_like(input_ids)
        entity_time_ids_padded = torch.tensor([entity_time_final]).long()
        entity_mask_padded = torch.tensor([entity_mask]).float()
        
        # Lấy ID hợp lệ
        valid_ids = [i for i in entity_time_final if i != padding_idx]
        head_id = valid_ids[0] if valid_ids else 0
        tail_id = valid_ids[1] if len(valid_ids) > 1 else head_id
        
        heads = torch.tensor([head_id]).long()
        tails = torch.tensor([tail_id]).long()
        times = torch.tensor([0]).long()
        start_times = torch.tensor([0]).long()
        end_times = torch.tensor([0]).long()
        tails2 = torch.tensor([head_id]).long()
        dummy_answers = torch.tensor([-1]).long()
        
        return (input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded,
                heads, tails, times, start_times, end_times, tails2, dummy_answers)
    
    print("\n" + "="*80)
    print("🔍 Đánh giá nhanh 5 câu hỏi mẫu (chạy trên CPU)")
    print("="*80)
    
    for idx, q in enumerate(sample_questions, 1):
        print(f"\n📝 Câu {idx}: {q}")
        try:
            input_tuple = prepare_question(q)
            # Chuyển toàn bộ tensor về CPU
            input_tuple = [t.to(args.device) if isinstance(t, torch.Tensor) else t for t in input_tuple]
            with torch.no_grad():
                scores = qa_model.forward(input_tuple)
            scores_np = scores[0].cpu().numpy()
            top3_indices = scores_np.argsort()[-3:][::-1]
            preds = []
            for iid in top3_indices:
                if iid < num_entities:
                    ent_wd = id2ent[iid]
                    preds.append(wd_id_to_text.get(ent_wd, f"Entity_{iid}"))
                else:
                    tid = iid - num_entities
                    if tid < len(id2ts):
                        preds.append(str(id2ts[tid][0]))
                    else:
                        preds.append(f"Time_{tid}")
            print(f"   🎯 Top-3 dự đoán: {preds}")
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Hoàn thành.")

if __name__ == "__main__":
    main()