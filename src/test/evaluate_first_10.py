#!/usr/bin/env python3
"""
📊 TempoQR - Đánh giá nhanh 10 câu đầu + Debug một câu hỏi mẫu
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import re
from tqdm import tqdm
from datetime import datetime
from core.utils import loadTkbcModel, getAllDicts
from core.qa_tempoqr import QA_TempoQR
from core.qa_datasets import QA_Dataset_TempoQR
import argparse

def inspect_question_structure(question_text, dataset, all_dicts, model=None, device='cpu'):
    """Inspect the processing steps for a natural language question"""
    print("\n" + "="*80)
    print(f"🔍 PHÂN TÍCH CÂU HỎI: {question_text}")
    print("="*80)
    
    # Step 1: Extract entities (simple substring matching)
    ent2id = all_dicts['ent2id']
    found_entities = []
    for ent in ent2id.keys():
        if ent.lower() in question_text.lower():
            found_entities.append(ent)
    print(f"\n📌 Bước 1: Tìm entity trong câu hỏi (dựa trên từ điển)")
    print(f"   Entity tìm thấy: {found_entities}")
    
    # Step 2: Extract years
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', question_text)
    print(f"\n📌 Bước 2: Tìm năm (time)")
    print(f"   Năm tìm thấy: {years}")
    
    # Step 3: Combine and get IDs
    ent_times_text = found_entities + years
    ent_times_ids = []
    ts2id = all_dicts['ts2id']
    num_entities = len(ent2id)
    for text in ent_times_text:
        if text in ent2id:
            ent_times_ids.append(ent2id[text])
        elif text.isdigit():
            year = int(text)
            found = False
            for (y,_,_), tid in ts2id.items():
                if y == year:
                    ent_times_ids.append(tid + num_entities)
                    found = True
                    break
            if not found:
                ent_times_ids.append(dataset.padding_idx)
        else:
            ent_times_ids.append(dataset.padding_idx)
    print(f"\n📌 Bước 3: Chuyển entity/time sang ID trong KG")
    print(f"   ID tương ứng: {ent_times_ids}")
    
    # Step 4: Entity-aware tokenization
    print(f"\n📌 Bước 4: Entity‑aware tokenization (thay entity/time bằng [MASK])")
    tokenized, entity_time_final, entity_mask = dataset.get_entity_aware_tokenization(
        question_text, ent_times_text, ent_times_text   # truyền text cho cả hai
    )
    print(f"   Tokenized câu: {tokenized}")
    print(f"   entity_time_final (ID tại mỗi token): {entity_time_final}")
    print(f"   entity_mask (1=non-entity, 0=entity): {entity_mask}")
    
    # Step 5: Determine head, tail (first valid entity as head, second as tail)
    valid_ids = [i for i in entity_time_final if i != dataset.padding_idx]
    head_id = valid_ids[0] if valid_ids else 0
    tail_id = valid_ids[1] if len(valid_ids) > 1 else head_id
    print(f"\n📌 Bước 5: Xác định head và tail cho model")
    print(f"   head ID: {head_id} (tên: {dataset.getEntityIdToText(head_id) if head_id < num_entities else 'time'})")
    print(f"   tail ID: {tail_id}")
    
    # Step 6: Optionally run model to see predictions
    if model is not None:
        # Create input tuple
        tokenizer = dataset.tokenizer
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized)]).long().to(device)
        attention_mask = torch.ones_like(input_ids)
        entity_time_ids_padded = torch.tensor([entity_time_final]).long().to(device)
        entity_mask_padded = torch.tensor([entity_mask]).float().to(device)
        heads = torch.tensor([head_id]).long().to(device)
        tails = torch.tensor([tail_id]).long().to(device)
        times = torch.tensor([0]).long().to(device)
        start_times = torch.tensor([0]).long().to(device)
        end_times = torch.tensor([0]).long().to(device)
        tails2 = torch.tensor([head_id]).long().to(device)
        dummy_answers = torch.tensor([-1]).long().to(device)
        input_tuple = (input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded,
                       heads, tails, times, start_times, end_times, tails2, dummy_answers)
        
        with torch.no_grad():
            scores = model.forward(input_tuple)
        scores_np = scores[0].cpu().numpy()
        top3_indices = scores_np.argsort()[-3:][::-1]
        wd_id_to_text = all_dicts['wd_id_to_text']
        id2ent = all_dicts['id2ent']
        id2ts = all_dicts['id2ts']
        preds = []
        for iid in top3_indices:
            if iid < num_entities:
                ent_wd = id2ent[iid]
                preds.append(wd_id_to_text.get(ent_wd, f"Entity_{iid}"))
            else:
                tid = iid - num_entities
                preds.append(str(id2ts[tid][0]) if tid < len(id2ts) else f"Time_{tid}")
        print(f"\n📌 Bước 6: Dự đoán từ model (top-3)")
        print(f"   {preds}")
    else:
        print(f"\n📌 Bước 6: Không chạy model (--no_run_model)")
    
    print("\n✅ Kết thúc phân tích.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", default="wikidata_big")
    parser.add_argument("--num_samples", default=10, type=int, help="Số câu hỏi đầu tiên để test")
    parser.add_argument("--debug", action="store_true", help="Chạy debug với câu hỏi mẫu")
    parser.add_argument("--debug_question", type=str, default="who is the president of the United States in 2020?")
    parser.add_argument("--no_run_model", action="store_true", help="Chỉ in quy trình, không chạy model")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Setup args cho model
    model_args = argparse.Namespace()
    model_args.dataset_name = args.dataset
    model_args.model = 'tempoqr'
    model_args.supervision = 'soft'
    model_args.fuse = 'add'
    model_args.extra_entities = False
    model_args.frozen = 1
    model_args.lm_frozen = 1
    model_args.corrupt_hard = 0.0
    model_args.tkg_file = os.path.join(project_root, f'data/data/{args.dataset}/kg/full.txt')
    
    device = args.device
    print(f"🖥️ Device: {device}")
    
    # Load TKBC
    print("📊 Loading TKBC model...")
    tkbc_path = os.path.join(project_root, f'models/models/{args.dataset}/kg_embeddings/tcomplex.ckpt')
    tkbc_model = loadTkbcModel(tkbc_path)
    tkbc_model = tkbc_model.to(device)
    
    # Load QA model
    print("🧠 Loading QA model...")
    qa_model = QA_TempoQR(tkbc_model, model_args)
    model_path = os.path.join(project_root, f'models/models/{args.dataset}/qa_models/tempoqr_full_export.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        qa_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        qa_model.load_state_dict(checkpoint)
    qa_model = qa_model.to(device)
    qa_model.eval()
    
    # Load dataset
    print(f"📊 Loading {args.split} dataset...")
    dataset = QA_Dataset_TempoQR(split=args.split, dataset_name=args.dataset, args=model_args)
    print(f"📝 Total questions in split: {len(dataset.data)}")
    
    all_dicts = getAllDicts(args.dataset)
    
    # Nếu chế độ debug, chỉ chạy phân tích câu hỏi mẫu
    if args.debug:
        inspect_question_structure(
            args.debug_question, dataset, all_dicts,
            model=qa_model if not args.no_run_model else None,
            device=device
        )
        return
    
    # Nếu không debug, chạy đánh giá trên num_samples câu đầu
    num_samples = min(args.num_samples, len(dataset.data))
    sample_indices = list(range(num_samples))
    
    from torch.utils.data import Subset, DataLoader
    subset = Subset(dataset, sample_indices)
    data_loader = DataLoader(subset, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=dataset._collate_fn)
    
    print(f"\n🎯 Đánh giá {num_samples} câu đầu tiên...")
    print("=" * 80)
    
    ent2id = all_dicts['ent2id']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    wd_id_to_text = all_dicts['wd_id_to_text']
    
    correct_top1 = 0
    correct_top5 = 0
    results = []
    
    for idx, batch in enumerate(tqdm(data_loader, total=num_samples)):
        answers_khot = batch[-1]
        batch = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in batch]
        answers_khot = answers_khot.to(device)
        
        with torch.no_grad():
            scores = qa_model.forward(batch)
        
        pred_texts = dataset.getAnswersFromScores(scores[0], k=5)
        question_obj = dataset.data[sample_indices[idx]]
        question_text = question_obj.get('question', '')
        actual_answers_raw = question_obj.get('answers', [])
        actual_texts = []
        for ans in actual_answers_raw:
            if isinstance(ans, str):
                actual_texts.append(ans)
            elif isinstance(ans, int):
                if ans < len(ent2id):
                    ent_wd = id2ent[ans]
                    actual_texts.append(wd_id_to_text.get(ent_wd, f"Entity_{ans}"))
                else:
                    time_idx = ans - len(ent2id)
                    actual_texts.append(str(id2ts[time_idx][0]) if time_idx < len(id2ts) else f"Time_{time_idx}")
            else:
                actual_texts.append(str(ans))
        
        correct1 = (pred_texts[0] in actual_texts) if pred_texts else False
        correct5 = any(p in actual_texts for p in pred_texts)
        
        if correct1:
            correct_top1 += 1
        if correct5:
            correct_top5 += 1
        
        results.append({
            'index': idx,
            'question': question_text,
            'actual': actual_texts,
            'predicted_top5': pred_texts,
            'correct_top1': correct1,
            'correct_top5': correct5
        })
        
        print(f"\n📝 Câu {idx+1}: {question_text}")
        print(f"   Ground truth: {actual_texts}")
        print(f"   Top-5 predictions: {pred_texts}")
        print(f"   Top-1 correct: {'✅' if correct1 else '❌'}, Top-5 correct: {'✅' if correct5 else '❌'}")
    
    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ TRÊN {num_samples} CÂU ĐẦU")
    print("=" * 80)
    print(f"Top-1 Accuracy: {correct_top1}/{num_samples} = {correct_top1/num_samples:.2%}")
    print(f"Top-5 Accuracy: {correct_top5}/{num_samples} = {correct_top5/num_samples:.2%}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"tempoqr_{args.split}_first{num_samples}_{timestamp}.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"TempoQR Evaluation on first {num_samples} questions of {args.split} split\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Top-1 Accuracy: {correct_top1}/{num_samples} ({correct_top1/num_samples:.2%})\n")
        f.write(f"Top-5 Accuracy: {correct_top5}/{num_samples} ({correct_top5/num_samples:.2%})\n\n")
        for res in results:
            f.write(f"Q{res['index']+1}: {res['question']}\n")
            f.write(f"  Actual: {res['actual']}\n")
            f.write(f"  Top-5: {res['predicted_top5']}\n")
            f.write(f"  Top-1 correct: {res['correct_top1']}, Top-5 correct: {res['correct_top5']}\n")
            f.write("-" * 60 + "\n")
    print(f"\n📝 Kết quả lưu tại: {results_file}")

if __name__ == "__main__":
    main()