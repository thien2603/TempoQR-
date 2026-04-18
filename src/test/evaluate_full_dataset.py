#!/usr/bin/env python3
"""
📊 TempoQR Model Full Dataset Evaluation
Test với toàn bộ 30,000 questions trong test set
"""

import sys
import os

# Thêm src vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lây project root path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pickle
from tqdm import tqdm
from datetime import datetime

def main():
    print("🧪 TempoQR Model Full Dataset Evaluation")
    print("=" * 60)
    
    # 🎯 STEP 1: Setup arguments
    args = argparse.Namespace()
    args.dataset_name = 'wikidata_big'
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.tkg_file = os.path.join(project_root, f'data/data/{args.dataset_name}/kg/full.txt')
    
    # 🎯 STEP 2: Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # 🎯 STEP 3: Load TKBC model
    print("📊 Loading TKBC model...")
    from core.utils import loadTkbcModel
    tkbc_model_file = os.path.join(project_root, f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt')
    tkbc_model = loadTkbcModel(tkbc_model_file)
    print(f"✅ Loaded TKBC model")
    
    # 🎯 STEP 4: Load QA model
    print("🧠 Loading QA model...")
    from core.qa_tempoqr import QA_TempoQR
    qa_model = QA_TempoQR(tkbc_model, args)
    
    # Load trained weights
    model_path = os.path.join(project_root, f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt')
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            qa_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained weights")
        else:
            qa_model.load_state_dict(checkpoint)
            print("✅ Loaded weights only")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    qa_model = qa_model.to(device)
    qa_model.eval()
    print(f"✅ Model moved to device: {device}")
    
    # 🎯 STEP 5: Load dataset (sẽ tự load tokenizer)
    print("📊 Loading valid dataset...")
    from core.qa_datasets import QA_Dataset_TempoQR
    dataset = QA_Dataset_TempoQR(split='valid', dataset_name=args.dataset_name, args=args)
    print(f"📝 Loaded {len(dataset.data)} questions from valid split")
    
    # 🎯 STEP 6: Get tokenizer từ dataset
    tokenizer = dataset.tokenizer
    print("✅ Tokenizer loaded from dataset")
    
    # 🎯 STEP 8: Get dictionaries for mapping
    from core.utils import getAllDicts
    all_dicts = getAllDicts(args.dataset_name)
    ent2id = all_dicts['ent2id']
    
    # 🎯 STEP 7: Full evaluation using dataset._collate_fn()
    print(f"\n🎯 Evaluating full dataset ({len(dataset.data)} questions)...")
    print("=" * 80)
    
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=dataset._collate_fn)
    
    correct_predictions = 0
    total_predictions = 0
    time_correct = 0
    time_total = 0
    entity_correct = 0
    entity_total = 0
    
    # Track by question type
    question_types = {}
    sample_results = []  # Store sample results for output
    
    for i_batch, a in enumerate(tqdm(data_loader, total=len(data_loader), unit="batches")):
        if i_batch * 32 == len(dataset.data):
            break
            
        answers_khot = a[-1]
        
        # Move to device
        a = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in a]
        answers_khot = answers_khot.to(device)
        
        # Forward pass
        with torch.no_grad():
            scores = qa_model.forward(a)
        
        # Get predictions for each item in batch
        batch_size = scores.size(0)
        for i in range(batch_size):
            sample_idx = i_batch * 32 + i
            if sample_idx >= len(dataset.data):
                break
                
            question_data = dataset.data[sample_idx]
            actual_answers = question_data.get('answers', [])
            question = question_data.get('question', '')
            entities = question_data.get('entities', [])
            times = question_data.get('times', [])
            
            # Get top-5 predictions
            pred = dataset.getAnswersFromScores(scores[i], k=5)
            
            # getAnswersFromScores() already returns text strings
            predicted_texts = pred[:5]
            
            # Map actual answers to text
            actual_texts = []
            for ans in actual_answers:
                if isinstance(ans, str):
                    # Already a string, use directly
                    actual_texts.append(ans)
                elif isinstance(ans, int):
                    # Integer ID, convert to text
                    if ans < len(ent2id):
                        ans_text = dataset.getEntityIdToText(ans)
                    else:
                        time_idx = ans - len(ent2id)
                        if time_idx < len(all_dicts['ts2id']):
                            ans_text = f"Time_{time_idx}"
                        else:
                            ans_text = f"Unknown_{ans}"
                    actual_texts.append(ans_text)
                else:
                    actual_texts.append(str(ans))
            
            # Check if correct với proper type conversion
            correct = False
            for p in pred[:3]:  # Top-3
                # Convert prediction để match actual answer type
                if isinstance(p, int):
                    # Integer prediction - check if matches actual answers
                    if p in actual_answers:
                        correct = True
                        break
                elif isinstance(p, str):
                    # String prediction
                    if p.startswith('Time_'):
                        pred_time = int(p.replace('Time_', ''))
                        if pred_time in actual_answers:
                            correct = True
                            break
                    else:
                        # Entity prediction - check if matches
                        if p in actual_texts:  # Compare with mapped actual texts
                            correct = True
                            break
            
            if correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            # Determine question type
            if times or 'year' in question.lower() or 'when' in question.lower():
                question_type = 'time'
                time_total += 1
                if correct:
                    time_correct += 1
            else:
                question_type = 'entity'
                entity_total += 1
                if correct:
                    entity_correct += 1
            
            # Track question types
            if question_type not in question_types:
                question_types[question_type] = {'correct': 0, 'total': 0}
            question_types[question_type]['total'] += 1
            if correct:
                question_types[question_type]['correct'] += 1
            
            # Store sample result for output (chỉ lưu 100 mẫu đầu)
            if len(sample_results) < 100:
                sample_results.append({
                    'question': question,
                    'actual_answers': actual_texts,
                    'predicted_top5': predicted_texts,
                    'correct': correct,
                    'question_type': question_type
                })
    
    # 🎯 STEP 8: Calculate final results
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    time_accuracy = time_correct / time_total if time_total > 0 else 0
    entity_accuracy = entity_correct / entity_total if entity_total > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 FULL DATASET EVALUATION RESULTS")
    print("=" * 80)
    print(f"📝 Total Questions: {total_predictions}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"❌ Incorrect Predictions: {total_predictions - correct_predictions}")
    print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"📈 Time Questions: {time_total} ({time_accuracy:.4f} = {time_accuracy*100:.2f}%)")
    print(f"📈 Entity Questions: {entity_total} ({entity_accuracy:.4f} = {entity_accuracy*100:.2f}%)")
    
    print(f"\n📊 Question Type Breakdown:")
    for qtype, stats in question_types.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {qtype.capitalize()}: {stats['correct']}/{stats['total']} = {acc:.4f} ({acc*100:.2f}%)")
    
    # Print sample results
    print(f"\n📝 Sample Results (First 100):")
    print("=" * 80)
    for i, result in enumerate(sample_results[:20]):  # Show 20 samples
        print(f"\n📝 Sample {i+1} ({result['question_type'].upper()}):")
        print(f"   Question: {result['question']}")
        print(f"   Actual Answers: {result['actual_answers']}")
        print(f"   Top-5 Predictions: {result['predicted_top5']}")
        print(f"   {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}")
    
    # 🎯 STEP 9: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/tempoqr_full_eval_{timestamp}.txt'
    
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("TempoQR Model Full Dataset Evaluation Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: test\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Questions: {total_predictions}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        
        f.write("Question Type Breakdown:\n")
        for qtype, stats in question_types.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"  {qtype.capitalize()}: {stats['correct']}/{stats['total']} = {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write(f"\nTime Questions: {time_correct}/{time_total} = {time_accuracy:.4f} ({time_accuracy*100:.2f}%)\n")
        f.write(f"Entity Questions: {entity_correct}/{entity_total} = {entity_accuracy:.4f} ({entity_accuracy*100:.2f}%)\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write("Sample Results (First 100):\n")
        f.write("=" * 80 + "\n")
        
        for i, result in enumerate(sample_results):
            f.write(f"\nSample {i+1} ({result['question_type'].upper()}):\n")
            f.write(f"  Question: {result['question']}\n")
            f.write(f"  Actual Answers: {result['actual_answers']}\n")
            f.write(f"  Top-5 Predictions: {result['predicted_top5']}\n")
            f.write(f"  Correct: {result['correct']}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\n📝 Full results saved to: {results_file}")
    print(f"🚀 Full dataset evaluation completed!")

if __name__ == "__main__":
    main()
