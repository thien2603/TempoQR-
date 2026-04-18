#!/usr/bin/env python3
"""
📊 Evaluate TempoQR Model với detailed predictions
Hiển thị từng câu hỏi và predictions chi tiết
"""

import argparse
import torch
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# Copy eval function
import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_detailed(qa_model, dataset, batch_size=16, split='valid', k=10, max_questions=50):
    """
    Evaluation với detailed predictions cho từng câu hỏi
    """
    num_workers = 0
    qa_model.eval()
    
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset._collate_fn)
    
    print(f"🎯 Evaluating {min(max_questions, len(dataset.data))} questions from {split} split...")
    print("=" * 80)
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for i_batch, a in enumerate(tqdm(data_loader, total=min(max_questions, len(dataset.data)), unit="questions")):
        if i_batch >= max_questions:
            break
            
        question_data = dataset.data[i_batch]
        answers_khot = a[-1]
        
        # Move to device
        device = next(qa_model.parameters()).device
        a = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in a]
        answers_khot = answers_khot.to(device)
        
        # Forward pass
        with torch.no_grad():
            scores = qa_model.forward(a)
            
            # Get top-k predictions
            pred = dataset.getAnswersFromScores(scores[0], k=k)
            
            # Get ground truth answers
            actual_answers = question_data.get('answers', [])
            
            # Check if correct
            correct = any(p in actual_answers for p in pred[:5])  # Top-5
            
            if correct:
                correct_predictions += 1
                
            total_predictions += 1
            
            # Map predictions to text
            pred_texts = []
            for p in pred[:5]:
                if p < len(dataset.ent2id):
                    pred_text = dataset.ent2text[p]
                else:
                    # Time prediction
                    time_idx = p - len(dataset.ent2id)
                    if time_idx < len(dataset.ts2id):
                        pred_text = f"Time_{time_idx}"
                    else:
                        pred_text = f"Unknown_{p}"
                pred_texts.append(pred_text)
            
            # Map actual answers to text
            actual_texts = []
            for ans in actual_answers:
                if ans < len(dataset.ent2id):
                    actual_text = dataset.ent2text[ans]
                else:
                    time_idx = ans - len(dataset.ent2id)
                    if time_idx < len(dataset.ts2id):
                        actual_text = f"Time_{time_idx}"
                    else:
                        actual_text = f"Unknown_{ans}"
                actual_texts.append(actual_text)
            
            # Store result
            result = {
                'question_id': i_batch,
                'question': question_data.get('question', ''),
                'actual_answers': actual_texts,
                'predicted_top5': pred_texts,
                'correct': correct,
                'entities': question_data.get('entities', []),
                'times': question_data.get('times', [])
            }
            results.append(result)
            
            # Print detailed result
            print(f"\n📝 Question {i_batch + 1}: {question_data.get('question', '')}")
            print(f"   🏷️ Entities: {question_data.get('entities', [])}")
            print(f"   ⏰ Times: {question_data.get('times', [])}")
            print(f"   ✅ Actual Answers: {actual_texts}")
            print(f"   🎯 Top-5 Predictions: {pred_texts}")
            print(f"   {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    hit_at_1 = sum(1 for r in results if r['predicted_top5'][0] in r['actual_answers']) / len(results)
    hit_at_3 = sum(1 for r in results if any(p in r['actual_answers'] for p in r['predicted_top5'][:3]) / len(results))
    hit_at_5 = sum(1 for r in results if any(p in r['actual_answers'] for p in r['predicted_top5'][:5]) / len(results))
    
    print("\n" + "=" * 80)
    print("📊 DETAILED EVALUATION RESULTS")
    print("=" * 80)
    print(f"📝 Total Questions: {len(results)}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"❌ Incorrect Predictions: {total_predictions - correct_predictions}")
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📈 Hit@1: {hit_at_1:.4f} ({hit_at_1*100:.2f}%)")
    print(f"📈 Hit@3: {hit_at_3:.4f} ({hit_at_3*100:.2f}%)")
    print(f"📈 Hit@5: {hit_at_5:.4f} ({hit_at_5*100:.2f}%)")
    
    return results, accuracy

def main():
    print("🧪 TempoQR Detailed Model Evaluation")
    print("=" * 60)
    
    # Setup arguments
    args = argparse.Namespace()
    args.dataset_name = 'wikidata_big'
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.tkg_file = f'data/data/{args.dataset_name}/kg/full.txt'
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load TKBC model
    print("📊 Loading TKBC model...")
    from src.core.utils import loadTkbcModel
    tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
    tkbc_model = loadTkbcModel(tkbc_model_file)
    
    # Extract embeddings để print info
    ent_emb = tkbc_model.embeddings[0].weight.data
    rel_emb = tkbc_model.embeddings[1].weight.data  
    time_emb = tkbc_model.embeddings[2].weight.data
    
    print(f"✅ Loaded TKBC model: {len(ent_emb)} entities, {len(rel_emb)} relations, {len(time_emb)} timestamps")
    
    # Load QA model
    print("🧠 Loading QA model...")
    from src.core.qa_tempoqr import QA_TempoQR
    qa_model = QA_TempoQR(tkbc_model, args)
    
    # Load trained weights
    model_path = f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt'
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
    print(f"✅ Model moved to device: {device}")
    
    # Load dataset
    print("📊 Loading dataset...")
    from src.core.qa_datasets import QA_Dataset_TempoQR
    
    # Test on validation split với detailed predictions
    split = 'valid'
    dataset = QA_Dataset_TempoQR(split=split, dataset_name=args.dataset_name, args=args)
    print(f"📝 Loaded {len(dataset.data)} questions from {split} split")
    
    # Run detailed evaluation
    set_seed(42)
    results, accuracy = eval_detailed(
        qa_model=qa_model,
        dataset=dataset,
        batch_size=1,  # Batch size 1 cho detailed
        split=split,
        k=10,
        max_questions=20  # Chỉ test 20 câu hỏi chi tiết
    )
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/tempoqr_detailed_eval_{timestamp}.txt'
    
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        f.write(f"TempoQR Detailed Evaluation Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: {split}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Question {i+1}:\n")
            f.write(f"  Question: {result['question']}\n")
            f.write(f"  Entities: {result['entities']}\n")
            f.write(f"  Times: {result['times']}\n")
            f.write(f"  Actual Answers: {result['actual_answers']}\n")
            f.write(f"  Top-5 Predictions: {result['predicted_top5']}\n")
            f.write(f"  Correct: {result['correct']}\n")
            f.write("-" * 60 + "\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    print(f"\n📝 Detailed results saved to: {results_file}")
    print(f"🚀 Detailed evaluation completed!")

if __name__ == "__main__":
    main()
