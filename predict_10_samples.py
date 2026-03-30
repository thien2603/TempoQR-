#!/usr/bin/env python3
"""
📊 TempoQR Model Prediction for 10 Test Samples
Lấy 10 mẫu từ test set và predict answers
"""

import argparse
import torch
import pickle
import os
from tqdm import tqdm
from datetime import datetime

def main():
    print("🧪 TempoQR Model Prediction - 50 Test Samples")
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
    args.tkg_file = f'data/data/{args.dataset_name}/kg/full.txt'
    
    # 🎯 STEP 2: Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # 🎯 STEP 3: Load TKBC model
    print("📊 Loading TKBC model...")
    from utils import loadTkbcModel
    tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
    tkbc_model = loadTkbcModel(tkbc_model_file)
    print(f"✅ Loaded TKBC model")
    
    # 🎯 STEP 4: Load QA model
    print("🧠 Loading QA model...")
    from qa_tempoqr import QA_TempoQR
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
    qa_model.eval()  # Set to eval mode for BatchNorm
    print(f"✅ Model moved to device: {device}")
    print(f"✅ Model set to eval mode")
    
    # 🎯 STEP 5: Load dataset
    print("📊 Loading test dataset...")
    from qa_datasets import QA_Dataset_TempoQR
    dataset = QA_Dataset_TempoQR(split='test', dataset_name=args.dataset_name, args=args)
    print(f"📝 Loaded {len(dataset.data)} questions from test split")
    
    # 🎯 STEP 6: Get 50 random samples
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset.data)), min(10, len(dataset.data)))
    
    print(f"\n🎯 Predicting answers for {len(sample_indices)} test samples...")
    print("=" * 80)
    
    # 🎯 STEP 7: Load tokenizer
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 🎯 STEP 8: Get dictionaries for mapping
    from utils import getAllDicts
    all_dicts = getAllDicts(args.dataset_name)
    ent2id = all_dicts['ent2id']
    
    predictions = []
    
    # 🎯 STEP 9: Process each sample
    for i, idx in enumerate(sample_indices):
        question_data = dataset.data[idx]
        question = question_data['question']
        actual_answers = question_data.get('answers', [])
        entities = question_data.get('entities', [])
        times = question_data.get('times', [])
        
        print(f"\n📝 Sample {i+1}:")
        print(f"   Question: {question}")
        print(f"   Entities: {entities}")
        print(f"   Times: {times}")
        print(f"   Actual Answers: {actual_answers}")
        
        # 🧠 STEP 9.1: Dùng dataset.__getitem__() và _collate_fn()
        from torch.utils.data import DataLoader
        # Get dataset item (proper format)
        dataset_item = dataset[idx]  # ← Dùng __getitem__
        
        # Create batch với proper format
        data_loader = DataLoader([dataset_item], batch_size=1, shuffle=False,
                               collate_fn=dataset._collate_fn)
        
        # Get collated batch
        for a in data_loader:
            # Move to device
            a = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in a]
            break  # Chỉ lấy batch đầu tiên
        
        # 🧠 STEP 9.2: Forward pass với proper batch data
        with torch.no_grad():
            scores = qa_model.forward(a)
        
        # 🧠 STEP 9.5: Get top-5 predictions
        top_scores, top_indices = torch.topk(scores[0], k=5)
        
        # Map actual answers sang tên
        actual_answer_names = []
        for ans in actual_answers:
            if isinstance(ans, int):
                if ans < len(ent2id):
                    ans_name = dataset.getEntityIdToText(ans)
                else:
                    ans_name = str(ans)  # Time values
            else:
                # String entity ID like 'Q333128'
                if ans.startswith('Q') and ans in ent2id:
                    ans_name = dataset.getEntityIdToText(ent2id[ans])
                else:
                    ans_name = str(ans)
            actual_answer_names.append(ans_name)
        
        # 🧠 STEP 9.6: Map predictions to text
        predicted_answers = []
        for idx in top_indices:
            idx_int = idx.item()  # Convert tensor to int
            if idx_int < len(ent2id):
                pred_text = dataset.getEntityIdToText(idx_int)
            else:
                time_idx = idx_int - len(ent2id)
                if time_idx < len(all_dicts['ts2id']):
                    pred_text = f"Time_{time_idx}"
                else:
                    pred_text = f"Unknown_{idx_int}"
            predicted_answers.append(pred_text)
        
        # 🧠 STEP 9.7: Check if correct với proper type conversion
        correct = False
        for pred in predicted_answers[:3]:
            # Convert prediction để match actual answer type
            if pred.startswith('Time_'):
                pred_time = int(pred.replace('Time_', ''))
                if pred_time in actual_answers:
                    correct = True
                    break
            else:
                # Entity prediction - check if matches with mapped names
                if pred in actual_answer_names:
                    correct = True
                    break
        
        # Store result
        result = {
            'sample_id': i + 1,
            'question': question,
            'entities': entities,
            'times': times,
            'actual_answers': actual_answers,
            'predicted_top5': predicted_answers,
            'correct': correct
        }
        predictions.append(result)
        
        print(f"   🎯 Top-5 Predictions: {predicted_answers}")
        print(f"   {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    
    # 🎯 STEP 10: Save predictions
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    correct_count = sum(1 for p in predictions if p['correct'])
    accuracy = correct_count / len(predictions) if predictions else 0
    
    print(f"📝 Total samples: {len(predictions)}")
    print(f"✅ Correct predictions: {correct_count}")
    print(f"❌ Incorrect predictions: {len(predictions) - correct_count}")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = f'results/tempoqr_predictions_{timestamp}.txt'
    
    os.makedirs('results', exist_ok=True)
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write("TempoQR Model Predictions - 10 Test Samples\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write("=" * 80 + "\n\n")
        
        for pred in predictions:
            f.write(f"Sample {pred['sample_id']}:\n")
            f.write(f"  Question: {pred['question']}\n")
            f.write(f"  Entities: {pred['entities']}\n")
            f.write(f"  Times: {pred['times']}\n")
            f.write(f"  Actual Answers: {pred['actual_answers']}\n")
            f.write(f"  Top-5 Predictions: {pred['predicted_top5']}\n")
            f.write(f"  Correct: {pred['correct']}\n")
            f.write("-" * 60 + "\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"Summary:\n")
        f.write(f"  Total Samples: {len(predictions)}\n")
        f.write(f"  Correct: {correct_count}\n")
        f.write(f"  Incorrect: {len(predictions) - correct_count}\n")
        f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    print(f"\n📝 Predictions saved to: {predictions_file}")
    print(f"🚀 Prediction completed!")

if __name__ == "__main__":
    main()
