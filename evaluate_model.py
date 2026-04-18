#!/usr/bin/env python3
"""
📊 Evaluate TempoQR Model với eval() function chính thức
Sử dụng eval() từ train_qa_model.py để đánh giá độ chính xác
"""

import argparse
import torch
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# Copy eval function để tránh import toàn bộ file
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

def eval(qa_model, dataset, batch_size = 128, split='valid', k=10):
    num_workers = 0  # Disable multiprocessing to avoid memory issues
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k # not change name in fn signature since named param used in places
    k_list = [1,10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1] # last one assumed to be target
        
        # Move answers_khot sang device TRƯỚC khi dùng
        device = next(qa_model.parameters()).device
        answers_khot = answers_khot.to(device)
        
        # DEBUG: Check device của inputs
        print(f"DEBUG: Model device: {device}")
        print(f"DEBUG: answers_khot device: {answers_khot.device}")
        
        # Move tất cả inputs sang device
        a = [tensor.to(device) if hasattr(tensor, 'to') else tensor for tensor in a]
        
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        
        loss = qa_model.loss(scores, answers_khot.long())
        total_loss += loss.item()

    # compute hits at k
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0

        # compute hits at k for each question
        for i, pred in enumerate(topk_answers):
            # get ground truth answers
            answers = dataset.data[i]['answers']
            # check if any of the top k predictions are in the ground truth answers
            if any(p in answers for p in pred[:k]):
                hits_at_k += 1
            total += 1

        eval_accuracy = hits_at_k/total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log
from qa_tempoqr import QA_TempoQR
from qa_datasets import QA_Dataset_TempoQR
from utils import loadTkbcModel, getAllDicts

def main():
    print("🧪 TempoQR Model Evaluation")
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
    args.tkg_file = f'data/data/{args.dataset_name}/kg/full.txt'  # Full path
    
    # 🎯 STEP 2: Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # 🎯 STEP 3: Load TKBC model
    print("📊 Loading TKBC model...")
    tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
    tkbc_model = loadTkbcModel(tkbc_model_file)
    
    # Extract embeddings để print info
    ent_emb = tkbc_model.embeddings[0].weight.data
    rel_emb = tkbc_model.embeddings[1].weight.data  
    time_emb = tkbc_model.embeddings[2].weight.data
    
    print(f"✅ Loaded TKBC model: {len(ent_emb)} entities, {len(rel_emb)} relations, {len(time_emb)} timestamps")
    
    # 🎯 STEP 4: Load QA model
    print("🧠 Loading QA model...")
    qa_model = QA_TempoQR(tkbc_model, args)
    
    # Load trained weights TRƯỚC khi move to device
    model_path = f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt'
    try:
        checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU first
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            qa_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained weights")
        else:
            qa_model.load_state_dict(checkpoint)
            print("✅ Loaded weights only")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Move to device SAU KHI load weights
    qa_model = qa_model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    # 🎯 STEP 5: Load dataset
    print("📊 Loading dataset...")
    
    # Test on different splits
    splits_to_test = ['valid', 'test']
    results = {}
    
    for split in splits_to_test:
        print(f"\n🎯 Evaluating on {split} split...")
        print("-" * 40)
        
        try:
            # Load dataset
            dataset = QA_Dataset_TempoQR(split=split, dataset_name=args.dataset_name, args=args)
            print(f"📝 Loaded {len(dataset.data)} questions from {split} split")
            
            # Run evaluation
            set_seed(42)  # For reproducibility
            accuracy, eval_log = eval(
                qa_model=qa_model,
                dataset=dataset,
                batch_size=16,  # Much smaller batch size for 4GB GPU
                split=split,
                k=10
            )
            
            results[split] = {
                'accuracy': accuracy,
                'log': eval_log
            }
            
            print(f"\n📊 Results for {split} split:")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {split} split: {e}")
            results[split] = {'error': str(e)}
    
    # 🎯 STEP 6: Save results
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    # Create results directory if not exists
    os.makedirs('results', exist_ok=True)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/tempoqr_eval_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write(f"TempoQR Model Evaluation Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Supervision: {args.supervision}\n")
        f.write("=" * 60 + "\n\n")
        
        for split, result in results.items():
            f.write(f"Split: {split}\n")
            f.write("-" * 40 + "\n")
            
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write("Detailed Log:\n")
                for line in result['log']:
                    f.write(f"  {line}\n")
            f.write("\n")
    
    print(f"📝 Results saved to: {results_file}")
    
    # Print summary
    for split, result in results.items():
        print(f"\n📊 {split.upper()} SPLIT:")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   🎯 Accuracy: {result['accuracy']:.4f}")
    
    print(f"\n🚀 Evaluation completed!")

if __name__ == "__main__":
    main()
