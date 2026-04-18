#!/usr/bin/env python3
"""
Test President Question - Test câu "Who was the president of the United States in 2020?"
"""

import sys
import os
import torch
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_president_question():
    """Test câu 'Who was the president of the United States in 2020?'"""
    print("President Question Test")
    print("=" * 60)
    
    # Câu hoi test
    question_text = "Who was the president of the United States in 2020?"
    print(f"Question: {question_text}")
    
    # Load model
    print(f"\nLoading model...")
    
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
    print(f"Using device: {device}")
    
    try:
        # Load TKBC model
        from core.utils import loadTkbcModel
        tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
        tkbc_model = loadTkbcModel(tkbc_model_file)
        print("Loaded TKBC model")
        
        # Load QA model
        from core.qa_tempoqr import QA_TempoQR
        qa_model = QA_TempoQR(tkbc_model, args)
        
        # Load trained weights
        model_path = f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt'
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            qa_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            qa_model.load_state_dict(checkpoint)
        
        qa_model = qa_model.to(device)
        qa_model.eval()
        print("Loaded QA model")
        
        # Load dictionaries
        from core.utils import getAllDicts
        all_dicts = getAllDicts(args.dataset_name)
        ent2id = all_dicts['ent2id']
        id2ent = all_dicts['id2ent']
        id2ts = all_dicts['id2ts']
        wd_id_to_text = all_dicts['wd_id_to_text']
        
        # Load entity mappings
        entity_mappings = {}
        entity_file = os.path.join(os.path.dirname(__file__), "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt")
        if os.path.exists(entity_file):
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            try:
                                id_part = parts[0].strip()
                                entity_name = parts[1].strip()
                                if id_part.startswith('Q'):
                                    entity_id = int(id_part[1:])
                                else:
                                    entity_id = int(id_part)
                                entity_mappings[entity_id] = entity_name
                            except ValueError:
                                continue
            print(f"Loaded {len(entity_mappings)} entity mappings")
        
        # Load dataset for tokenizer
        from core.qa_datasets import QA_Dataset_TempoQR
        dataset = QA_Dataset_TempoQR(split='valid', dataset_name=args.dataset_name, args=args)
        tokenizer = dataset.tokenizer
        
        print(f"\nProcessing question: {question_text}")
        
        # Tìm entities trong câu
        found_entities = []
        for ent_name in ent2id.keys():
            if ent_name.lower() in question_text.lower():
                found_entities.append(ent_name)
        
        # Tìm years trong câu
        import re
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', question_text)
        
        print(f"Found entities: {found_entities}")
        print(f"Found years: {years}")
        
        # Convert entities sang IDs
        entity_ids = []
        for entity in found_entities:
            if entity in ent2id:
                entity_ids.append(ent2id[entity])
        
        print(f"Entity IDs: {entity_ids}")
        
        # Prepare input
        # Simple tokenization
        inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        num_entities = len(ent2id)
        num_times = len(id2ts)
        padding_idx = num_entities + num_times
        entity_time_ids = torch.full((batch_size, seq_len), padding_idx, dtype=torch.long, device=device)
        entity_mask = torch.ones((batch_size, seq_len), dtype=torch.float, device=device)
        
        tokenized = tokenizer.convert_ids_to_tokens(input_ids[0])
        entity_time_final = entity_time_ids[0].cpu().numpy()
        entity_mask = entity_mask[0].cpu().numpy()
        
        # Chèn entity IDs vào positions phù trong
        for i, entity_id in enumerate(entity_ids):
            if i < len(entity_time_final):
                entity_time_final[i] = entity_id
        
        # Prepare tensors
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized)]).long().to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        entity_time_ids_padded = torch.tensor([entity_time_final]).long().to(device)
        entity_mask_padded = torch.tensor([entity_mask]).float().to(device)
        
        # Get head and tail from entities
        if entity_ids:
            heads = torch.tensor([entity_ids[0]]).long().to(device)
            tails = torch.tensor([entity_ids[1] if len(entity_ids) > 1 else entity_ids[0]]).long().to(device)
        else:
            heads = torch.tensor([0]).long().to(device)
            tails = torch.tensor([0]).long().to(device)
        
        times_tensor = torch.tensor([0]).long().to(device)
        start_times = torch.tensor([0]).long().to(device)
        end_times = torch.tensor([0]).long().to(device)
        tails2 = torch.tensor([heads[0]]).long().to(device)
        dummy_answers = torch.tensor([-1]).long().to(device)
        
        # Create input tuple
        input_tuple = (
            input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded,
            heads, tails, times_tensor, start_times, end_times, tails2, dummy_answers
        )
        
        print(f"\nInput shapes:")
        print(f"  Input IDs: {input_ids.shape}")
        print(f"  Attention Mask: {attention_mask.shape}")
        print(f"  Entity Time IDs: {entity_time_ids_padded.shape}")
        print(f"  Entity Mask: {entity_mask_padded.shape}")
        print(f"  Heads: {heads}")
        print(f"  Tails: {tails}")
        
        # Forward pass
        with torch.no_grad():
            scores = qa_model.forward(input_tuple)
        
        # Get top-10 predictions
        import numpy as np
        scores_np = scores[0].cpu().numpy()
        top_indices = np.argsort(scores_np)[-10:][::-1]
        
        print(f"\nTop-10 Predictions:")
        found_president = False
        
        for i, idx in enumerate(top_indices, 1):
            if idx < len(ent2id):
                # Convert entity ID to name
                ent_wd = id2ent[idx]
                entity_name = wd_id_to_text.get(ent_wd, f"Entity_{idx}")
                print(f"  {i}. {entity_name}")
                
                # Check if president found
                if 'president' in entity_name.lower() or 'trump' in entity_name.lower() or 'biden' in entity_name.lower():
                    found_president = True
                    print(f"    *** FOUND PRESIDENT! ***")
            else:
                time_idx = idx - len(ent2id)
                if time_idx < len(id2ts):
                    time_value = str(id2ts[time_idx][0])
                    print(f"  {i}. {time_value}")
                else:
                    print(f"  {i}. time_{time_idx}")
        
        if found_president:
            print(f"\nSUCCESS: Found president in predictions!")
        else:
            print(f"\nFAILED: No president found in top-10 predictions")
        
        print(f"\nTest completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_president_question()
