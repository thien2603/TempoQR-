#!/usr/bin/env python3
"""
Test Dataset Question Predict - Tìm câu có structure và predict qua model
"""

import sys
import os
import torch
import pickle
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dataset_question_predict():
    """Test predict with dataset question structure"""
    print("Dataset Question Predict Test")
    print("=" * 60)
    
    # Load dataset question structure
    dataset_name = 'wikidata_big'
    split = 'valid'
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(project_root, f'data/data/{dataset_name}/questions/{split}.pickle')
    
    print(f"Loading dataset from: {dataset_file}")
    
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            questions = pickle.load(f)
        print(f"Loaded {len(questions)} questions from {split} split")
    else:
        print(f"Dataset file not found: {dataset_file}")
        return
    
    # Find question with structure like the example
    target_question = None
    for question in questions:
        if question.get('type') == 'first_last' and 'first' in str(question.get('annotation', {})):
            target_question = question
            break
    
    if not target_question:
        # Fallback to any question with entities
        for question in questions:
            if question.get('entities'):
                target_question = question
                break
    
    if not target_question:
        print("No suitable question found")
        return
    
    print(f"\nFound question with structure:")
    print(f"Question: {target_question.get('question', 'N/A')}")
    print(f"Type: {target_question.get('type', 'N/A')}")
    print(f"Template: {target_question.get('template', 'N/A')}")
    print(f"Entities: {target_question.get('entities', 'N/A')}")
    print(f"Times: {target_question.get('times', 'N/A')}")
    print(f"Answers: {target_question.get('answers', 'N/A')}")
    print(f"Annotation: {target_question.get('annotation', 'N/A')}")
    print(f"Paraphrases: {target_question.get('paraphrases', 'N/A')}")
    
    # Load model
    print(f"\nLoading model...")
    
    # Setup arguments
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
        entity_file = os.path.join(project_root, "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt")
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
        
        # Test with original question
        print(f"\nTesting with original question:")
        print(f"Question: {target_question.get('question', 'N/A')}")
        
        # Prepare input using dataset structure
        question_text = target_question.get('question', '')
        
        # Get entities from question structure
        entities = list(target_question.get('entities', set()))
        times = list(target_question.get('times', set()))
        
        # Convert to IDs
        entity_ids = []
        for entity in entities:
            if entity in ent2id:
                entity_ids.append(ent2id[entity])
        
        time_ids = []
        for time in times:
            # Convert time to ID
            for (y, _, _), tid in all_dicts['ts2id'].items():
                if y == time:
                    time_ids.append(tid + len(ent2id))
                    break
        
        # Use dataset tokenization
        ent_times_text = entities + [str(t) for t in times]
        ent_times_ids = entity_ids + time_ids
        
        try:
            tokenized, entity_time_final, entity_mask = dataset.get_entity_aware_tokenization(
                question_text, ent_times_text, ent_times_ids
            )
        except Exception as e:
            print(f"Error in tokenization: {e}")
            # Fallback to simple tokenization
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
        
        times = torch.tensor([0]).long().to(device)
        start_times = torch.tensor([0]).long().to(device)
        end_times = torch.tensor([0]).long().to(device)
        tails2 = torch.tensor([heads[0]]).long().to(device)
        dummy_answers = torch.tensor([-1]).long().to(device)
        
        # Create input tuple
        input_tuple = (
            input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded,
            heads, tails, times, start_times, end_times, tails2, dummy_answers
        )
        
        # Forward pass
        with torch.no_grad():
            scores = qa_model.forward(input_tuple)
        
        # Get top-5 predictions
        import numpy as np
        scores_np = scores[0].cpu().numpy()
        top_indices = np.argsort(scores_np)[-5:][::-1]
        
        print(f"\nPredictions for original question:")
        for i, idx in enumerate(top_indices, 1):
            if idx < len(ent2id):
                # Convert entity ID to name
                ent_wd = id2ent[idx]
                entity_name = wd_id_to_text.get(ent_wd, f"Entity_{idx}")
                print(f"  {i}. {entity_name}")
            else:
                time_idx = idx - len(ent2id)
                if time_idx < len(id2ts):
                    time_value = str(id2ts[time_idx][0])
                    print(f"  {i}. {time_value}")
                else:
                    print(f"  {i}. time_{time_idx}")
        
        # Test with paraphrases
        if 'paraphrases' in target_question:
            print(f"\nTesting with paraphrases:")
            for i, paraphrase in enumerate(target_question['paraphrases'], 1):
                print(f"\nParaphrase {i}: {paraphrase}")
                
                # Simple tokenization for paraphrase
                inputs = tokenizer(paraphrase, return_tensors="pt", padding=True, truncation=True, max_length=128)
                input_ids_p = inputs["input_ids"].to(device)
                attention_mask_p = inputs["attention_mask"].to(device)
                
                batch_size = input_ids_p.shape[0]
                seq_len = input_ids_p.shape[1]
                
                num_entities = len(ent2id)
                num_times = len(id2ts)
                padding_idx = num_entities + num_times
                entity_time_ids_p = torch.full((batch_size, seq_len), padding_idx, dtype=torch.long, device=device)
                entity_mask_p = torch.ones((batch_size, seq_len), dtype=torch.float, device=device)
                
                # Use same head/tail as original
                input_tuple_p = (
                    input_ids_p, attention_mask_p, entity_time_ids_p, entity_mask_p,
                    heads, tails, times, start_times, end_times, tails2, dummy_answers
                )
                
                # Forward pass
                with torch.no_grad():
                    scores_p = qa_model.forward(input_tuple_p)
                
                # Get top-3 predictions
                scores_np_p = scores_p[0].cpu().numpy()
                top_indices_p = scores_np_p.argsort()[-3:][::-1]
                
                print(f"Predictions:")
                for j, idx in enumerate(top_indices_p, 1):
                    if idx < len(ent2id):
                        ent_wd = id2ent[idx]
                        entity_name = wd_id_to_text.get(ent_wd, f"Entity_{idx}")
                        print(f"  {j}. {entity_name}")
                    else:
                        time_idx = idx - len(ent2id)
                        if time_idx < len(id2ts):
                            time_value = str(id2ts[time_idx][0])
                            print(f"  {j}. {time_value}")
                        else:
                            print(f"  {j}. time_{time_idx}")
        
        # Compare with expected answers
        if 'answers' in target_question:
            expected_answers = target_question['answers']
            print(f"\nExpected answers: {expected_answers}")
            
            # Convert expected answers to names
            expected_names = []
            for answer in expected_answers:
                if answer in wd_id_to_text:
                    expected_names.append(wd_id_to_text[answer])
                else:
                    expected_names.append(answer)
            print(f"Expected names: {expected_names}")
        
        print(f"\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_question_predict()
