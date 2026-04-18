#!/usr/bin/env python3
"""
Test Custom Structure Predict - T câu ví có structure và test model predict
"""

import sys
import os
import torch
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_custom_structure_predict():
    """Test predict with custom structure questions"""
    print("Custom Structure Predict Test")
    print("=" * 60)
    
    # T câu ví có structure tng t (first_last type) - 3 ví du có day du chi so giong dataset
    custom_questions = [
        {
            'question': 'What was the award that was awarded to Barack Obama for the first time ever',
            'answers': {'Q76'},  # Barack Obama's QID
            'answer_type': 'entity',
            'template': 'What was the award that was awarded to {head} for the first time ever',
            'entities': {'Q76'},  # Barack Obama's QID
            'times': set(),
            'relations': {'P166'},
            'type': 'first_last',
            'annotation': {'head': 'Q76', 'adj': 'first'},
            'uniq_id': 10001,  # Custom unique ID
            'expected_answer': 'Nobel Peace Prize',
            'paraphrases': [
                'What was the award that was awarded to Barack Obama for the first time ever'
            ]
        },
        {
            'question': 'What was the prize that was awarded to Albert Einstein for the first time ever',
            'answers': {'Q937'},  # Albert Einstein's QID
            'answer_type': 'entity',
            'template': 'What was the prize that was awarded to {head} for the first time ever',
            'entities': {'Q937'},  # Albert Einstein's QID
            'times': set(),
            'relations': {'P166'},
            'type': 'first_last',
            'annotation': {'head': 'Q937', 'adj': 'first'},
            'uniq_id': 10002,  # Custom unique ID
            'expected_answer': 'Nobel Prize in Physics',
            'paraphrases': [
                'What was the prize that was awarded to Albert Einstein for the first time ever'
            ]
        },
        {
            'question': 'What was the award that was awarded to Marie Curie for the first time ever',
            'answers': {'Q7186'},  # Marie Curie's QID
            'answer_type': 'entity',
            'template': 'What was the award that was awarded to {head} for the first time ever',
            'entities': {'Q7186'},  # Marie Curie's QID
            'times': set(),
            'relations': {'P166'},
            'type': 'first_last',
            'annotation': {'head': 'Q7186', 'adj': 'first'},
            'uniq_id': 10003,  # Custom unique ID
            'expected_answer': 'Nobel Prize in Chemistry',
            'paraphrases': [
                'What was the award that was awarded to Marie Curie for the first time ever'
            ]
        }
    ]
    
    print(f"Testing {len(custom_questions)} custom structure questions")
    
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
        
        # Test each custom question
        for i, question_data in enumerate(custom_questions, 1):
            print(f"\n{'='*80}")
            print(f"Test {i}: Custom Structure Question")
            print('=' * 80)
            
            question_text = question_data['question']
            print(f"Question: {question_text}")
            print(f"Type: {question_data['type']}")
            print(f"Template: {question_data['template']}")
            print(f"Entities: {question_data['entities']}")
            print(f"Answers: {question_data['answers']}")
            print(f"Answer Type: {question_data['answer_type']}")
            print(f"Relations: {question_data['relations']}")
            print(f"Annotation: {question_data['annotation']}")
            print(f"Unique ID: {question_data['uniq_id']}")
            print(f"Expected Answer: {question_data['expected_answer']}")
            
            # Get entities from question structure (QIDs)
            entity_qids = list(question_data['entities'])
            times = list(question_data['times'])
            
            # Convert QIDs to entity names and IDs
            entity_names = []
            entity_ids = []
            for qid in entity_qids:
                if qid in wd_id_to_text:
                    entity_name = wd_id_to_text[qid]
                    entity_names.append(entity_name)
                    # Find entity ID from name
                    for name, eid in ent2id.items():
                        if name == entity_name or qid in name:
                            entity_ids.append(eid)
                            break
                    else:
                        print(f"  Warning: Entity '{qid}' ({entity_name}) not found in ent2id")
                else:
                    print(f"  Warning: QID '{qid}' not found in wd_id_to_text")
                    entity_names.append(qid)
            
            time_ids = []
            for time in times:
                # Convert time to ID
                for (y, _, _), tid in all_dicts['ts2id'].items():
                    if y == time:
                        time_ids.append(tid + len(ent2id))
                        break
            
            print(f"Entity QIDs: {entity_qids}")
            print(f"Entity Names: {entity_names}")
            print(f"Entity IDs: {entity_ids}")
            print(f"Time IDs: {time_ids}")
            
            # Prepare input
            try:
                # Use simple tokenization instead of get_entity_aware_tokenization to avoid textToEntTimeId error
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
                
                # Forward pass
                with torch.no_grad():
                    scores = qa_model.forward(input_tuple)
                
                # Get top-5 predictions
                import numpy as np
                scores_np = scores[0].cpu().numpy()
                top_indices = np.argsort(scores_np)[-5:][::-1]
                
                print(f"\nPredictions:")
                found_expected = False
                for j, idx in enumerate(top_indices, 1):
                    if idx < len(ent2id):
                        # Convert entity ID to name
                        ent_wd = id2ent[idx]
                        entity_name = wd_id_to_text.get(ent_wd, f"Entity_{idx}")
                        print(f"  {j}. {entity_name}")
                        
                        # Check if expected answer found
                        if question_data['expected_answer'].lower() in entity_name.lower():
                            found_expected = True
                            print(f"    *** FOUND EXPECTED ANSWER! ***")
                    else:
                        time_idx = idx - len(ent2id)
                        if time_idx < len(id2ts):
                            time_value = str(id2ts[time_idx][0])
                            print(f"  {j}. {time_value}")
                        else:
                            print(f"  {j}. time_{time_idx}")
                
                if found_expected:
                    print(f"\nSUCCESS: Expected answer '{question_data['expected_answer']}' found in predictions!")
                else:
                    print(f"\nFAILED: Expected answer '{question_data['expected_answer']}' not found in top-5 predictions")
                
                # Test with paraphrases
                if 'paraphrases' in question_data:
                    print(f"\nTesting with paraphrases:")
                    for j, paraphrase in enumerate(question_data['paraphrases'], 1):
                        print(f"\nParaphrase {j}: {paraphrase}")
                        
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
                            heads, tails, times_tensor, start_times, end_times, tails2, dummy_answers
                        )
                        
                        # Forward pass
                        with torch.no_grad():
                            scores_p = qa_model.forward(input_tuple_p)
                        
                        # Get top-3 predictions
                        scores_np_p = scores_p[0].cpu().numpy()
                        top_indices_p = scores_np_p.argsort()[-3:][::-1]
                        
                        print(f"Predictions:")
                        paraphrase_found = False
                        for k, idx in enumerate(top_indices_p, 1):
                            if idx < len(ent2id):
                                ent_wd = id2ent[idx]
                                entity_name = wd_id_to_text.get(ent_wd, f"Entity_{idx}")
                                print(f"  {k}. {entity_name}")
                                
                                if question_data['expected_answer'].lower() in entity_name.lower():
                                    paraphrase_found = True
                                    print(f"    *** FOUND EXPECTED ANSWER! ***")
                            else:
                                time_idx = idx - len(ent2id)
                                if time_idx < len(id2ts):
                                    time_value = str(id2ts[time_idx][0])
                                    print(f"  {k}. {time_value}")
                                else:
                                    print(f"  {k}. time_{time_idx}")
                        
                        if paraphrase_found:
                            print(f"SUCCESS: Expected answer found in paraphrase!")
                        else:
                            print(f"FAILED: Expected answer not found in paraphrase")
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print("Custom Structure Test Summary")
        print('=' * 80)
        print("Test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_structure_predict()
