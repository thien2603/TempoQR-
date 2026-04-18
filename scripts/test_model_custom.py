#!/usr/bin/env python3
"""
Test script for TempoQR Model with custom questions
Based on evaluate_full_dataset.py structure but with custom test questions
No torch dependency for basic testing
"""

import argparse
import pickle
import os
import sys
from tqdm import tqdm
from datetime import datetime

def test_entity_mapping_only():
    """Test entity mapping without torch dependency"""
    print("🧪 TempoQR Entity Mapping Test")
    print("=" * 60)
    
    # Path to entity mapping file
    entity_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt"))
    
    print(f"Entity file path: {entity_file}")
    print(f"File exists: {os.path.exists(entity_file)}")
    
    if os.path.exists(entity_file):
        entity_mappings = {}
        
        print(f"\nLoading entity mappings...")
        with open(entity_file, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        try:
                            # Handle QID format (Q23008452)
                            id_part = parts[0].strip()
                            entity_name = parts[1].strip()
                            
                            # Remove Q prefix if present
                            if id_part.startswith('Q'):
                                entity_id = int(id_part[1:])  # Remove 'Q' and convert to int
                            else:
                                entity_id = int(id_part)
                            
                            entity_mappings[entity_id] = entity_name
                            count += 1
                            if count <= 10:  # Show first 10
                                print(f"  {entity_id} -> {entity_name}")
                        except ValueError as e:
                            print(f"  Skipping line: {line.strip()} - Error: {e}")
                            continue
        
        print(f"\nTotal mappings loaded: {len(entity_mappings)}")
        
        # Test specific IDs from API output
        test_ids = [50280, 87195, 42897, 51657, 100872, 30616]
        
        print(f"\nTesting specific entity IDs:")
        print("-" * 40)
        
        for entity_id in test_ids:
            if entity_id in entity_mappings:
                entity_name = entity_mappings[entity_id]
                print(f"ID {entity_id} -> {entity_name}")
            else:
                print(f"ID {entity_id} -> NOT FOUND")
        
        # Test some famous entities
        famous_ids = [76, 142, 30, 51657]  # Obama, Shakespeare, Jesus, Donald Trump
        
        print(f"\nTesting famous entities:")
        print("-" * 40)
        
        for entity_id in famous_ids:
            if entity_id in entity_mappings:
                entity_name = entity_mappings[entity_id]
                print(f"ID {entity_id} -> {entity_name}")
            else:
                print(f"ID {entity_id} -> NOT FOUND")
    
    else:
        print("Entity mapping file not found!")
        
        # Try to find the file
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        print(f"\nLooking in data directory: {data_dir}")
        if os.path.exists(data_dir):
            print("Contents:")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if "entity" in file.lower() or "wd_id" in file:
                        print(f"  {os.path.join(root, file)}")
    
    print("\n" + "=" * 60)

def test_model_with_custom_questions():
    """Test model with custom questions - full implementation"""
    print("🧪 TempoQR Model Test with Custom Questions")
    print("=" * 60)
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # 🎯 STEP 1: Setup arguments (same as evaluate_full_dataset.py)
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
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Using device: {device}")
        
        # 🎯 STEP 3: Load TKBC model
        print("📊 Loading TKBC model...")
        from utils import loadTkbcModel
        tkbc_model_file = f'models/models/{args.dataset_name}/kg_embeddings/tcomplex.ckpt'
        tkbc_model = loadTkbcModel(tkbc_model_file)
        print("✅ Loaded TKBC model")
        
        # 🎯 STEP 4: Load QA model
        print("🧠 Loading QA model...")
        from qa_tempoqr import QA_TempoQR
        qa_model = QA_TempoQR(tkbc_model, args)
        
        # Load trained weights
        model_path = f'models/models/{args.dataset_name}/qa_models/tempoqr_full_export.pt'
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            qa_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained weights")
        else:
            qa_model.load_state_dict(checkpoint)
            print("✅ Loaded weights only")
        
        qa_model = qa_model.to(device)
        qa_model.eval()
        print(f"✅ Model moved to device: {device}")
        
        # 🎯 STEP 5: Get dictionaries for mapping
        print("📊 Loading dictionaries...")
        from utils import getAllDicts
        all_dicts = getAllDicts(args.dataset_name)
        ent2id = all_dicts['ent2id']
        print("✅ Dictionaries loaded")
        
        # 🎯 STEP 6: Custom test questions
        test_questions = [
            "Who was the president of the United States in 2020?",
            "What happened in 2015?",
            "Who won the Nobel Prize in Physics in 2021?",
            "When was Barack Obama born?",
            "What company did Steve Jobs found?",
            "Who wrote Romeo and Juliet?",
            "What is the capital of France?",
            "When did World War II end?",
            "Who discovered penicillin?",
            "What is the largest planet in our solar system?",
            "Who painted the Mona Lisa?",
            "What is the currency of Japan?",
            "When was the first moon landing?",
            "Who invented the telephone?",
            "What is the highest mountain in the world?"
        ]
        
        print(f"\n🎯 Testing {len(test_questions)} custom questions...")
        print("=" * 80)
        
        # Load entity mappings
        entity_mappings = {}
        entity_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt"))
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
            print(f"✅ Loaded {len(entity_mappings)} entity mappings")
        
        # 🎯 STEP 7: Test each question
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 50)
            
            try:
                # Simple tokenization (same as model_loader.py)
                from transformers import DistilBertTokenizer
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                
                # Create dummy tensors (same as model_loader.py)
                num_entities = tkbc_model.sizes[0]
                num_times = tkbc_model.sizes[3]
                padding_idx = num_entities + num_times
                entity_time_ids = torch.full((batch_size, seq_len), padding_idx, dtype=torch.long, device=device)
                entity_mask = torch.ones((batch_size, seq_len), dtype=torch.float, device=device)
                heads = torch.zeros(batch_size, dtype=torch.long, device=device)
                tails = torch.zeros(batch_size, dtype=torch.long, device=device)
                times = torch.zeros(batch_size, dtype=torch.long, device=device)
                start_times = torch.zeros(batch_size, dtype=torch.long, device=device)
                end_times = torch.zeros(batch_size, dtype=torch.long, device=device)
                tails2 = torch.zeros(batch_size, dtype=torch.long, device=device)
                dummy_answers = torch.tensor([-1], device=device).long()
                
                # Create input tuple
                input_tuple = (
                    input_ids, attention_mask, entity_time_ids, entity_mask,
                    heads, tails, times, start_times, end_times,
                    tails2, dummy_answers
                )
                
                # Forward pass
                with torch.no_grad():
                    scores = qa_model.forward(input_tuple)
                
                # Get top-5 predictions
                import numpy as np
                scores_np = scores[0].cpu().numpy()
                top_indices = np.argsort(scores_np)[-5:][::-1]
                
                print(f"🎯 Results for Question {i}:")
                for j, idx in enumerate(top_indices, 1):
                    if idx < num_entities:
                        # Convert entity ID to name
                        entity_name = entity_mappings.get(int(idx), f"entity_{int(idx)}")
                        print(f"  {j}. {entity_name}")
                    else:
                        time_idx = idx - num_entities
                        print(f"  {j}. time_{time_idx}")
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("🎯 Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to choose test mode"""
    print("🧪 TempoQR Test Suite")
    print("=" * 60)
    print("Choose test mode:")
    print("1. Entity mapping test only (no torch)")
    print("2. Full model test with custom questions")
    print("3. Both tests")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        test_entity_mapping_only()
    elif choice == "2":
        test_model_with_custom_questions()
    elif choice == "3":
        test_entity_mapping_only()
        print("\n" + "=" * 60)
        test_model_with_custom_questions()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
