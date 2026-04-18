#!/usr/bin/env python3
"""
Simple test script for TempoQR Model with 10 custom questions
No torch dependency - just test entity mapping
"""

import os

def test_entity_mapping():
    """Test entity mapping file directly"""
    
    # Path to entity mapping file
    entity_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data", "wikidata_big", "kg", "wd_id2entity_text.txt"))
    
    print("=" * 60)
    print("Testing Entity Mapping File")
    print("=" * 60)
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

if __name__ == "__main__":
    test_entity_mapping()
