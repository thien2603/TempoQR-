#!/usr/bin/env python3
"""
Test script for entity ID to name conversion (after fixing model_loader)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.model_loader import model_manager

print("=" * 60)
print("Testing Entity ID to Name Conversion")
print("=" * 60)

test_ids = [51657, 100872, 42897, 30616]
for eid in test_ids:
    name = model_manager._id_to_entity_text(eid)
    print(f"ID {eid} -> {name}")

print("\nTest time conversion:")
# Lấy 3 time ID đầu tiên từ id2ts (nếu có)
if hasattr(model_manager, 'id2ts') and model_manager.id2ts:
    time_ids = list(model_manager.id2ts.keys())[:3]
    for tid in time_ids:
        tname = model_manager._id_to_time_text(tid)
        print(f"Time ID {tid} -> {tname}")
else:
    print("No time mappings available.")

print("=" * 60)