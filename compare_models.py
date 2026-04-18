#!/usr/bin/env python3
"""
Compare model configuration between API and evaluate_full_dataset.py
"""

def compare_model_configs():
    print("=" * 60)
    print("🔍 Model Configuration Comparison")
    print("=" * 60)
    
    # API config from src/core/config.py
    api_config = {
        'model': 'tempoqr',
        'dataset_name': 'wikidata_big',
        'supervision': 'soft',
        'fuse': 'add',
        'extra_entities': False,
        'frozen': 1,
        'lm_frozen': 1,
        'corrupt_hard': 0.0,
        'model_path': 'models/models/wikidata_big/qa_models/tempoqr_full_export.pt',
        'tkbc_path': 'models/models/wikidata_big/kg_embeddings/tcomplex.ckpt'
    }
    
    # Evaluate config from evaluate_full_dataset.py
    eval_config = {
        'model': 'tempoqr',
        'dataset_name': 'wikidata_big',
        'supervision': 'soft',
        'fuse': 'add',
        'extra_entities': False,
        'frozen': 1,
        'lm_frozen': 1,
        'corrupt_hard': 0.0,
        'model_path': 'models/models/wikidata_big/qa_models/tempoqr_full_export.pt',
        'tkbc_path': 'models/models/wikidata_big/kg_embeddings/tcomplex.ckpt'
    }
    
    print("\n📊 API Configuration:")
    for key, value in api_config.items():
        print(f"  {key}: {value}")
    
    print("\n📊 Evaluate Configuration:")
    for key, value in eval_config.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Comparison Results:")
    all_match = True
    for key in api_config:
        if key in eval_config:
            if api_config[key] == eval_config[key]:
                print(f"  ✅ {key}: MATCH")
            else:
                print(f"  ❌ {key}: MISMATCH")
                print(f"    API: {api_config[key]}")
                print(f"    EVAL: {eval_config[key]}")
                all_match = False
        else:
            print(f"  ⚠️  {key}: Missing in eval")
    
    if all_match:
        print("\n🎯 All configurations MATCH!")
    else:
        print("\n❌ Configuration MISMATCHES found!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    compare_model_configs()
