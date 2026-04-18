import torch
import numpy as np
from transformers import DistilBertTokenizer
from src.core.qa_tempoqr import QA_TempoQR
from src.core.utils import loadTkbcModel

def test_model_loading():
    """Test model loading without data"""
    
    print("🔧 Testing Model Loading...")
    
    # Test 1: Load TKBC model
    try:
        print("1. Loading TKBC model...")
        tkbc_model = loadTkbcModel('models/models/wikidata_big/kg_embeddings/tcomplex.ckpt')
        print("✅ TKBC model loaded successfully!")
        print(f"   - Entities: {tkbc_model.embeddings[0].weight.shape[0]}")
        print(f"   - Relations: {tkbc_model.embeddings[1].weight.shape[0]}")
        print(f"   - Times: {tkbc_model.embeddings[2].weight.shape[0]}")
    except Exception as e:
        print(f"❌ TKBC model loading failed: {e}")
        return False
    
    # Test 2: Create dummy args
    class DummyArgs:
        def __init__(self):
            self.model = 'tempoqr'
            self.supervision = 'soft'
            self.extra_entities = False
            self.fuse = 'add'
            self.lm_frozen = 1
            self.frozen = 1
    
    args = DummyArgs()
    
    # Test 3: Create QA model
    try:
        print("2. Creating QA model...")
        qa_model = QA_TempoQR(tkbc_model, args)
        
        # Try GPU first, fallback to CPU
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.memory_reserved(0) < 3e9 else 'cpu'
        qa_model = qa_model.to(device)
        print(f"✅ QA model created successfully on {device}!")
        print(f"   - Embedding dim: {qa_model.tkbc_embedding_dim}")
        print(f"   - Transformer dim: {qa_model.transformer_dim}")
    except Exception as e:
        print(f"❌ QA model creation failed: {e}")
        return False
    
    # Test 4: Test tokenizer
    try:
        print("3. Testing tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        test_text = "What is the capital of France?"
        tokens = tokenizer.tokenize(test_text)
        print("✅ Tokenizer works!")
        print(f"   - Text: {test_text}")
        print(f"   - Tokens: {tokens}")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False
    
    # Test 5: Test model forward with dummy data
    try:
        print("4. Testing model forward pass...")
        
        # Create dummy data
        batch_size = 2
        seq_length = 10
        device = next(qa_model.parameters()).device
        
        # Dummy input_ids (tokenized question)
        input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        
        # Dummy entity/time data
        heads = torch.randint(0, 1000, (batch_size,)).to(device).long()
        tails = torch.randint(0, 1000, (batch_size,)).to(device).long()
        times = torch.randint(0, 1000, (batch_size,)).to(device).long()
        t1 = torch.randint(0, 1000, (batch_size,)).to(device).long()
        t2 = torch.randint(0, 1000, (batch_size,)).to(device).long()
        tails2 = torch.randint(0, 1000, (batch_size,)).to(device).long()
        
        # Dummy entity_time_ids and mask
        entity_time_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device).long()
        entity_mask = torch.zeros(batch_size, seq_length).to(device)
        entities_times_padded = torch.randint(0, 1000, (batch_size, seq_length)).to(device).long()
        
        # Create dummy batch tuple
        dummy_batch = (input_ids, attention_mask, entity_time_ids, entity_mask, 
                      heads, tails, times, t1, t2, tails2, entities_times_padded)
        
        # Test forward pass
        qa_model.eval()
        with torch.no_grad():
            scores = qa_model.forward(dummy_batch)
        
        print("✅ Model forward pass works!")
        print(f"   - Scores shape: {scores.shape}")
        print(f"   - Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! Model is ready for training!")
    return True

def test_gpu_memory():
    """Test GPU memory"""
    print("\n🔧 Testing GPU Memory...")
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"   - Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   - Free memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("❌ No GPU available!")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEMPOQR MODEL TEST")
    print("=" * 60)
    
    # Test GPU
    gpu_ok = test_gpu_memory()
    
    # Test model
    if gpu_ok:
        model_ok = test_model_loading()
        
        if model_ok:
            print("\n✅ Ready to train!")
            print("Run: python train_qa_model.py --model tempoqr --dataset_name wikidata_big --tkbc_model_file tcomplex.ckpt --tkg_file full.txt --supervision soft --max_epochs 1 --batch_size 2 --lr 2e-4 --valid_freq 1 --eval_k 10 --frozen 1 --lm_frozen 1 --fuse add --extra_entities False --corrupt_hard 0.0 --mode train --save_to tempoqr_test")
        else:
            print("\n❌ Model test failed!")
    else:
        print("\n❌ GPU test failed!")
