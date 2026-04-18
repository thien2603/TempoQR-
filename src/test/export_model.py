import torch
import argparse
from src.core.qa_tempoqr import QA_TempoQR
from src.core.qa_datasets import QA_Dataset_TempoQR
from src.core.utils import loadTkbcModel, getAllDicts
import os

def export_model(checkpoint_path, export_path, dataset_name='wikidata_big', tkbc_model_file='tcomplex.ckpt'):
    """
    Export trained TempoQR model for deployment
    """
    print(f'🔧 Exporting model from {checkpoint_path} to {export_path}')
    
    # Load TKBC model
    tkbc_model = loadTkbcModel(f'models/models/{dataset_name}/kg_embeddings/{tkbc_model_file}')
    print('✅ Loaded TKBC model')
    
    # Create QA model with same config as training
    args = argparse.Namespace()
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.dataset_name = dataset_name
    
    qa_model = QA_TempoQR(tkbc_model, args)
    print('✅ Created QA model')
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        qa_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print('✅ Loaded trained weights')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    # Set to evaluation mode
    qa_model.eval()
    
    # Export full model (includes architecture + weights)
    torch.save({
        'model_state_dict': qa_model.state_dict(),
        'model_config': {
            'model': 'tempoqr',
            'supervision': 'soft',
            'fuse': 'add',
            'extra_entities': False,
            'frozen': 1,
            'lm_frozen': 1,
            'corrupt_hard': 0.0,
            'dataset_name': dataset_name
        },
        'model_class': 'QA_TempoQR'
    }, export_path)
    
    print(f'✅ Model exported to {export_path}')
    
    # Test loading to verify
    try:
        checkpoint = torch.load(export_path, map_location='cpu')
        print('✅ Export verification successful')
        print(f'   Model class: {checkpoint["model_class"]}')
        print(f'   Config: {checkpoint["model_config"]}')
    except Exception as e:
        print(f'❌ Export verification failed: {e}')
    
    return export_path

def export_model_only_weights(checkpoint_path, export_path):
    """
    Export only model weights (smaller file size)
    """
    print(f'🔧 Exporting weights only from {checkpoint_path} to {export_path}')
    
    # Load only state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Save only weights
    torch.save(state_dict, export_path)
    
    print(f'✅ Weights exported to {export_path}')
    
    # Check file size
    file_size = os.path.getsize(export_path) / (1024 * 1024)  # MB
    print(f'📦 File size: {file_size:.2f} MB')
    
    return export_path

def export_onnx(checkpoint_path, onnx_path, dataset_name='wikidata_big', tkbc_model_file='tcomplex.ckpt'):
    """
    Export model to ONNX format for better deployment
    """
    print(f'🔧 Exporting to ONNX: {checkpoint_path} -> {onnx_path}')
    
    # Load model
    tkbc_model = loadTkbcModel(f'models/models/{dataset_name}/kg_embeddings/{tkbc_model_file}')
    
    args = argparse.Namespace()
    args.model = 'tempoqr'
    args.supervision = 'soft'
    args.fuse = 'add'
    args.extra_entities = False
    args.frozen = 1
    args.lm_frozen = 1
    args.corrupt_hard = 0.0
    args.dataset_name = dataset_name
    
    qa_model = QA_TempoQR(tkbc_model, args)
    qa_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    qa_model.eval()
    
    # Create dummy input for tracing
    batch_size = 1
    seq_len = 50
    
    dummy_input = (
        torch.randint(0, 30000, (batch_size, seq_len)),  # input_ids
        torch.ones(batch_size, seq_len),  # attention_mask
        torch.randint(0, 125726, (batch_size, seq_len)),  # entity_time_ids
        torch.zeros(batch_size, seq_len),  # entity_mask
        torch.randint(0, 125726, (batch_size,)),  # heads
        torch.randint(0, 125726, (batch_size,)),  # tails
        torch.randint(0, 1000, (batch_size,)),  # times
        torch.randint(0, 1000, (batch_size,)),  # start_times
        torch.randint(0, 1000, (batch_size,)),  # end_times
        torch.randint(0, 125726, (batch_size,)),  # tails2
        torch.randint(0, 125726, (batch_size, 125726))  # answers_khot
    )
    
    # Move to CPU for ONNX export
    dummy_input_cpu = tuple(x.cpu() for x in dummy_input)
    qa_model.cpu()
    
    # Export to ONNX
    torch.onnx.export(
        qa_model,
        dummy_input_cpu,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'entity_time_ids', 'entity_mask', 
                    'heads', 'tails', 'times', 'start_times', 'end_times', 'tails2', 'answers_khot'],
        output_names=['scores'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'entity_time_ids': {0: 'batch_size', 1: 'seq_len'},
            'entity_mask': {0: 'batch_size', 1: 'seq_len'},
            'heads': {0: 'batch_size'},
            'tails': {0: 'batch_size'},
            'times': {0: 'batch_size'},
            'start_times': {0: 'batch_size'},
            'end_times': {0: 'batch_size'},
            'tails2': {0: 'batch_size'},
            'answers_khot': {0: 'batch_size', 1: 'num_entities'},
            'scores': {0: 'batch_size', 1: 'num_entities'}
        }
    )
    
    print(f'✅ ONNX model exported to {onnx_path}')
    
    # Check file size
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    print(f'📦 File size: {file_size:.2f} MB')
    
    return onnx_path

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "models/models/wikidata_big/qa_models/tempoqr_simple.ckpt"
    
    # Export options
    export_full_path = "models/models/wikidata_big/qa_models/tempoqr_full_export.pt"
    export_weights_path = "models/models/wikidata_big/qa_models/tempoqr_weights_only.pt"
    export_onnx_path = "models/models/wikidata_big/qa_models/tempoqr_model.onnx"
    
    print("🚀 Starting model export process...")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("📝 Please train your model first or specify correct checkpoint path")
        exit(1)
    
    # Option 1: Export full model (recommended for deployment)
    print("\n📦 Option 1: Full model export")
    export_model(checkpoint_path, export_full_path)
    
    # Option 2: Export weights only (smaller file)
    print("\n📦 Option 2: Weights only export")
    export_model_only_weights(checkpoint_path, export_weights_path)
    
    # Option 3: Export to ONNX (for production deployment)
    print("\n📦 Option 3: ONNX export")
    try:
        export_onnx(checkpoint_path, export_onnx_path)
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        print("   This is normal for complex models with custom components")
    
    print("\n🎉 Export process completed!")
    print("📁 Available exports:")
    print(f"   Full model: {export_full_path}")
    print(f"   Weights only: {export_weights_path}")
    print(f"   ONNX model: {export_onnx_path}")
    
    print("\n💡 Next steps:")
    print("   1. Use full model for FastAPI deployment")
    print("   2. Use ONNX model for high-performance deployment")
    print("   3. Create Docker container with exported model")
