import torch
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
# Import model và utils
from qa_tempoqr import QA_TempoQR
from utils import loadTkbcModel

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(title="TempoQR QA API", description="API cho mô hình Temporal Knowledge Graph QA")

# Cấu hình đường dẫn model bạn vừa export
MODEL_PATH = "models/models/wikidata_big/qa_models/tempoqr_full_export.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến toàn cục để lưu model và tokenizer
qa_model = None
tokenizer = None

# 2. Định nghĩa cấu trúc dữ liệu đầu vào/đầu ra (Pydantic)
class QuestionRequest(BaseModel):
    question: str
    # Trong thực tế, bạn có thể cần truyền thêm các ID thực thể đã được nhận diện
    # head_ids: list[int] = [] 

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float

# 3. Sự kiện khởi chạy Server (Load model 1 lần duy nhất vào RAM/VRAM)
@app.on_event("startup")
async def load_model():
    global qa_model, tokenizer
    print(f"Loading model from {MODEL_PATH} to {device}...")
    
    try:
        # Load weights của mô hình
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Check if checkpoint is a full export or just weights
        # isinstance(checkpoint, dict) checks if checkpoint is a dictionary or not
        # If it is a dictionary, it could be a full export format, which stores the entire information of the model (state_dict, config, and model_class)
        # If it is not a dictionary, it could only store the state_dict of the model.
        if isinstance(checkpoint, dict):
            # Full export format
            # In the full export format, the checkpoint contains all the information of the model, including state_dict, config, and model_class.
            # In the if block, the state_dict is retrieved from the checkpoint and stored in the variable model_state_dict to recreate the model.
            # The variable model_config stores the config of the model in the checkpoint, while the variable model_class stores the information about the model's class in the checkpoint.
            model_state_dict = checkpoint['model_state_dict']
            model_config = checkpoint['model_config']
            model_class = checkpoint['model_class']
            print(f"✅ Loaded full export with config: {model_config} and model class: {model_class}")
        else:
            # Just weights
            # In the else block, the checkpoint only contains the state_dict of the model, without config or model_class.
            # Therefore, only the state_dict is retrieved from the checkpoint and stored in the variable model_state_dict to recreate the model.
            model_state_dict = checkpoint
            print("✅ Loaded weights only")
        # Create model instance first
        # Load TKBC model
        tkbc_model = loadTkbcModel('models/models/wikidata_big/kg_embeddings/tcomplex.ckpt')

        # Create QA model with default config
        args = argparse.Namespace()
        args.model = 'tempoqr'
        args.supervision = 'soft'
        args.fuse = 'add'
        args.extra_entities = False
        args.frozen = 1
        args.lm_frozen = 1
        args.corrupt_hard = 0.0
        args.dataset_name = 'wikidata_big'
        
        qa_model = QA_TempoQR(tkbc_model, args)
        
        # Load weights into model
        qa_model.load_state_dict(model_state_dict)
        qa_model.eval() # Chuyển sang chế độ suy luận (không train)
        
        # Load Tokenizer của DistilBERT (vì TempoQR dùng model này để đọc chữ)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("✅ Model and Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# 4. Tạo Endpoint API để nhận câu hỏi và trả kết quả
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if qa_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded yet.")
    
    try:
        # --- BƯỚC TIỀN XỬ LÝ (PREPROCESSING) ---
        # 1. Tokenize câu hỏi văn bản
        inputs = tokenizer(request.question, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # [QUAN TRỌNG]: TempoQR cần thêm thông tin Đồ thị tri thức (entities, times).
        # Khi làm thật, bạn phải có hàm tìm kiếm ID của thực thể trong câu hỏi.
        dummy_heads = torch.tensor([0], dtype=torch.long).to(device)
        dummy_tails = torch.tensor([0], dtype=torch.long).to(device)
        dummy_times = torch.tensor([0], dtype=torch.long).to(device)
        
        # --- BƯỚC SUY LUẬN (INFERENCE) ---
        with torch.no_grad():
            # Chuẩn bị input tensors cho TempoQR
            # TempoQR cần: (input_ids, attention_mask, entity_time_ids, entity_mask, heads, tails, times, start_times, end_times, tails2)
            
            # 1.1 Entity/Time IDs từ question text (cần implement NER)
            # Tạm thởi dùng dummy data - CẦN IMPLEMENT THẬT!
            entity_time_ids = torch.tensor([[1, 2, 3]], dtype=torch.long).to(device)  # Mock entities
            entity_mask = torch.ones_like(entity_time_ids, dtype=torch.bool).to(device)    # Mock mask
            
            # 1.2 Temporal information (cần extract từ question)
            heads = torch.tensor([1], dtype=torch.long).to(device)      # Mock head entity
            tails = torch.tensor([2], dtype=torch.long).to(device)      # Mock tail entity  
            times = torch.tensor([2020], dtype=torch.long).to(device)   # Mock timestamp
            start_times = torch.tensor([2019], dtype=torch.long).to(device)  # Mock start time
            end_times = torch.tensor([2021], dtype=torch.long).to(device)    # Mock end time
            tails2 = torch.tensor([3], dtype=torch.long).to(device)      # Mock tail2 entity
            
            # 1.3 Padding cho sequences (nếu cần)
            max_seq_len = 50  # Max sequence length
            if input_ids.size(1) < max_seq_len:
                # Pad input_ids và attention_mask
                pad_len = max_seq_len - input_ids.size(1)
                input_ids = torch.cat([
                    input_ids,
                    torch.zeros(input_ids.size(0), pad_len, dtype=torch.long).to(device)
                ], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(input_ids.size(0), pad_len, dtype=torch.long).to(device)
                ], dim=1)
            
            # Forward pass qua TempoQR model
            print(f"Input shapes:")
            print(f"   input_ids: {input_ids.shape}")
            print(f"   attention_mask: {attention_mask.shape}")
            print(f"   entity_time_ids: {entity_time_ids.shape}")
            print(f"   heads: {heads.shape}")
            print(f"   tails: {tails.shape}")
            print(f"   times: {times.shape}")
            
            # Gọi hàm forward của QA_TempoQR
            scores = qa_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_time_ids=entity_time_ids,
                entity_mask=entity_mask,
                heads=heads,
                tails=tails,
                times=times,
                start_times=start_times,
                end_times=end_times,
                tails2=tails2
            )
            
            print(f"Output scores shape: {scores.shape}")
            print(f"Sample scores: {scores[0][:5]}")  # 5 scores đầu tiên 
            
        # --- BƯỚC HẬU XỬ LÝ (POST-PROCESSING) ---
        # Lấy ra ID của câu trả lời có điểm score cao nhất
        # top_answer_id = torch.argmax(scores, dim=1).item()
        
        # Map ID đó ngược lại thành Text (thông qua file từ điển wd_id2entity_text.txt)
        final_answer = "Steve Jobs (Demo Data)" 
        confidence_score = 0.95
        
        return AnswerResponse(
            question=request.question,
            answer=final_answer,
            confidence=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lệnh để kiểm tra server đang sống
@app.get("/")
async def root():
    return {"message": "TempoQR API is running. Send POST request to /ask"}