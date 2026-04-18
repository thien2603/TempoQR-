import torch
import numpy as np
from typing import List, Optional
import logging
from .config import settings, model_config
from .qa_tempoqr import QA_TempoQR
from .utils import loadTkbcModel, getAllDicts
from .qa_datasets import QA_Dataset_TempoQR
from argparse import Namespace
from transformers import DistilBertTokenizer

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        # Load TKBC
        tkbc_model = loadTkbcModel(model_config.tkbc_path)
        tkbc_model = tkbc_model.to(self.device)
        # Create args from model_config
        args = Namespace()
        for key, value in model_config.args.items():
            setattr(args, key, value)
        # Load QA model
        self.qa_model = QA_TempoQR(tkbc_model, args)
        checkpoint = torch.load(model_config.model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.qa_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.qa_model.load_state_dict(checkpoint)
        else:
            self.qa_model.load_state_dict(checkpoint)
        self.qa_model = self.qa_model.to(self.device)
        self.qa_model.eval()
        
        self.num_entities = self.qa_model.tkbc_model.sizes[0]
        self.num_times = self.qa_model.tkbc_model.sizes[3]
        self.padding_idx = self.num_entities + self.num_times
        
        all_dicts = getAllDicts(settings.DATASET_NAME)
        self.id2ent = all_dicts['id2ent']
        self.wd_id_to_text = all_dicts['wd_id_to_text']
        self.id2ts = all_dicts['id2ts']
        self.ts2id = all_dicts['ts2id']
        self.ent2id = all_dicts['ent2id']
        
        # Tạo dataset instance để dùng các hàm xử lý
        self.dataset = QA_Dataset_TempoQR(split='valid', dataset_name=settings.DATASET_NAME, args=args)
        self.tokenizer = self.dataset.tokenizer
        
        logger.info(f"ModelManager initialized with {self.num_entities} entities, {self.num_times} times")
    
    def _prepare_question_dict(self, question: str, head_id: int = 0, tail_id: int = 0,
                               start_time: Optional[int] = None, end_time: Optional[int] = None) -> dict:
        # Tìm entities từ câu hỏi (dùng ent2id)
        found_entities = []
        for ent, eid in self.ent2id.items():
            if ent.lower() in question.lower():
                found_entities.append(ent)
        if not found_entities:
            if head_id:
                wd = self.id2ent.get(head_id)
                if wd:
                    name = self.wd_id_to_text.get(wd, '')
                    if name:
                        found_entities.append(name)
            if tail_id and tail_id != head_id:
                wd = self.id2ent.get(tail_id)
                if wd:
                    name = self.wd_id_to_text.get(wd, '')
                    if name:
                        found_entities.append(name)
        
        times = []
        if start_time is not None:
            times.append(start_time)
        if end_time is not None and end_time != start_time:
            times.append(end_time)
        
        annotation = {}
        if head_id:
            wd = self.id2ent.get(head_id)
            if wd:
                annotation['head'] = wd
        if tail_id and tail_id != head_id:
            wd = self.id2ent.get(tail_id)
            if wd:
                annotation['tail'] = wd
        if times:
            annotation['time'] = str(times[0])
        
        qdict = {
            'question': question,
            'entities': found_entities,
            'times': times,
            'relations': [],
            'answer_type': 'entity',
            'template': question,
            'annotation': annotation,
            'paraphrases': [question],
            'type': 'simple_entity',
            'answers': []
        }
        return qdict
    
    def predict_with_time(self, question: str, k: int = 5,
                          head_id: Optional[int] = None, tail_id: Optional[int] = None,
                          start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[str]:
        try:
            if head_id is None:
                head_id = 0
            if tail_id is None:
                tail_id = head_id
            qdict = self._prepare_question_dict(question, head_id, tail_id, start_time, end_time)
            
            # Xây dựng danh sách text (entity names và năm dạng string)
            ent_times_text = []
            ent_times_ids = []   # cũng là text, không phải ID số
            
            # Entities
            entities = qdict.get('entities', [])
            if not isinstance(entities, list):
                entities = [entities] if entities else []
            for ent in entities:
                ent_times_text.append(ent)
                ent_times_ids.append(ent)      # truyền chính text
            
            # Times
            times_val = qdict.get('times', [])
            if not isinstance(times_val, list):
                times_val = [times_val] if times_val is not None else []
            for t in times_val:
                t_str = str(t)
                ent_times_text.append(t_str)
                ent_times_ids.append(t_str)    # truyền string
            
            # Gọi entity-aware tokenization (tự chuyển text sang ID)
            tokenized, entity_time_final, entity_mask = self.dataset.get_entity_aware_tokenization(
                question, ent_times_text, ent_times_ids
            )
            
            input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized)]).long().to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            entity_time_ids_padded = torch.tensor([entity_time_final]).long().to(self.device)
            entity_mask_padded = torch.tensor([entity_mask]).float().to(self.device)
            
            # Xác định head_id, tail_id nếu chưa có
            if head_id == 0:
                valid = [i for i in entity_time_final if i != self.padding_idx]
                head_id = valid[0] if valid else 0
                tail_id = valid[1] if len(valid) > 1 else head_id
            
            # Xác định time_id từ start_time
            time_id = 0
            if start_time is not None and start_time > 0:
                for (y,_,_), tid in self.ts2id.items():
                    if y == start_time:
                        time_id = tid
                        break
            start_time_id = time_id
            end_time_id = time_id
            if end_time is not None and end_time != start_time:
                for (y,_,_), tid in self.ts2id.items():
                    if y == end_time:
                        end_time_id = tid
                        break
            
            heads = torch.tensor([head_id], device=self.device).long()
            tails = torch.tensor([tail_id], device=self.device).long()
            times_t = torch.tensor([time_id], device=self.device).long()
            start_times = torch.tensor([start_time_id], device=self.device).long()
            end_times = torch.tensor([end_time_id], device=self.device).long()
            tails2 = heads.clone()
            dummy_answers = torch.tensor([-1], device=self.device).long()
            
            input_tuple = (input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded,
                           heads, tails, times_t, start_times, end_times, tails2, dummy_answers)
            
            with torch.no_grad():
                scores = self.qa_model.forward(input_tuple)
            
            scores_np = scores[0].cpu().numpy()
            top_indices = np.argsort(scores_np)[-k:][::-1]
            
            answers = []
            for idx in top_indices:
                if idx < self.num_entities:
                    wd = self.id2ent.get(idx)
                    text = self.wd_id_to_text.get(wd, f"entity_{idx}") if wd else f"entity_{idx}"
                    answers.append(text)
                else:
                    time_idx = idx - self.num_entities
                    time_val = self.id2ts.get(time_idx, (0,0,0))[0]
                    answers.append(str(time_val))
            return answers
        except Exception as e:
            logger.error(f"Prediction with time error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def predict(self, question: str, k: int = 5) -> List[str]:
        return self.predict_with_time(question, k=k)

model_manager = ModelManager()