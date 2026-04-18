import os
import json
import logging
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from groq import Groq
from torch.utils.data import DataLoader

from .config import settings
from .utils import getAllDicts
from .model_loader import model_manager
from .qa_datasets import QA_Dataset_TempoQR

logger = logging.getLogger(__name__)

class TempoQRAgent:
    def __init__(self, groq_api_key: Optional[str] = None):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            # Hardcode cho test (không commit)
            api_key = "..."
        self.client = Groq(api_key=api_key)
        
        # Load KG dictionaries
        self.all_dicts = getAllDicts(settings.DATASET_NAME)
        self.ent2id = self.all_dicts['ent2id']
        self.rel2id = self.all_dicts['rel2id']
        self.ts2id = self.all_dicts['ts2id']
        self.id2ent = self.all_dicts['id2ent']
        self.id2ts = self.all_dicts['id2ts']
        self.wd_id_to_text = self.all_dicts['wd_id_to_text']
        
        self.num_entities = len(self.ent2id)
        self.num_times = len(self.id2ts)
        
        # Tạo dataset instance để dùng các hàm xử lý
        import argparse as ap
        args = ap.Namespace()
        args.dataset_name = settings.DATASET_NAME
        args.model = 'tempoqr'
        args.supervision = 'soft'
        args.fuse = 'add'
        args.extra_entities = False
        args.frozen = 1
        args.lm_frozen = 1
        args.corrupt_hard = 0.0
        args.tkg_file = f'data/data/{args.dataset_name}/kg/full.txt'
        self.dataset = QA_Dataset_TempoQR(split='valid', dataset_name=args.dataset_name, args=args)
        
        # Model manager (chỉ để lấy qa_model và device)
        self.model_manager = model_manager
        
        # Pre-compile regex for time ranges
        self.time_range_patterns = [
            (re.compile(r'between\s+(\d{4})\s+and\s+(\d{4})', re.IGNORECASE), lambda m: (m.group(1), m.group(2))),
            (re.compile(r'from\s+(\d{4})\s+to\s+(\d{4})', re.IGNORECASE), lambda m: (m.group(1), m.group(2))),
            (re.compile(r'after\s+(\d{4})', re.IGNORECASE), lambda m: (m.group(1), None)),
            (re.compile(r'before\s+(\d{4})', re.IGNORECASE), lambda m: (None, m.group(1))),
            (re.compile(r'in\s+(\d{4})', re.IGNORECASE), lambda m: (m.group(1), m.group(1))),
            (re.compile(r'(\d{4})\s*-\s*(\d{4})', re.IGNORECASE), lambda m: (m.group(1), m.group(2))),
            (re.compile(r'(\d{4})', re.IGNORECASE), lambda m: (m.group(1), m.group(1))),
        ]
        
        logger.info("Agent initialized with Groq + TempoQR")
    
    def _replace_qids(self, text: str) -> str:
        def repl(match):
            qid = match.group(0)
            return self.wd_id_to_text.get(qid, qid)
        return re.sub(r'Q[0-9]+', repl, text)
    
    def extract_time_range(self, question: str) -> Tuple[Optional[int], Optional[int]]:
        start = None
        end = None
        for pattern, extractor in self.time_range_patterns:
            match = pattern.search(question)
            if match:
                s, e = extractor(match)
                if s:
                    start = int(s)
                if e:
                    end = int(e)
                if start is not None and end is not None and start == end:
                    end = None
                break
        return start, end
    
    def extract_components(self, question: str) -> Dict:
        prompt = f"""You are an expert in temporal knowledge graphs. Extract the following components from the question:
- head: the first main subject entity (as a string, exactly as it appears in Wikidata if possible)
- tail: the second main entity (if any), otherwise null
- relation: the predicate (e.g., "president", "born", "found", "award")
- time: the year (if any) as a string, otherwise null
- answer_type: either "entity" or "time"

Return ONLY a JSON object. Examples:
Question: Who was the president of the United States in 2020?
Output: {{"head": "United States", "tail": null, "relation": "president", "time": "2020", "answer_type": "entity"}}
Question: When was Barack Obama born?
Output: {{"head": "Barack Obama", "tail": null, "relation": "born", "time": "1961", "answer_type": "time"}}

Question: {question}
Output:"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"head": None, "tail": None, "relation": None, "time": None, "answer_type": "entity"}
    
    def resolve_entity(self, name: str) -> int:
        if not name:
            return 0
        if name in self.ent2id:
            return self.ent2id[name]
        name_lower = name.lower()
        for ent, eid in self.ent2id.items():
            if ent.lower() == name_lower:
                return eid
        for qid, text in self.wd_id_to_text.items():
            if text.lower() == name_lower:
                if qid in self.ent2id:
                    return self.ent2id[qid]
        return 0
    
    def resolve_time(self, year_str: str) -> int:
        if not year_str:
            return 0
        try:
            year = int(year_str)
        except:
            return 0
        for (y,_,_), tid in self.ts2id.items():
            if y == year:
                return tid
        return 0
    
    def _id_to_text(self, idx: int) -> str:
        if idx < self.num_entities:
            wd = self.id2ent.get(idx)
            return self.wd_id_to_text.get(wd, f"entity_{idx}") if wd else f"entity_{idx}"
        else:
            time_idx = idx - self.num_entities
            return str(self.id2ts.get(time_idx, (0,0,0))[0])
    
    def _create_item_from_components(self, question: str, head_id: int, tail_id: int,
                                      start_time: Optional[int], end_time: Optional[int]) -> tuple:
        # Lấy Q‑id cho entity
        head_qid = self.id2ent.get(head_id, '') if head_id else ''
        tail_qid = self.id2ent.get(tail_id, '') if tail_id and tail_id != head_id else ''
        
        entities_qid = []
        if head_qid:
            entities_qid.append(head_qid)
        if tail_qid and tail_qid != head_qid:
            entities_qid.append(tail_qid)
        
        times = []
        if start_time is not None:
            times.append(str(start_time))
        if end_time is not None and end_time != start_time:
            times.append(str(end_time))
        
        ent_times_text = entities_qid + times
        ent_times_ids = ent_times_text.copy()
        
        # Entity-aware tokenization
        tokenized, entity_time_final, entity_mask = self.dataset.get_entity_aware_tokenization(
            question, ent_times_text, ent_times_ids
        )
        
        # Chuyển tokenized thành list of token ids
        tokenizer = self.dataset.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)
        
        # Trả về tuple đúng định dạng (giống dataset.__getitem__)
        return (question, token_ids, entity_time_final, entity_mask,
                head_id, tail_id, 0, 0, 0, head_id, -1)
    
    def predict(self, question: str, k: int = 5) -> List[str]:
        logger.info(f"Original question: {question}")
        question_replaced = self._replace_qids(question)
        if question_replaced != question:
            logger.info(f"After QID replacement: {question_replaced}")
        
        # Time range
        start_time, end_time = self.extract_time_range(question_replaced)
        if start_time is not None or end_time is not None:
            logger.info(f"Time range: start={start_time}, end={end_time}")
        
        # LLM extraction
        comp = self.extract_components(question_replaced)
        logger.info(f"LLM extraction: {comp}")
        
        head_name = comp.get("head")
        tail_name = comp.get("tail")
        time_str = comp.get("time")
        
        head_id = self.resolve_entity(head_name) if head_name else 0
        tail_id = self.resolve_entity(tail_name) if tail_name else head_id
        
        if start_time is None and time_str and time_str.isdigit():
            start_time = end_time = int(time_str)
        
        # Tạo item và batch
        item = self._create_item_from_components(question_replaced, head_id, tail_id, start_time, end_time)
        loader = DataLoader([item], batch_size=1, collate_fn=self.dataset._collate_fn)
        batch = next(iter(loader))
        
        # Chuyển sang device
        device = self.model_manager.device
        batch = [t.to(device) if hasattr(t, 'to') else t for t in batch]
        
        with torch.no_grad():
            scores = self.model_manager.qa_model.forward(batch)
        
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
        logger.info(f"Predictions: {answers}")
        return answers

# Singleton
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = TempoQRAgent()
    return agent