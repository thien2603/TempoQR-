#!/usr/bin/env python3
"""
📊 Question Processor - Xử lý phân tách entity và time từ câu hỏi
"""

import re
from typing import Dict, List, Tuple, Optional

class QuestionProcessor:
    """
    Class xử lý phân tách entities và times từ câu hỏi
    """
    
    def __init__(self, all_dicts: Dict):
        """
        Khởi tạo QuestionProcessor với dictionaries
        """
        self.all_dicts = all_dicts
        self.ent2id = all_dicts['ent2id']
        self.id2ent = all_dicts['id2ent']
        self.id2ts = all_dicts['id2ts']
        self.ts2id = all_dicts['ts2id']
        self.wd_id_to_text = all_dicts['wd_id_to_text']
        
    def extract_entities_from_question(self, question_text: str) -> List[str]:
        """
        Trích xuất entities từ câu hỏi
        
        Args:
            question_text: Câu hỏi cần xử lý
            
        Returns:
            List[str]: Danh sách các entities tìm thấy
        """
        found_entities = []
        
        # Tìm entities dựa trên dictionary ent2id
        for ent_name in self.ent2id.keys():
            if ent_name.lower() in question_text.lower():
                found_entities.append(ent_name)
        
        return found_entities
    
    def extract_times_from_question(self, question_text: str) -> List[str]:
        """
        Trích xuất times (năm) từ câu hỏi
        
        Args:
            question_text: Câu hỏi cần xử lý
            
        Returns:
            List[str]: Danh sách các times tìm thấy
        """
        # Tìm các năm trong câu hỏi (4 chữ số hoặc 5 chữ số)
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', question_text)
        
        return years
    
    def convert_entities_to_ids(self, entities: List[str]) -> List[int]:
        """
        Chuyển entities sang IDs
        
        Args:
            entities: Danh sách entities
            
        Returns:
            List[int]: Danh sách entity IDs
        """
        entity_ids = []
        for entity in entities:
            if entity in self.ent2id:
                entity_ids.append(self.ent2id[entity])
        
        return entity_ids
    
    def convert_times_to_ids(self, times: List[str]) -> List[int]:
        """
        Chuyển times (năm) sang IDs
        
        Args:
            times: Danh sách times (năm)
            
        Returns:
            List[int]: Danh sách time IDs
        """
        time_ids = []
        for year in times:
            year = int(year)
            # Tìm time ID từ ts2id
            for (y, _, _), tid in self.ts2id.items():
                if y == year:
                    time_ids.append(tid + len(self.ent2id))
                    break
        
        return time_ids
    
    def extract_question_attributes(self, question_text: str, max_length: int = 256) -> Dict:
        """
        Trích xuất tất cả attributes từ câu hỏi
        
        Args:
            question_text: Câu hỏi cần xử lý
            max_length: Độ dài tối đa của token
            
        Returns:
            Dict: Dictionary chứa entities, times, và IDs tương ứng
        """
        # Trích xuất entities
        entities = self.extract_entities_from_question(question_text)
        
        # Trích xuất times
        times = self.extract_times_from_question(question_text)
        
        # Chuyển sang IDs
        entity_ids = self.convert_entities_to_ids(entities)
        time_ids = self.convert_times_to_ids(times)
        
        # Kết hợp entities và times
        all_text = entities + times
        
        return {
            'question_text': question_text,
            'entities': entities,
            'times': times,
            'entity_ids': entity_ids,
            'time_ids': time_ids,
            'all_text': all_text,
            'max_length': max_length
        }
    
    def get_entity_name_by_id(self, entity_id: int) -> str:
        """
        Lấy tên entity từ ID
        
        Args:
            entity_id: Entity ID
            
        Returns:
            str: Tên entity
        """
        if entity_id in self.id2ent:
            wd_id = self.id2ent[entity_id]
            return self.wd_id_to_text.get(wd_id, f"Entity_{entity_id}")
        else:
            return f"Entity_{entity_id}"
    
    def get_time_by_id(self, time_id: int) -> str:
        """
        Lấy time từ ID
        
        Args:
            time_id: Time ID
            
        Returns:
            str: Time value
        """
        if time_id >= len(self.ent2id):
            actual_time_id = time_id - len(self.ent2id)
            if actual_time_id in self.id2ts:
                return str(self.id2ts[actual_time_id][0])
            else:
                return f"Time_{actual_time_id}"
        else:
            return f"Entity_{time_id}"
    
    def format_question_for_model(self, question_text: str, max_length: int = 256) -> Tuple:
        """
        Format câu hỏi cho model input
        
        Args:
            question_text: Câu hỏi cần format
            max_length: Độ dài tối đa
            
        Returns:
            Tuple: (input_ids, attention_mask, entity_time_ids, entity_mask, heads, tails, times, start_times, end_times, tails2, dummy_answers)
        """
        # Trích xuất attributes
        attributes = self.extract_question_attributes(question_text, max_length)
        
        # Tạo input tensors
        import torch
        
        # Tokenize câu hỏi
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Tạo entity_time_ids tensor
        num_entities = len(self.ent2id)
        num_times = len(self.id2ts)
        padding_idx = num_entities + num_times
        
        # Sử dụng entity_ids và time_ids từ attributes
        seq_len = input_ids.shape[1]
        entity_time_ids = torch.full((1, seq_len), padding_idx, dtype=torch.long)
        entity_mask = torch.ones((1, seq_len), dtype=torch.float)
        
        # Đặt entity_time_ids dựa trên attributes
        for i, entity_id in enumerate(attributes['entity_ids'][:seq_len]):
            entity_time_ids[0, i] = entity_id
        
        # Tạo các tensors khác
        heads = torch.tensor([attributes['entity_ids'][0] if attributes['entity_ids'] else 0]).long()
        tails = torch.tensor([attributes['entity_ids'][1] if len(attributes['entity_ids']) > 1 else heads[0]]).long()
        times = torch.tensor([0]).long()
        start_times = torch.tensor([0]).long()
        end_times = torch.tensor([0]).long()
        tails2 = torch.tensor([heads[0]]).long()
        dummy_answers = torch.tensor([-1]).long()
        
        return (
            input_ids, attention_mask, entity_time_ids, entity_mask,
            heads, tails, times, start_times, end_times, tails2, dummy_answers
        )
    
    def print_question_analysis(self, question_text: str):
        """
        In ra phân tích câu hỏi
        
        Args:
            question_text: Câu hỏi cần phân tích
        """
        print(f"\n📝 Phân tích câu hỏi: {question_text}")
        print("-" * 50)
        
        # Trích xuất attributes
        attributes = self.extract_question_attributes(question_text)
        
        # In ra entities
        if attributes['entities']:
            print("🏷️ Entities tìm thấy:")
            for i, entity in enumerate(attributes['entities'], 1):
                entity_id = attributes['entity_ids'][i-1] if i-1 < len(attributes['entity_ids']) else 0
                entity_name = self.get_entity_name_by_id(entity_id)
                print(f"  {i}. {entity} (ID: {entity_id})")
        else:
            print("🏷️ Entities tìm thấy: Không có")
        
        # In ra times
        if attributes['times']:
            print("📅 Times tìm thấy:")
            for i, time in enumerate(attributes['times'], 1):
                time_id = attributes['time_ids'][i-1] if i-1 < len(attributes['time_ids']) else 0
                time_value = self.get_time_by_id(time_id)
                print(f"  {i}. {time} (ID: {time_id})")
        else:
            print("📅 Times tìm thấy: Không có")
        
        print("-" * 50)


# Test function
def test_question_processor():
    """
    Test function cho QuestionProcessor
    """
    # Mock dictionaries (giống như trong dataset)
    mock_dicts = {
        'ent2id': {
            'Barack Obama': 76,
            'United States': 30,
            'Donald Trump': 51657,
            'France': 142,
            'Apple Inc.': 119129,
            'Steve Jobs': 124231,
            'president': 100872,  # Thêm từ valid split
            'Nobel Prize': 125525,  # Thêm từ valid split
            'capital': 100872,  # Thêm từ valid split
        },
        'id2ent': {
            76: 'Q76',
            30: 'Q30',
            51657: 'Q51657',
            142: 'Q142',
            119129: 'Q119129',
            124231: 'Q124231',
            100872: 'Q100872',  # Thêm từ valid split
            125525: 'Q125525',  # Thêm từ valid split
        },
        'id2ts': {
            0: (2020, 1, 1),  # 2020
            1: (2015, 1, 1),  # 2015
            2: (2021, 1, 1),  # 2021
        },
        'ts2id': {
            (2020, 1, 1): 0,
            (2015, 1, 1): 1,
            (2021, 1, 1): 2
        },
        'wd_id_to_text': {
            'Q76': 'Barack Obama',
            'Q30': 'United States',
            'Q51657': 'Donald Trump',
            'Q142': 'France',
            'Q119129': 'Apple Inc.',
            'Q124231': 'Steve Jobs',
            'Q100872': 'president',  # Thêm từ valid split
            'Q125525': 'Nobel Prize'  # Thêm từ valid split
        }
    }
    
    # Tạo processor
    processor = QuestionProcessor(mock_dicts)
    
    # Test câu hỏi (bao gồm cả từ valid split)
    test_questions = [
        "Who was the president of the United States in 2020?",
        "What happened in 2015?",
        "Who won the Nobel Prize in 2021?",
        "What company did Steve Jobs found?",
        "What is the capital of France?",
        # Thêm các câu hỏi từ valid split
        "Who was the president in 2020?",  # Dùng entity 'president'
        "What is the capital city?",  # Dùng entity 'capital'
        "When did the Nobel Prize ceremony happen?",  # Dùng entity 'Nobel Prize'
        "Where is the Apple headquarters?",  # Dùng entity 'Apple Inc.'
    ]
    
    print("🧪 Question Processor Test (với entities từ valid split)")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {question}")
        print('=' * 60)
        
        # Phân tích câu hỏi
        processor.print_question_analysis(question)
        
        # Format cho model
        input_tuple = processor.format_question_for_model(question)
        
        print(f"\n🎯 Model Input Shape:")
        print(f"  Input IDs: {input_tuple[0].shape}")
        print(f"  Attention Mask: {input_tuple[1].shape}")
        print(f"  Entity Time IDs: {input_tuple[2].shape}")
        print(f"  Entity Mask: {input_tuple[3].shape}")
        print(f"  Heads: {input_tuple[4]}")
        print(f"  Tails: {input_tuple[5]}")
        print(f"  Times: {input_tuple[6]}")
        print(f"  Start Times: {input_tuple[7]}")
        print(f"  End Times: {input_tuple[8]}")
        print(f"  Tails2: {input_tuple[9]}")
        print(f"  Dummy Answers: {input_tuple[10]}")


if __name__ == "__main__":
    test_question_processor()
