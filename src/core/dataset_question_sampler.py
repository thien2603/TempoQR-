#!/usr/bin/env python3
"""
📊 Dataset Question Sampler - Lấy câu hỏi mẫu từ dataset file
"""

import os
import sys
import pickle
import random
from typing import Dict, List, Optional

class DatasetQuestionSampler:
    """
    Class để lấy câu hỏi mẫu từ dataset file
    """
    
    def __init__(self, dataset_name: str = 'wikidata_big', split: str = 'valid'):
        """
        Khởi tạo DatasetQuestionSampler
        
        Args:
            dataset_name: Tên dataset
            split: Split (train/valid/test)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load dataset file
        dataset_file = os.path.join(self.project_root, f'data/data/{dataset_name}/questions/{split}.pickle')
        
        if os.path.exists(dataset_file):
            print(f"📁 Loading dataset from: {dataset_file}")
            with open(dataset_file, 'rb') as f:
                self.questions = pickle.load(f)
            print(f"✅ Loaded {len(self.questions)} questions from {split} split")
        else:
            print(f"❌ Dataset file not found: {dataset_file}")
            self.questions = []
    
    def get_random_question(self) -> Optional[Dict]:
        """
        Lấy một câu hỏi ngẫu nhiên
        
        Returns:
            Optional[Dict]: Câu hỏi hoặc None nếu không có
        """
        if not self.questions:
            print("❌ No questions loaded")
            return None
        
        return random.choice(self.questions)
    
    def get_n_questions(self, n: int = 5) -> List[Dict]:
        """
        Lấy n câu hỏi ngẫu nhiên
        
        Args:
            n: Số câu hỏi cần lấy
            
        Returns:
            List[Dict]: Danh sách n câu hỏi
        """
        if not self.questions:
            print("❌ No questions loaded")
            return []
        
        if n > len(self.questions):
            n = len(self.questions)
            print(f"⚠️ Requested {n} questions, but only {len(self.questions)} available")
        
        return random.sample(self.questions, n)
    
    def get_all_questions(self) -> List[Dict]:
        """
        Lấy tất cả câu hỏi
        
        Returns:
            List[Dict]: Tất cả câu hỏi
        """
        return self.questions
    
    def print_question_details(self, question: Dict, index: int = None):
        """
        In ra chi tiết câu hỏi
        
        Args:
            question: Dict chứa thông tin câu hỏi
            index: Số thứ tự (tùy chọn)
        """
        prefix = f"Q{index}: " if index is not None else ""
        print(f"{prefix}Question: {question.get('question', 'N/A')}")
        
        # In ra entities
        if 'entities' in question:
            print(f"{prefix}Entities: {question['entities']}")
        else:
            print(f"{prefix}Entities: None")
        
        # In ra times
        if 'times' in question:
            print(f"{prefix}Times: {question['times']}")
        else:
            print(f"{prefix}Times: None")
        
        # In ra answers
        if 'answers' in question:
            print(f"{prefix}Answers: {question['answers']}")
        else:
            print(f"{prefix}Answers: None")
        
        print("-" * 50)
    
    def print_summary(self):
        """
        In ra tóm tắt về dataset
        """
        if not self.questions:
            print("❌ No questions loaded")
            return
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Split: {self.split}")
        print(f"  Total Questions: {len(self.questions)}")
        
        # Thống kê entities
        all_entities = set()
        all_times = set()
        
        for question in self.questions:
            if 'entities' in question:
                all_entities.update(question['entities'])
            if 'times' in question:
                all_times.update(question['times'])
        
        print(f"  Unique Entities: {len(all_entities)}")
        print(f"  Unique Times: {len(all_times)}")
        
        # In ra vài câu hỏi mẫu
        print(f"\n📝 Sample Questions:")
        sample_questions = self.get_n_questions(3)
        for i, question in enumerate(sample_questions, 1):
            self.print_question_details(question, i)
    
    def save_sample_questions(self, output_file: str = 'sample_questions.txt', n: int = 10):
        """
        Lưu n câu hỏi mẫu ra file
        
        Args:
            output_file: Tên file output
            n: Số câu hỏi cần lưu
        """
        sample_questions = self.get_n_questions(n)
        
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Sample Questions from {self.dataset_name} - {self.split} split\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, question in enumerate(sample_questions, 1):
                f.write(f"# Question {i}\n")
                f.write(f"Question: {question.get('question', 'N/A')}\n")
                
                if 'entities' in question:
                    f.write(f"Entities: {', '.join(question['entities'])}\n")
                else:
                    f.write("Entities: None\n")
                
                if 'times' in question:
                    f.write(f"Times: {', '.join(map(str, question['times']))}\n")
                else:
                    f.write("Times: None\n")
                
                if 'answers' in question:
                    f.write(f"Answers: {', '.join(map(str, question['answers']))}\n")
                else:
                    f.write("Answers: None\n")
                
                f.write("\n" + "="*50 + "\n")
        
        print(f"✅ Saved {n} sample questions to: {output_path}")


# Test function
def test_dataset_question_sampler():
    """
    Test function cho DatasetQuestionSampler
    """
    print("🧪 Dataset Question Sampler Test")
    print("=" * 60)
    
    # Tạo sampler
    sampler = DatasetQuestionSampler(dataset_name='wikidata_big', split='valid')
    
    # In ra summary
    sampler.print_summary()
    
    # Lấy và in ra 1 câu hỏi ngẫu nhiên
    print(f"\n🎲 Random Question:")
    random_question = sampler.get_random_question()
    if random_question:
        sampler.print_question_details(random_question)
    
    # Lấy và in ra 5 câu hỏi mẫu
    print(f"\n📝 5 Sample Questions:")
    sample_questions = sampler.get_n_questions(5)
    for i, question in enumerate(sample_questions, 1):
        sampler.print_question_details(question, i)
    
    # Lưu 10 câu hỏi mẫu ra file
    sampler.save_sample_questions('sample_questions_from_dataset.txt', 10)
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    test_dataset_question_sampler()
