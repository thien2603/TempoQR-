# Import thư viện cần thiết cho dataset processing
from pathlib import Path  # Xử lý file paths
import pkg_resources  # Quản lý package resources
import pickle  # Serialize/deserialize Python objects
from collections import defaultdict  # Dictionary với default values
from typing import Dict, Tuple, List  # Type hints cho code clarity
import json  # Xử lý JSON data

import numpy as np  # Numerical computing
import torch  # PyTorch framework
# from qa_models import QA_model  # Import model class (commented out)
import utils  # Utility functions
from tqdm import tqdm  # Progress bar
from transformers import RobertaTokenizer  # RoBERTa tokenizer
from transformers import DistilBertTokenizer  # DistilBERT tokenizer
from transformers import BertTokenizer  # BERT tokenizer
import random  # Random number generation
from torch.utils.data import Dataset, DataLoader  # PyTorch dataset utilities
# from nltk import word_tokenize  # NLTK tokenization (commented out)
# warning: padding id 0 is being used, can have issue like in Tucker
# however since so many entities (and timestamps?), it may not pose problem
import pdb  # Python debugger
from copy import deepcopy  # Deep copy objects
from collections import defaultdict  # Dictionary với default values (lại import)
import random  # Random number generation (lại import)

from hard_supervision_functions import retrieve_times  # Import time retrieval functions

class QA_Dataset(Dataset):  # Base class cho QA datasets
    def __init__(self,  # Constructor
                split,  # Dataset split: 'train', 'valid', 'test'
                dataset_name,  # Tên dataset: 'wikidata_big'
                tokenization_needed=True):  # Có cần tokenization không
        # Đường dẫn đến file pickle chứa questions
        filename = 'data/data/{dataset_name}/questions/{split}.pickle'.format(
            dataset_name=dataset_name,
            split=split
        )
        # Load questions từ pickle file
        questions = pickle.load(open(filename, 'rb'))
        
        #probably change for bert/roberta?
        # Chọn tokenizer class - mặc định là DistilBERT
        self.tokenizer_class = DistilBertTokenizer 
        # Khởi tạo tokenizer với pre-trained model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load tất cả dictionaries (entities, relations, timestamps)
        self.all_dicts = utils.getAllDicts(dataset_name)
        # In tổng số questions
        print('Total questions = ', len(questions))
        # Lưu questions vào class attribute
        self.data = questions
        # Lưu flag cho tokenization
        self.tokenization_needed = tokenization_needed

    def getEntitiesLocations(self, question):  # Tìm vị trí entities trong câu hỏi
        question_text = question['question']  # Lấy text của câu hỏi
        entities = question['entities']  # Lấy danh sách entities từ question
        ent2id = self.all_dicts['ent2id']  # Lấy entity-to-ID dictionary
        loc_ent = []  # List để lưu (location, entity_id)
        for e in entities:  # Loop qua từng entity
            e_id = ent2id[e]  # Lấy ID của entity
            location = question_text.find(e)  # Tìm vị trí của entity trong text
            loc_ent.append((location, e_id))  # Thêm (vị trí, ID) vào list
        return loc_ent  # Return list các (location, entity_id)

    def getTimesLocations(self, question):  # Tìm vị trí timestamps trong câu hỏi
        question_text = question['question']  # Lấy text của câu hỏi
        times = question['times']  # Lấy danh sách timestamps
        ts2id = self.all_dicts['ts2id']  # Lấy timestamp-to-ID dictionary
        loc_time = []  # List để lưu (location, time_id)
        for t in times:  # Loop qua từng timestamp
            # Tính time ID: ts2id[(t,0,0)] + num_entities
            # (t,0,0) là format của timestamp key
            # + num_entities để separate time IDs từ entity IDs
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id'])
            location = question_text.find(str(t))  # Tìm vị trí của timestamp
            loc_time.append((location, t_id))  # Thêm (vị trí, ID) vào list
        return loc_time  # Return list các (location, time_id)

    def isTimeString(self, s):  # Kiểm tra string có phải là timestamp không
        # todo: cant do len == 4 since 3 digit times also there
        if 'Q' not in s:  # Nếu không có 'Q' (Q-prefixed entity IDs)
            return True  # Là timestamp
        else:
            return False  # Là entity

    def textToEntTimeId(self, text):  # Chuyển text thành entity hoặc time ID
        if self.isTimeString(text):  # Nếu là timestamp
            t = int(text)  # Convert sang integer
            ts2id = self.all_dicts['ts2id']  # Lấy timestamp dictionary
            # Tính time ID với offset
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id'])
            return t_id  # Return time ID
        else:  # Nếu là entity
            ent2id = self.all_dicts['ent2id']  # Lấy entity dictionary
            e_id = ent2id[text]  # Lấy entity ID
            return e_id  # Return entity ID

    def getOrderedEntityTimeIds(self, question):  # Lấy entity/time IDs theo thứ tự xuất hiện
        loc_ent = self.getEntitiesLocations(question)  # Lấy entity locations
        loc_time = self.getTimesLocations(question)  # Lấy time locations
        loc_all = loc_ent + loc_time  # Gộp tất cả locations
        loc_all.sort()  # Sort theo vị trí trong câu
        ordered_ent_time = [x[1] for x in loc_all]  # Chỉ lấy IDs, bỏ locations
        return ordered_ent_time  # Return IDs theo thứ tự

    def entitiesToIds(self, entities):  # Chuyển entity names sang IDs
        output = []  # List để lưu entity IDs
        ent2id = self.all_dicts['ent2id']  # Lấy entity-to-ID dictionary
        for e in entities:  # Loop qua từng entity name
            output.append(ent2id[e])  # Convert name sang ID và thêm vào list
        return output  # Return list của entity IDs
    
    def getIdType(self, id):  # Xác định ID là entity hay time
        if id < len(self.all_dicts['ent2id']):  # Nếu ID < số entities
            return 'entity'  # Là entity ID
        else:  # Nếu ID >= số entities
            return 'time'  # Là time ID
    
    def getEntityToText(self, entity_wd_id):  # Lấy text description của entity
        return self.all_dicts['wd_id_to_text'][entity_wd_id]  # Lookup trong dictionary
    
    def getEntityIdToText(self, id):  # Lấy text từ entity ID
        ent = self.all_dicts['id2ent'][id]  # Convert ID sang WD ID
        return self.getEntityToText(ent)  # Lấy text từ WD ID
    
    def getEntityIdToWdId(self, id):  # Convert entity ID sang WD ID
        return self.all_dicts['id2ent'][id]  # Direct lookup

    def timesToIds(self, times):  # Chuyển timestamps sang IDs
        output = []  # List để lưu time IDs
        ts2id = self.all_dicts['ts2id']  # Lấy timestamp-to-ID dictionary
        for t in times:  # Loop qua từng timestamp
            output.append(ts2id[(t, 0, 0)])  # Convert (t,0,0) sang ID
        return output  # Return list của time IDs

    def getAnswersFromScores(self, scores, largest=True, k=10):  # Lấy top-k answers từ scores
        _, ind = torch.topk(scores, k, largest=largest)  # Lấy indices của top-k scores
        predict = ind  # Indices của top-k answers
        answers = []  # List để lưu answer texts
        for a_id in predict:  # Loop qua từng answer ID
            a_id = a_id.item()  # Convert tensor sang Python int
            type = self.getIdType(a_id)  # Xác định là entity hay time
            if type == 'entity':  # Nếu là entity
                # answers.append(self.getEntityIdToText(a_id))  # Lấy text description
                answers.append(self.getEntityIdToWdId(a_id))  # Lấy WD ID
            else:  # Nếu là time
                time_id = a_id - len(self.all_dicts['ent2id'])  # Tính actual time ID
                time = self.all_dicts['id2ts'][time_id]  # Lấy timestamp tuple
                answers.append(time[0])  # Lấy giá trị timestamp
        return answers  # Return list của answer texts
    
    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):  # Lấy top-k với scores
        s, ind = torch.topk(scores, k, largest=largest)  # Lấy scores và indices
        predict = ind  # Indices của top-k answers
        answers = []  # List để lưu answer texts
        for a_id in predict:  # Loop qua từng answer ID
            a_id = a_id.item()  # Convert tensor sang Python int
            type = self.getIdType(a_id)  # Xác định là entity hay time
            if type == 'entity':  # Nếu là entity
                answers.append(self.getEntityIdToWdId(a_id))  # Lấy WD ID
            else:  # Nếu là time
                time_id = a_id - len(self.all_dicts['ent2id'])  # Tính actual time ID
                time = self.all_dicts['id2ts'][time_id]  # Lấy timestamp tuple
                answers.append(time[0])  # Lấy giá trị timestamp
        return s, answers  # Return cả scores và answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored 
    # while the position with the value of False will be unchanged.
    # 
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len = -1):  # Padding sequences đến cùng độ dài
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)  # Số sequences
        if max_len == -1:  # Nếu không chỉ định max_len
            max_len = max([s.size(0) for s in sequences])  # Tìm độ dài lớn nhất
        out_dims = (num, max_len)  # Kích thước output tensor
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)  # Tạo tensor filled với 0
        # mask = sequences[0].data.new(*out_dims).fill_(0)  # Alternative mask
        mask = torch.ones((num, max_len), dtype=torch.bool)  # Tạo mask filled với True
        for i, tensor in enumerate(sequences):  # Loop qua từng sequence
            length = tensor.size(0)  # Độ dài của sequence hiện tại
            out_tensor[i, :length] = tensor  # Copy sequence vào output tensor
            mask[i, :length] = False  # Đánh dấu positions thực tế với False
        return out_tensor, mask  # Return padded tensor và mask
    
    def toOneHot(self, indices, vec_len):  # Chuyển indices sang one-hot vector
        indices = torch.LongTensor(indices)  # Convert sang LongTensor
        one_hot = torch.FloatTensor(vec_len)  # Tạo float vector
        one_hot.zero_()  # Zero vector
        one_hot.scatter_(0, indices, 1)  # Set 1 tại positions của indices
        return one_hot  # Return one-hot vector

    def prepare_data(self, data):  # Chuẩn bị data cho training
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []  # List để lưu question texts
        entity_time_ids = []  # List để lưu entity/time IDs
        num_total_entities = len(self.all_dicts['ent2id'])  # Tổng số entities
        answers_arr = []  # List để lưu answers
        for question in data:  # Loop qua từng question
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]  # Lấy paraphrase đầu tiên
            question_text.append(q_text)  # Thêm vào list
            et_id = self.getOrderedEntityTimeIds(question)  # Lấy entity/time IDs theo thứ tự
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))  # Convert sang tensor
            if question['answer_type'] == 'entity':  # Nếu answer là entity
                answers = self.entitiesToIds(question['answers'])  # Convert entity names sang IDs
            else:  # Nếu answer là time
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)  # Thêm answers vào list
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text,  # Return prepared data
                'entity_time_ids': entity_time_ids, 
                'answers_arr': answers_arr}
    
    def is_template_keyword(self, word):  # Kiểm tra word có phải template keyword không
        if '{' in word and '}' in word:  # Nếu có braces
            return True  # Là template keyword
        else:
            return False  # Không phải template keyword

    def get_keyword_dict(self, template, nl_question):  # Lấy keyword dictionary từ template
        template_tokenized = self.tokenize_template(template)  # Tokenize template
        keywords = []  # List để lưu keywords
        for word in template_tokenized:  # Loop qua từng word trong template
            if not self.is_template_keyword(word):  # Nếu không phải keyword
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)  # Replace với *
            else:  # Nếu là keyword
                keywords.append(word[1:-1]) # no brackets - bỏ { }
        text_for_keywords = []  # List để lưu text cho keywords
        for word in nl_question.split('*'):  # Split theo *
            if word != '':  # Nếu không rỗng
                text_for_keywords.append(word)  # Thêm vào list
        keyword_dict = {}  # Dictionary để lưu keyword->text mapping
        for keyword, text in zip(keywords, text_for_keywords):  # Zip keywords và texts
            keyword_dict[keyword] = text  # Create mapping
        return keyword_dict  # Return keyword dictionary

    def addEntityAnnotation(self, data):  # Thêm entity annotation vào data
        for i in range(len(data)):  # Loop qua tất cả data
            question = data[i]  # Lấy question hiện tại
            keyword_dicts = [] # we want for each paraphrase
            template = question['template']  # Lấy template
            #for nl_question in question['paraphrases']:
            nl_question =  question['paraphrases'][0]  # Lấy paraphrase đầu tiên
            keyword_dict = self.get_keyword_dict(template, nl_question)  # Lấy keyword dict
            keyword_dicts.append(keyword_dict)  # Thêm vào list
            data[i]['keyword_dicts'] = keyword_dicts  # Lưu vào question data
            #print(keyword_dicts)
            #print(template, nl_question)
        return data  # Return data đã được annotate

    def tokenize_template(self, template):  # Tokenize template string
        output = []  # List để lưu tokens
        buffer = ''  # Buffer để build tokens
        i = 0  # Index pointer
        while i < len(template):  # Loop qua template string
            c = template[i]  # Lấy character hiện tại
            if c == '{':  # Nếu gặp opening brace
                if buffer != '':  # Nếu buffer không rỗng
                    output.append(buffer)  # Thêm buffer vào output
                    buffer = ''  # Reset buffer
                # Đọc đến closing brace
                while template[i] != '}':  # While chưa gặp }
                    buffer += template[i]  # Thêm character vào buffer
                    i += 1  # Increment index
                buffer += template[i]  # Thêm closing brace
                output.append(buffer)  # Thêm keyword vào output
                buffer = ''  # Reset buffer
            else:  # Nếu không phải opening brace
                buffer += c  # Thêm character vào buffer
            i += 1  # Increment index
        if buffer != '':  # Nếu buffer còn lại
            output.append(buffer)  # Thêm buffer vào output
        return output  # Return list của tokens

# ==================== BASELINE DATASET ====================
class QA_Dataset_Baseline(QA_Dataset):  # Baseline QA Dataset class
    def __init__(self, split, dataset_name,  tokenization_needed=True):  # Constructor
        super().__init__(split, dataset_name, tokenization_needed)  # Call parent constructor
        print('Preparing data for split %s' % split)  # In split hiện tại
        # self.data = self.data[:30000]  # Limit data cho testing (commented)
        # new_data = []
        # # qn_type = 'simple_time'
        # # qn_type = 'simple_entity'
        # # print('Only {} questions'.format(qn_type))
        # # for qn in self.data:
        # #     if qn['type'] == qn_type:
        # #         new_data.append(qn)
        # # self.data = new_data
        ents = self.all_dicts['ent2id'].keys()  # Lấy tất cả entity keys
        # Tạo timestamp string to ID mapping
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()  # Lấy tất cả timestamp string keys
        rels = self.all_dicts['rel2id'].keys()  # Lấy tất cả relation keys
        
        # Chuẩn bị data
        self.prepared_data = self.prepare_data_(self.data)  # Chuẩn bị data cho baseline
