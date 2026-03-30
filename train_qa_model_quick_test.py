import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np
import random
from datetime import datetime
import os
from collections import defaultdict

# Import các models và datasets
from qa_tempoqr import QA_TempoQR
from qa_baselines import QA_lm, QA_embedkgqa, QA_cronkgqa
from qa_datasets import QA_Dataset_Baseline, QA_Dataset_TempoQR
from utils import print_info, getAllDicts, loadTkbcModel, loadTkbcModel_complex
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    """
    Đặt random seed để đảm bảo kết quả có thể tái tạo
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parse command line arguments
parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--model', default='tempoqr', type=str,
                    help="Model type: tempoqr, embedkgqa, cronkgqa, bert, roberta")
parser.add_argument('--supervision', default='soft', type=str,
                    help="Supervision type: soft, hard, none")
parser.add_argument('--fuse', default='add', type=str,
                    help="Fusion method: add, cat")
parser.add_argument('--extra_entities', default=False, type=bool,
                    help="Whether to use extra entities for before/after questions")

# Dataset arguments
parser.add_argument('--dataset_name', default='wikidata_big', type=str,
                    help="Dataset name")
parser.add_argument('--tkbc_model_file', default='tcomplex.ckpt', type=str,
                    help="TKBC model checkpoint file")
parser.add_argument('--tkg_file', default='full.txt', type=str,
                    help="TKG file for hard supervision")

# Training arguments
parser.add_argument('--max_epochs', default=20, type=int,
                    help="Maximum number of training epochs")
parser.add_argument('--batch_size', default=32, type=int,
                    help="Training batch size")
parser.add_argument('--valid_batch_size', default=128, type=int,
                    help="Validation batch size")
parser.add_argument('--lr', default=2e-4, type=float,
                    help="Learning rate")
parser.add_argument('--valid_freq', default=1, type=int,
                    help="Validation frequency (every N epochs)")
parser.add_argument('--eval_k', default=10, type=int,
                    help="Hits@k for evaluation")

# Model configuration
parser.add_argument('--frozen', default=1, type=int,
                    help="Whether to freeze KG embeddings (1=frozen, 0=trainable)")
parser.add_argument('--lm_frozen', default=1, type=int,
                    help="Whether to freeze language model (1=frozen, 0=trainable)")
parser.add_argument('--corrupt_hard', default=0.0, type=float,
                    help="Hard corruption probability")

# Training mode and saving
parser.add_argument('--mode', default='train', type=str,
                    help="Mode: train, eval, test_kge")
parser.add_argument('--load_from', default='', type=str,
                    help="Load model from checkpoint")
parser.add_argument('--save_to', default='', type=str,
                    help="Save model to checkpoint")
parser.add_argument('--eval_split', default='valid', type=str,
                    help="Evaluation split: valid, test")
parser.add_argument('--test', default="test", type=str,
                    help="Test data split.")

# QUICK TEST ARGUMENTS
parser.add_argument('--quick_test', default=True, type=bool,
                    help="Quick test with small subset of data")
parser.add_argument('--train_samples', default=100, type=int,
                    help="Number of training samples for quick test")
parser.add_argument('--test_samples', default=50, type=int,
                    help="Number of test samples for quick test")

args = parser.parse_args()
print_info(args)

# Set random seed for reproducibility
set_seed(42)


def eval(qa_model, dataset, batch_size = 128, split='valid', k=10):
    """
    Đánh giá model performance trên dataset
    """
    num_workers = 0  # Disable multiprocessing để tránh memory issues trên Windows
    qa_model.eval()  # Chuyển sang evaluation mode
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k
    k_list = [1,10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    # Tạo DataLoader cho evaluation
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    # Evaluation loop
    for i_batch, a in enumerate(loader):
        if i_batch * batch_size == len(dataset.data):
            break
        
        answers_khot = a[-1]  # Target answers
        scores = qa_model.forward(a)
        
        # Lấy top-k predictions
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        
        # Tính loss
        loss = qa_model.loss(scores, answers_khot.cuda().long())
        total_loss += loss.item()
    
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # Tính Hits@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['type']
            
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            
            entity_time_type = question['answer_type']
            predicted = topk_answers[i][:k]
            
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
            
            question_types_count[question_type].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1
        
        eval_accuracy = hits_at_k / total
        eval_accuracy_for_reporting = eval_accuracy
        
        eval_log.append('Hits at %d: %f' % (k, eval_accuracy))
        
        # Log chi tiết
        if not print_numbers_only:
            for question_type in question_types_count:
                eval_log.append('Type %s: %f' % (question_type, np.mean(question_types_count[question_type])))
            for simple_complex_type in simple_complex_count:
                eval_log.append('%s: %f' % (simple_complex_type, np.mean(simple_complex_count[simple_complex_type])))
            for entity_time_type in entity_time_count:
                eval_log.append('%s: %f' % (entity_time_type, np.mean(entity_time_count[entity_time_type])))
    
    return eval_accuracy_for_reporting, eval_log


def train(qa_model, dataset, valid_dataset, args, result_filename=None):
    """
    Training loop cho QA model
    """
    num_workers = 0  # Disable multiprocessing
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    
    # Tạo DataLoader cho training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset._collate_fn)
    max_eval_score = 0
    
    # Xử lý model saving filename
    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = 'results/{dataset_name}/{model_file}.log'.format(
            dataset_name=args.dataset_name,
            model_file=args.save_to
        )
    
    # Tạo thư mục và file log
    os.makedirs('results/{dataset_name}'.format(dataset_name=args.dataset_name), exist_ok=True)
    os.makedirs('models/models/{dataset_name}/qa_models'.format(dataset_name=args.dataset_name), exist_ok=True)
    checkpoint_file_name = 'models/models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.save_to
    )
    
    # Tạo log file mới
    with open(result_filename, 'w') as f:
        f.write('QUICK TEST LOG FILE\n')
        f.write('Model %s\n' % args.model)
        f.write('Supervision %s\n' % args.supervision)
        f.write('Learning rate %f\n' % args.lr)
        f.write('Batch size %d\n' % args.batch_size)
        f.write('Max epochs %d\n' % args.max_epochs)
        f.write('Train samples %d\n' % len(dataset.data))
        f.write('Test samples %d\n' % len(valid_dataset.data))
        f.write('Fuse %s\n' % args.fuse)
        f.write('Frozen %d\n' % args.frozen)
        f.write('LM frozen %d\n' % args.lm_frozen)
        f.write('Extra entities %d\n' % args.extra_entities)
        f.write('Corrupt hard %f\n' % args.corrupt_hard)
        f.write('Eval split %s\n' % args.eval_split)
        f.write('Eval k %d\n' % args.eval_k)
        f.write('TKG file %s\n' % args.tkg_file)
        f.write('TKBC model file %s\n' % args.tkbc_model_file)
        f.write('Dataset name %s\n' % args.dataset_name)
    
    print(f'🚀 QUICK TEST MODE')
    print(f'📊 Training samples: {len(dataset.data)}')
    print(f'📊 Test samples: {len(valid_dataset.data)}')
    print(f'🔄 Batch size: {batch_size}')
    print(f'📈 Epochs: {args.max_epochs}')
    
    # Training loop chính
    for epoch in range(args.max_epochs):
        qa_model.train()  # Chuyển sang training mode
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        
        # Batch training loop
        for i_batch, a in enumerate(loader):
            # Forward pass
            scores = qa_model.forward(a)
            answers_khot = a[-1]  # Target answers
            
            # Tính loss và backward pass
            loss = qa_model.loss(scores, answers_khot.cuda().long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            loader.update()
        
        print(f'📊 Epoch {epoch+1}/{args.max_epochs} - Loss: {epoch_loss:.4f}')
        
        # Evaluation theo frequency
        if (epoch + 1) % args.valid_freq == 0:
            print('🔍 Starting evaluation...')
            eval_score, eval_log = eval(qa_model, valid_dataset, 
                                      batch_size=args.valid_batch_size, 
                                      split=args.eval_split, k=args.eval_k)
            
            # Lưu model nếu performance tốt hơn
            if eval_score > max_eval_score:
                print('✅ Valid score increased - Saving model!') 
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            else:
                print('❌ Valid score not improved')
            
            # Log kết quả evaluation
            append_log_to_file(eval_log, epoch, result_filename)
    
    print(f'🎉 Training completed!')
    print(f'🏆 Best validation score: {max_eval_score:.4f}')


def save_model(qa_model, filename):
    """
    Lưu model checkpoint
    """
    print('💾 Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('✅ Saved model to ', filename)
    return


def append_log_to_file(eval_log, epoch, result_filename):
    """
    Ghi log kết quả evaluation vào file
    """
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    
    with open(result_filename, 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Epoch %d\n' % epoch)
        for line in eval_log:
            f.write('%s\n' % line)
        f.write('\n')


# ==================== MAIN EXECUTION ====================

# Load TKBC model
if args.model != 'embedkgqa':
    tkbc_model = loadTkbcModel('models/models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))
else:
    tkbc_model = loadTkbcModel_complex('models/models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))

# Test KGE embeddings nếu mode là test_kge
if args.mode == 'test_kge':
    from utils import checkIfTkbcEmbeddingsTrained
    checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

# Xác định train/test splits
train_split = 'train'
test = args.test

# Khởi tạo model và datasets
if args.model == 'bert' or args.model == 'roberta':
    qa_model = QA_lm(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
    
elif args.model == 'embedkgqa':
    qa_model = QA_embedkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
    
elif args.model == 'cronkgqa' and args.supervision != 'hard':
    qa_model = QA_cronkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
    
elif args.model in ['tempoqr', 'entityqr', 'cronkgqa']:
    qa_model = QA_TempoQR(tkbc_model, args)
    if args.mode == 'train':
        dataset = QA_Dataset_TempoQR(split=train_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_TempoQR(split=test, dataset_name=args.dataset_name, args=args)
    
else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)

# QUICK TEST MODE - Tạo subset nhỏ của data
if args.quick_test:
    print('🚀 QUICK TEST MODE - Creating random subsets...')
    
    # Set seed cho reproducibility
    import random
    random.seed(42)
    
    # Tạo training subset ngẫu nhiên
    train_indices = random.sample(range(len(dataset.data)), min(args.train_samples, len(dataset.data)))
    train_data = [dataset.data[i] for i in train_indices]
    
    # Tạo test subset ngẫu nhiên (không trùng với train)
    available_test_indices = [i for i in range(len(test_dataset.data)) if i not in train_indices]
    test_indices = random.sample(available_test_indices, min(args.test_samples, len(available_test_indices)))
    test_data = [test_dataset.data[i] for i in test_indices]
    
    print(f'📊 Random training data: {len(dataset.data)} → {len(train_data)} samples')
    print(f'📊 Random test data: {len(test_dataset.data)} → {len(test_data)} samples')
    print(f'🎲 Train indices: {train_indices[:10]}...' if len(train_indices) > 10 else f'🎲 Train indices: {train_indices}')
    print(f'🎲 Test indices: {test_indices[:10]}...' if len(test_indices) > 10 else f'🎲 Test indices: {test_indices}')
    
    # Tạo new dataset objects với data đã reduce
    # Giữ nguyên reference đến original dataset để có _collate_fn
    class QuickTestDataset:
        def __init__(self, original_dataset, data_subset):
            self.original_dataset = original_dataset
            self.data = data_subset
            # Copy tất cả methods từ original dataset
            self.__dict__.update(original_dataset.__dict__)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # Debug: Print data structure
            print(f"Debug: self.data[{idx}] = {type(self.data[idx])}")
            if hasattr(self.data[idx], 'keys'):
                print(f"Debug: Keys = {self.data[idx].keys()}")
            
            # self.data[idx] là dictionary đã có 'original_index'
            original_idx = self.data[idx]['original_index']
            return self.original_dataset.__getitem__(original_idx)
        
        def _collate_fn(self, items):
            return self.original_dataset._collate_fn(items)
    
    # Tạo mapping từ subset index → original index
    train_index_map = {i: train_indices[i] for i in range(len(train_indices))}
    test_index_map = {i: test_indices[i] for i in range(len(test_indices))}
    
    # Tạo quick test datasets với mapping
    class QuickTestDataset:
        def __init__(self, original_dataset, index_map, data_subset):
            self.original_dataset = original_dataset
            self.index_map = index_map
            self.data = data_subset
            # Copy tất cả methods từ original dataset
            self.__dict__.update(original_dataset.__dict__)
        
        def __len__(self):
            return len(self.index_map)  # Số samples thực tế
        
        def __getitem__(self, idx):
            # idx là subset index, map về original index
            original_idx = self.index_map[idx]
            return self.original_dataset.__getitem__(original_idx)
        
        def _collate_fn(self, items):
            return self.original_dataset._collate_fn(items)
    
    # Tạo datasets với mapping
    dataset = QuickTestDataset(dataset, train_index_map, train_data)
    test_dataset = QuickTestDataset(test_dataset, test_index_map, test_data)

# Load model từ checkpoint nếu có
if args.load_from != '':
    filename = 'models/models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.load_from
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
else:
    print('Not loading from checkpoint. Starting fresh!')

# Chuyển model sang GPU
qa_model = qa_model.cuda()

# Evaluation mode nếu chỉ muốn đánh giá
if args.mode == 'eval':
    score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, 
                      split=args.eval_split, k=args.eval_k)
    exit(0)

# Tạo result filename
result_filename = 'results/{dataset_name}/{model_file}_quick_test.log'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)

# Bắt đầu training
train(qa_model, dataset, test_dataset, args, result_filename=result_filename)

print('🎉 Quick test training finished!')
print('📁 Check results at:', result_filename)
