#dataset.py
import os
import re
import pickle as pk
import esm
import esm.sdk
from esm.sdk import esmc_client
import torch
import warnings
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
warnings.simplefilter('ignore')


def prepare_dataset_artifacts(dataset_name, root, device='cpu', model=None):
    """Chuẩn bị các file artifacts cho dataset"""
    os.makedirs(root, exist_ok=True)

    if dataset_name == 'BCE_633':
        csv_path = os.path.join(root, 'total.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'BCE_633 dataset file not found: {csv_path}')

        if not os.path.exists(os.path.join(root, 'total.pkl')):
            initial('total.csv', root, model=model, device=device, from_native_pdb=True)

        if not os.path.exists(os.path.join(root, 'train.pkl')) or not os.path.exists(os.path.join(root, 'test.pkl')):
            with open(os.path.join(root, 'total.pkl'), 'rb') as f:
                samples = pk.load(f)
            with open(os.path.join(root, 'train.pkl'), 'wb') as f:
                pk.dump(samples, f)
            with open(os.path.join(root, 'test.pkl'), 'wb') as f:
                pk.dump(samples, f)

        if not os.path.exists(os.path.join(root, 'cross-validation.npy')):
            with open(os.path.join(root, 'total.pkl'), 'rb') as f:
                samples = pk.load(f)
            np.save(os.path.join(root, 'cross-validation.npy'), np.arange(len(samples)))
        return

    if dataset_name == 'Epitope3D':
        if os.path.exists(os.path.join(root, 'train.pkl')) and os.path.exists(os.path.join(root, 'test.pkl')):
            return
        raise FileNotFoundError(
            'Epitope3D preprocessing files are missing. Run dataset.py first to generate train.pkl/test.pkl.'
        )

    raise ValueError(f'Unsupported dataset: {dataset_name}')


def check_protein_files(protein, root, use_saprot=True):
    """
    Kiểm tra xem protein đã có đủ các file features chưa
    
    Args:
        protein: Protein object
        root: Root directory
        use_saprot: Nếu True thì kiểm tra cả saprot files
    
    Returns:
        dict: {
            'feat': bool,
            'saprot': bool,
            'adj': bool,
            'edge': bool,
            'all_exist': bool
        }
    """
    name = protein.name
    
    # Đường dẫn các file
    files = {
        'feat': f'{root}/feat/{name}.pt',
        'adj': f'{root}/graph/{name}_adj.pt',
        'edge': f'{root}/graph/{name}_edge.pt',
    }
    
    if use_saprot:
        files['saprot'] = f'{root}/saprot/{name}.pt'
    
    # Kiểm tra từng file
    status = {}
    for key, path in files.items():
        status[key] = os.path.exists(path)
    
    status['all_exist'] = all(status.values())
    return status


def load_protein_from_cache(protein, root, use_saprot=True):
    """Load protein features từ cache"""
    name = protein.name
    
    protein.feat = torch.load(f'{root}/feat/{name}.pt')
    protein.adj = torch.load(f'{root}/graph/{name}_adj.pt')
    protein.edge = torch.load(f'{root}/graph/{name}_edge.pt')
    
    if use_saprot:
        protein.saprot = torch.load(f'{root}/saprot/{name}.pt')
    
    return protein


def save_protein_to_cache(protein, root, use_saprot=True):
    """Lưu protein features vào cache"""
    name = protein.name
    
    # Tạo thư mục nếu chưa có
    os.makedirs(f'{root}/feat', exist_ok=True)
    os.makedirs(f'{root}/graph', exist_ok=True)
    if use_saprot:
        os.makedirs(f'{root}/saprot', exist_ok=True)
    
    # Lưu files
    torch.save(protein.feat, f'{root}/feat/{name}.pt')
    torch.save(protein.adj, f'{root}/graph/{name}_adj.pt')
    if hasattr(protein, 'edge') and protein.edge is not None:
        torch.save(protein.edge, f'{root}/graph/{name}_edge.pt')
    if use_saprot and hasattr(protein, 'saprot'):
        torch.save(protein.saprot, f'{root}/saprot/{name}.pt')


def parse_date(date_str):
    """
    Parse date string thành (d, m, y)
    Hỗ trợ các định dạng:
    - 'DD-MON-YY' hoặc 'DD-MON-YYYY' (ví dụ: '11-FEB-21')
    - 'YYYY-MM-DD' (ví dụ: '2021-02-11')
    - 'DD/MM/YYYY' (ví dụ: '11/02/2021')
    """
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    raw = str(date_str).strip()
    parts = re.split(r'[-/\s]+', raw)
    
    try:
        if len(parts) >= 3 and parts[1].isalpha():
            # Định dạng: 'DD-MON-YY' hoặc 'DD-MON-YYYY'
            d = int(parts[0])
            m = month_map[parts[1].upper()[:3]]
            y_str = parts[2]
            if len(y_str) == 4:
                y = int(y_str)
            else:
                y2 = int(y_str)
                y = 2000 + y2 if y2 < 23 else 1900 + y2
                
        elif len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
            # Định dạng: 'YYYY-MM-DD'
            y = int(parts[0])
            m = int(parts[1])
            d = int(parts[2])
            
        elif len(parts) >= 3 and parts[0].isdigit() and parts[0].isdigit():
            # Định dạng: 'DD/MM/YYYY'
            d = int(parts[0])
            m = int(parts[1])
            y = int(parts[2])
            
        else:
            print(f"[WARN] Unrecognized date format: '{raw}'")
            return None
            
        return (d, m, y)
        
    except Exception as e:
        print(f"[WARN] Failed to parse date '{raw}': {e}")
        return None


def check_existing_features(root, dataset, use_saprot=True):
    """
    Kiểm tra toàn bộ dataset, đếm số protein đã có features
    
    Returns:
        dict: Thống kê số lượng file
    """
    stats = {
        'total': len(dataset),
        'has_feat': 0,
        'has_saprot': 0,
        'has_adj': 0,
        'has_edge': 0,
        'all_complete': 0
    }
    
    for protein in dataset:
        name = protein.name
        if os.path.exists(f'{root}/feat/{name}.pt'):
            stats['has_feat'] += 1
        if use_saprot and os.path.exists(f'{root}/saprot/{name}.pt'):
            stats['has_saprot'] += 1
        if os.path.exists(f'{root}/graph/{name}_adj.pt'):
            stats['has_adj'] += 1
        if os.path.exists(f'{root}/graph/{name}_edge.pt'):
            stats['has_edge'] += 1
            
        # Kiểm tra đầy đủ
        all_exist = True
        if not os.path.exists(f'{root}/feat/{name}.pt'):
            all_exist = False
        if use_saprot and not os.path.exists(f'{root}/saprot/{name}.pt'):
            all_exist = False
        if not os.path.exists(f'{root}/graph/{name}_adj.pt'):
            all_exist = False
        if not os.path.exists(f'{root}/graph/{name}_edge.pt'):
            all_exist = False
            
        if all_exist:
            stats['all_complete'] += 1
    
    return stats


class PDB(Dataset):
    def __init__(
        self, mode='train', fold=-1, root='./data/Epitope3D', 
        self_cycle=False, use_cv=True, use_saprot=True
    ):
        self.root = root
        self.use_cv = use_cv
        self.use_saprot = use_saprot
        assert mode in ['train', 'val', 'test', 'all']
        
        # Load samples
        if mode == 'all':
            with open(f'{self.root}/total.pkl', 'rb') as f:
                self.samples = pk.load(f)
        elif mode in ['train', 'val']:
            with open(f'{self.root}/train.pkl', 'rb') as f:
                self.samples = pk.load(f)
        else:  # test
            with open(f'{self.root}/test.pkl', 'rb') as f:
                self.samples = pk.load(f)
        
        self.data = []
        
        # Xác định order
        if mode == 'all':
            order = list(range(len(self.samples)))
        else:
            if self.use_cv:
                idx = np.load(f'{self.root}/cross-validation.npy')
                cv = 10
                inter = len(idx) // cv
                ex = len(idx) % cv
                
                if mode == 'train':
                    order = []
                    for i in range(cv):
                        if i == fold:
                            continue
                        order += list(idx[i*inter:(i+1)*inter + ex*(i == cv-1)])
                elif mode == 'val':
                    order = list(idx[fold*inter:(fold+1)*inter + ex*(fold == cv-1)])
                else:  # test
                    order = list(range(len(self.samples)))
            else:
                order = list(range(len(self.samples)))
        
        order.sort()
        
        # Load features với kiểm tra cache
        tbar = tqdm(order, desc=f'Loading {mode} set')
        cached_count = 0
        processed_count = 0
        
        for i in tbar:
            protein = self.samples[i]
            name = protein.name
            
            # Kiểm tra file tồn tại
            file_status = check_protein_files(protein, self.root, self.use_saprot)
            
            if file_status['all_exist']:
                # Đã có file, load từ cache
                load_protein_from_cache(protein, self.root, self.use_saprot)
                cached_count += 1
                tbar.set_postfix(chain=f'{name} ✓')
            else:
                # Chưa có file, tính toán và lưu
                protein.load_feat(self.root)
                if self.use_saprot:
                    protein.load_saprot(self.root)
                protein.load_adj(self.root, self_cycle)
                
                # Lưu vào cache
                save_protein_to_cache(protein, self.root, self.use_saprot)
                processed_count += 1
                tbar.set_postfix(chain=f'{name} ⚡')
            
            self.data.append(protein)
        
        print(f"[INFO] Loaded {len(self.data)} proteins: {cached_count} from cache, {processed_count} newly processed")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # Fix size mismatch
        f = seq.feat
        d = seq.saprot if self.use_saprot else None
        
        # Cắt BOS/EOS của ESM-C (646 -> 644)
        if d is not None and f.shape[0] == d.shape[0] + 2:
            f = f[1:-1, :]
        
        # Đảm bảo độ dài khớp
        if d is not None:
            min_len = min(f.shape[0], d.shape[0])
            f = f[:min_len, :]
            d = d[:min_len, :]
            feat = torch.cat([f, d], dim=1)
        else:
            feat = f
            min_len = f.shape[0]
        
        return {
            'feat': feat,
            'label': seq.label[:min_len],
            'adj': seq.adj,
            'edge': seq.edge if hasattr(seq, 'edge') else None,
        }


def process_bce633_dataset(root, model, device):
    """Xử lý dataset BCE_633"""
    print("[INFO] Processing BCE_633...")
    
    # Kiểm tra xem đã có total.pkl chưa
    total_pkl = f'{root}/total.pkl'
    if os.path.exists(total_pkl):
        print(f"[INFO] Found existing {total_pkl}, loading...")
        with open(total_pkl, 'rb') as f:
            dataset = pk.load(f)
    else:
        # Chưa có, xử lý từ CSV
        initial("total.csv", root, model=model, device=device, from_native_pdb=True)
        with open(total_pkl, 'rb') as f:
            dataset = pk.load(f)
    
    # Kiểm tra features đã tồn tại
    print("[INFO] Checking existing features...")
    stats = check_existing_features(root, dataset, use_saprot=True)
    print(f"[INFO] Feature stats: {stats}")
    
    # Lọc dữ liệu
    filt_data = []
    for i in dataset:
        if len(i) < 1024 and i.label.sum() > 0:
            filt_data.append(i)
    
    print(f"[INFO] Filtered {len(filt_data)} proteins (length < 1024, has epitope)")
    
    # Chia train/test theo ngày tháng
    TEST_CUTOFF = 20210401
    trainset = []
    testset = []
    dates_for_cv = []
    
    for protein in filt_data:
        date_info = parse_date(str(protein.date))
        if date_info is None:
            print(f"[WARN] Skip {protein.name}: invalid date")
            continue
        
        d, m, y = date_info
        date_int = y * 10000 + m * 100 + d
        
        if date_int < TEST_CUTOFF:
            dates_for_cv.append(date_int)
            trainset.append(protein)
        else:
            testset.append(protein)
    
    # Lưu kết quả
    with open(f'{root}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl', 'wb') as f:
        pk.dump(testset, f)
    
    # Tạo cross-validation indices
    idx = np.array(dates_for_cv).argsort()
    np.save(f'{root}/cross-validation.npy', idx)
    
    print(f"[INFO] BCE_633 Done. Train: {len(trainset)}, Test: {len(testset)}, CV idx shape: {idx.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='./data/Epitope3D',
        help='dataset path'
    )
    parser.add_argument(
        "--dataset",
        default="Epitope3D",
        choices=["Epitope3D", "BCE_633"]
    )
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument('--train_csv', type=str, default='epitope3d_dataset_200_Train.csv',
                        help='Tên file CSV train cho tập epitope3d')
    parser.add_argument('--test_csv', type=str, default='epitope3d_dataset_45_Blind_Test.csv',
                        help='Tên file CSV test cho tập epitope3d')
    parser.add_argument('--use_saprot', action='store_true', default=True,
                        help='Sử dụng Saprot features')
    args = parser.parse_args()
    
    root = args.root
    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    
    # Tạo thư mục
    for d in [root, f'{root}/PDB', f'{root}/purePDB', f'{root}/feat', 
              f'{root}/saprot', f'{root}/graph']:
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] Prepared folders under {root}")
    
    # Khởi tạo ESM-C model
    token = '5zPJa56XnPf91N4L8yWdMQ'
    print("[INFO] Đang kết nối tới Forge API cho ESM-C 6B...")
    model = esmc_client(
        model="esmc-6b-2024-12",
        url="https://biohub.ai",
        token=token
    )
    print("Model connected successfully!")
    
    if args.dataset == "BCE_633":
        # Xử lý BCE_633
        process_bce633_dataset(root, model, device)
        
    elif args.dataset == "Epitope3D":
        # Xử lý Epitope3D
        print("[INFO] Đang xử lý tập dữ liệu Epitope3D (Đã chia sẵn Train/Test)...")
        
        # Tạo thư mục
        for d in [f'{root}/PDB', f'{root}/purePDB', f'{root}/feat', 
                  f'{root}/saprot', f'{root}/graph']:
            os.makedirs(d, exist_ok=True)
        
        # Kiểm tra xem đã có features chưa
        print(f"--> Xử lý tập Train: {args.train_csv}")
        trainset = initial_epitope3D(args.train_csv, root, model, device)
        
        print(f"--> Xử lý tập Test: {args.test_csv}")
        testset = initial_epitope3D(args.test_csv, root, model, device)
        
        # Lọc dữ liệu
        trainset = [i for i in trainset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]
        testset = [i for i in testset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]
        
        # Kiểm tra features đã tồn tại
        print("[INFO] Checking existing features...")
        train_stats = check_existing_features(root, trainset, use_saprot=True)
        test_stats = check_existing_features(root, testset, use_saprot=True)
        print(f"[INFO] Train stats: {train_stats}")
        print(f"[INFO] Test stats: {test_stats}")
        
        # Lưu kết quả
        np.random.seed(42)
        idx = np.random.permutation(len(trainset))
        with open(f'{root}/train.pkl', 'wb') as f:
            pk.dump(trainset, f)
        with open(f'{root}/test.pkl', 'wb') as f:
            pk.dump(testset, f)
        
        print(f"[INFO] TỔNG KẾT -> Train: {len(trainset)} chains, Test: {len(testset)} chains")