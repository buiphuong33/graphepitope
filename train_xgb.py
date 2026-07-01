import os
import torch
import joblib
import argparse
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from dataset import PDB, collate_fn
from model import GraphBepi
from torch.utils.data import DataLoader

def get_all_embeddings(model, dataset, device, batch_size=4):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_features = []
    all_labels = []
    all_names = []
    
    print(f"[INFO] Đang trích xuất embeddings từ model...")
    with torch.no_grad():
        for batch in loader:
            # batch chứa [feats, edges, labels]
            V_list, E_list, A_list, y_list = batch
            
            # Đẩy vào hàm embed đã viết trong model.py
            h_list = model.embed(V_list, E_list, A_list)
            
            for h, y, s in zip(h_list, y_list, batch): # batch ở đây cần refactor nhẹ để lấy name
                all_features.append(h.cpu().numpy())
                all_labels.append(y.numpy())
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    return X, y

def main(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load model PyTorch đã train xong
    print(f"[INFO] Load model từ: {args.ckpt}")
    model = GraphBepi(feat_dim=2560, hidden_dim=256, exfeat_dim=1280, edge_dim=51)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
    model.to(device)

    # 2. Hút đặc trưng từ dataset
    trainset = PDB(mode='train', root=args.root)
    X_train, y_train = get_all_embeddings(model, trainset, device)
    
    # 3. PCA giảm chiều (SaProt 2560 chiều hơi lớn, nén còn 32/64 là đẹp)
    print(f"[INFO] PCA nén dữ liệu xuống {args.pca_dim} chiều...")
    pca = PCA(n_components=args.pca_dim)
    X_train_reduced = pca.fit_transform(X_train)
    
    # 4. Train XGBoost
    print(f"[INFO] Training XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist', # Dùng GPU nếu có: 'gpu_hist'
        n_jobs=-1
    )
    clf.fit(X_train_reduced, y_train)
    
    # 5. Lưu kết quả
    os.makedirs('./model/xgb', exist_ok=True)
    joblib.dump(clf, './model/xgb/xgb_model.joblib')
    joblib.dump(pca, './model/xgb/pca_model.joblib')
    print("[DONE] Đã lưu model XGBoost!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--root', type=str, default='./data/Epitope3D')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pca_dim', type=int, default=32)
    args = parser.parse_args()
    main(args)