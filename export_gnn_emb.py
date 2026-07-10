# export_gnn_emb.py
import os
import argparse
import torch
import numpy as np
from model import GraphBepi
from dataset import PDB
from torch.utils.data import DataLoader


def load_model(ckpt, device='cpu'):
    m = GraphBepi()
    state = torch.load(ckpt, map_location=device)
    if 'state_dict' in state:
        m.load_state_dict(state['state_dict'])
    else:
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def export_embeddings(ckpt, root, out_dir='./tabular', split='train', batch=8, gpu=-1, gnn_only=False, limit=None):
    device = 'cpu' if gpu == -1 or not torch.cuda.is_available() else f'cuda:{gpu}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Loading model {ckpt} on {device}")
    model = load_model(ckpt, device=device)

    dataset = PDB(mode=split, fold=-1, root=root)

    names_out = []
    idxs_out = []
    resn_out = []
    emb_out = []

    # batch by indices from dataset.data to keep names/order
    count = 0
    for start in range(0, len(dataset), batch):
        end = min(start + batch, len(dataset))
        batch_samples = dataset.data[start:end]
        V_list = []
        E_list = []
        A_list = []
        valid_samples = []

        for s in batch_samples:
            try:
                s.load_feat(root)
                s.load_saprot(root)
                s.load_adj(root)
            except Exception as e:
                print(f"[WARN] Skip {s.name} due to feature/graph error: {repr(e)}")
                continue

            # sanity check lengths
            if s.feat is None or s.saprot is None:
                print(f"[WARN] Skip {s.name}: missing feat/saprot")
                continue

            feat_len = s.feat.shape[0]
            saprot_len = s.saprot.shape[0]

            if feat_len != saprot_len:
                if feat_len == saprot_len + 2:
                    print(f"[FIXED] {s.name}: Cắt 2 token dư (feat {feat_len} -> {saprot_len}) để khớp với saprot.")
                    s.feat = s.feat[1:-1]
                elif saprot_len == feat_len + 2:
                    print(f"[FIXED] {s.name}: Cắt 2 token dư (saprot {saprot_len} -> {feat_len}) để khớp với feat.")
                    s.saprot = s.saprot[1:-1]
                else:
                    min_len = min(feat_len, saprot_len)
                    print(f"[WARN] Skip {s.name}: feat len {feat_len} != saprot len {saprot_len}; using min_len={min_len}")
                    s.feat = s.feat[:min_len]
                    s.saprot = s.saprot[:min_len]

            L = s.feat.shape[0]
            if len(s.sequence) < L:
                print(f"[WARN] Skip {s.name}: sequence len {len(s.sequence)} < feat len {L}")
                continue

            V_list.append(torch.cat([s.feat, s.saprot], 1))
            E_list.append(s.edge)
            A_list.append(s.adj)
            valid_samples.append(s)

        # nếu batch này không còn sample hợp lệ thì skip batch
        if len(valid_samples) == 0:
            continue

        # move tensors to device
        V_list = [v.to(device) for v in V_list]
        E_list = [e.to(device) for e in E_list]
        A_list = [a.to(device) for a in A_list]

        with torch.no_grad():
            h_list = model.embed_stage1(
                V_list,
                E_list,
                A_list
            )

        # zip theo valid_samples (không phải batch_samples nữa)
        for s, h in zip(valid_samples, h_list):
            h_cpu = h.cpu().numpy()
            L = h_cpu.shape[0]
            for i in range(L):
                names_out.append(s.name)
                idxs_out.append(i)
                resn_out.append(s.sequence[i])
                emb_out.append(h_cpu[i])
                count += 1
                if limit is not None and count >= limit:
                    break
            if limit is not None and count >= limit:
                break

    emb_arr = np.vstack(emb_out).astype(np.float32)
    names_arr = np.array(names_out, dtype=object)
    idxs_arr = np.array(idxs_out, dtype=np.int32)
    resn_arr = np.array(resn_out, dtype=object)

    suffix = '_stage1' if gnn_only else ''
    out_path = os.path.join(out_dir, f'gnn_{split}{suffix}.npz')
    np.savez_compressed(out_path, emb=emb_arr, names=names_arr, idxs=idxs_arr, resn=resn_arr)
    print(f"[DONE] Exported GNN embeddings to {out_path} (shape {emb_arr.shape})")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--root', type=str, default='/kaggle/input/dataset/data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='train', choices=['train','val','test', 'all'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gnn-only', action='store_true', help='Export embeddings from GNN only (skip LSTM)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of residues to export (for smoke tests)')
    args = parser.parse_args()
    export_embeddings(args.ckpt, args.root, args.out, args.split, args.batch, args.gpu, gnn_only=args.gnn_only, limit=args.limit)