# merge_tabular_and_gnn.py
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import joblib


def merge(tabular_dir='./tabular', split='train', pca_dim=10, out_dir=None, pca_model_path=None):
    if out_dir is None:
        out_dir = tabular_dir
    tab_path = os.path.join(tabular_dir, f'{split}.npz')
    # prefer gnn-only variant if present
    gnn_candidates = [os.path.join(tabular_dir, f'gnn_{split}_stage1.npz'), os.path.join(tabular_dir, f'gnn_{split}.npz')]
    gnn_path = None
    for candidate in gnn_candidates:
        if os.path.exists(candidate):
            gnn_path = candidate
            break
    if not os.path.exists(tab_path):
        raise FileNotFoundError(f"Tabular file {tab_path} not found. Run export_tabular first.")
    if gnn_path is None:
        raise FileNotFoundError(f"GNN embeddings not found. Run export_gnn_emb first (looked for: {gnn_candidates})")

    tab = np.load(tab_path, allow_pickle=True)
    gnn = np.load(gnn_path, allow_pickle=True)
    print(f"[INFO] Merging using GNN file: {gnn_path}")

    X = tab['X']
    y = tab['y'] if 'y' in tab else None
    names = tab['names']
    idxs = tab['idxs']
    resn = tab['resn']

    emb = gnn['emb']
    emb_names = gnn['names']
    emb_idxs = gnn['idxs']

    # build mapping from (name, idx) -> emb row index
    emb_map = {(n, int(i)): j for j, (n, i) in enumerate(zip(emb_names, emb_idxs))}

    merged_emb = []
    missing = 0
    for n,i in zip(names, idxs):
        key = (n, int(i))
        j = emb_map.get(key, None)
        if j is None:
            missing += 1
            merged_emb.append(np.zeros(emb.shape[1], dtype=np.float32))
        else:
            merged_emb.append(emb[j])
    merged_emb = np.vstack(merged_emb)

    # PCA
    pca_out_path = os.path.join(out_dir, 'pca_{}_{}.joblib'.format(split, pca_dim))
    if pca_dim is not None and pca_dim > 0:

        # Với BCE_633 (split=all) hoặc train thì fit PCA luôn
        if split in ['train', 'all'] and pca_model_path is None:

            print(f"[INFO] Fitting PCA ({pca_dim} dims)...")

            pca = PCA(n_components=pca_dim, random_state=42)
            emb_reduced = pca.fit_transform(merged_emb)

            joblib.dump(pca, pca_out_path)
            print(f"[DONE] PCA saved to {pca_out_path}")

            pca_model_used = pca_out_path

        else:
            # test/val dùng PCA đã fit
            if pca_model_path is None:
                guess = os.path.join(out_dir, f"pca_train_{pca_dim}.joblib")

                if not os.path.exists(guess):
                    guess = os.path.join(out_dir, f"pca_all_{pca_dim}.joblib")

                if os.path.exists(guess):
                    pca_model_path = guess

            if pca_model_path is None or not os.path.exists(pca_model_path):
                raise FileNotFoundError("Cannot find PCA model.")

            pca = joblib.load(pca_model_path)
            emb_reduced = pca.transform(merged_emb)
            pca_model_used = pca_model_path
    else:
        emb_reduced = merged_emb
        pca_model_used = None

    X_merged = np.concatenate([X.astype(np.float32), emb_reduced.astype(np.float32)], axis=1)
    out_path = os.path.join(out_dir, f'{split}_merged.npz')
    np.savez_compressed(out_path, X=X_merged, y=y, names=names, idxs=idxs, resn=resn)
    print(f"[DONE] Wrote merged features to {out_path} (X shape {X_merged.shape}). Missing emb rows: {missing}")
    return out_path, pca_model_used


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabular', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='train', choices=['train','test', 'all'])
    parser.add_argument('--pca-dim', type=int, default=10)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--pca-model', type=str, default=None)
    args = parser.parse_args()
    merge(args.tabular, args.split, args.pca_dim, args.out, args.pca_model)