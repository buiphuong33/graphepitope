#utils.py
import os
import esm.sdk
from esm.sdk import client
import torch
import numpy as np
import pandas as pd
import pickle as pk
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,trange
from preprocess import *
from graph_construction import calcPROgraph
import requests as rq
from esm.sdk.api import ESMProtein, LogitsConfig
EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, 
    return_embeddings=True 
)
# prot_amino2id={
#     '<pad>': 0, '</s>': 1, '<unk>': 2, 'A': 3,
#     'L': 4, 'G': 5, 'V': 6, 'S': 7,
#     'R': 8, 'E': 9, 'D': 10, 'T': 11,
#     'I': 12, 'P': 13, 'K': 14, 'F': 15,
#     'Q': 16, 'N': 17, 'Y': 18, 'M': 19,
#     'H': 20, 'W': 21, 'C': 22, 'X': 23,
#     'B': 24, 'O': 25, 'U': 26, 'Z': 27
# }
amino2id={
    '<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
    'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 
    'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, 
    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32, '<cath>': 33, '<af2>': 34
}
class chain:
    def __init__(self):
        self.sequence=[]
        self.amino=[]
        self.coord=[]
        self.site={}
        self.date=''
        self.length=0
        self.adj=None
        self.edge=None
        self.feat=None
        self.saprot=None
        self.name=''
        self.chain_name=''
        self.protein_name=''
    def add(self,amino,pos,coord):
        self.sequence.append(DICT[amino])
        self.amino.append(amino2id[DICT[amino]])
        self.coord.append(coord)
        self.site[pos]=self.length
        self.length+=1
    def process(self):
        self.amino=torch.LongTensor(self.amino)
        self.coord=torch.FloatTensor(self.coord)
        self.label=torch.zeros_like(self.amino)
        self.sequence=''.join(self.sequence)
    def extract(self, model, device, path):
        # 1. Kiểm tra điều kiện cơ bản
        if len(self.sequence) > 1024:
            return

        target_file = f'{path}/feat/{self.name}_esmc6b.ts'
        if os.path.exists(target_file):
            return

        # Đảm bảo thư mục tồn tại
        os.makedirs(f'{path}/feat/', exist_ok=True)

        # Two modes supported:
        #  - remote Forge client (has `encode` and `logits`) -- original flow
        #  - local ESM model (torch) as fallback
        try:
            if model is not None and hasattr(model, 'encode') and hasattr(model, 'logits'):
                # remote client flow
                with torch.no_grad():
                    protein = ESMProtein(sequence=self.sequence)
                    protein_tensor = model.encode(protein)
                    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
                    feat = output.embeddings.cpu().squeeze(0)
                    torch.save(feat, target_file)
                    return

            # fallback: use local esm model to produce embeddings
            # Prefer local ESMC models (if installed) before falling back to ESM-2
            esm_model = None
            alphabet = None
            # Try multiple candidate constructors from esm.pretrained
            pref_names = [
                'ESMC_6B_202412',
                'ESMC_600M_202412',
                'ESMC_300M_202412',
                'esmc_6b',
                'esmc_600m',
                'esmc_300m',
            ]
            for name in pref_names:
                ctor = getattr(esm.pretrained, name, None)
                if ctor is None:
                    continue
                try:
                    esm_model, alphabet = ctor(device=device)
                    break
                except Exception:
                    try:
                        esm_model = ctor()
                        # some ctors return (model, alphabet)
                        if isinstance(esm_model, tuple) and len(esm_model) == 2:
                            esm_model, alphabet = esm_model
                        break
                    except Exception:
                        esm_model = None
                        alphabet = None
                        continue

            # final fallbacks to esm2 if no esmc available
            if esm_model is None:
                try:
                    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                except Exception:
                    esm_model, alphabet = esm.pretrained.esm2_t36_3B()

            batch_converter = alphabet.get_batch_converter()
            esm_model = esm_model.eval().to(device)
            _, _, tokens = batch_converter([(self.name, self.sequence)])
            with torch.no_grad():
                # Different model families may return different dict shapes; handle common cases
                out = esm_model(tokens.to(device))
                if isinstance(out, dict) and 'representations' in out:
                    rep_layer = list(out['representations'].keys())[-1]
                    feat = out['representations'][rep_layer][0, 1:len(self.sequence)+1].cpu()
                elif hasattr(out, 'representations'):
                    # some APIs return an object with `.representations`
                    rep = out.representations
                    if isinstance(rep, dict):
                        rep_layer = list(rep.keys())[-1]
                        feat = rep[rep_layer][0, 1:len(self.sequence)+1].cpu()
                    else:
                        feat = rep[0][0, 1:len(self.sequence)+1].cpu()
                else:
                    # fallback: try the first returned tensor-like
                    try:
                        # assume out["representations"][0] or out[0]
                        feat = out[0][0, 1:len(self.sequence)+1].cpu()
                    except Exception:
                        raise RuntimeError('Unknown model output format from local ESM/ESMC model')

                torch.save(feat, target_file)
                return

        except Exception as e:
            print(f"❌ Lỗi trích xuất embedding cho {self.name}: {e}")

    def load_saprot(self, path):
        # Load embedding SaProt đã được lưu sẵn (file .pt hoặc .npy)
        self.saprot = torch.load(f'{path}/saprot/{self.name}.pt')
    def load_feat(self,path):
        self.feat = torch.load(f'{path}/feat/{self.name}_esmc6b.ts')
    def load_adj(self,path,self_cycle=False):
        graph=torch.load(f'{path}/graph/{self.name}.graph')
        self.adj=graph['adj'].to_dense()
        self.edge=graph['edge'].to_dense()
        if not self_cycle:
            self.adj[range(len(self)),range(len(self))]=0
            self.edge[range(len(self)),range(len(self))]=0
    def get_adj(self,path,dseq=3,dr=10,dlong=5,k=10):
        graph=calcPROgraph(self.sequence,self.coord,dseq,dr,dlong,k)
        os.makedirs(f'{path}/graph', exist_ok=True)
        torch.save(graph,f'{path}/graph/{self.name}.graph')
    def update(self,pos,amino):
        if amino not in DICT.keys():
            return
        amino_id=amino2id[DICT[amino]]
        idx=self.site.get(pos,None)
        if idx is None:
            for i in self.site.keys():
                # print(i,pos)
                if i[:len(pos)]==pos:
                    idx=self.site.get(i)
                    if amino_id==self.amino[idx]:
                        self.label[idx]=1
                        return
        elif amino_id!=self.amino[idx]:
            for i in self.site.keys():
                if i[:len(pos)]==pos:
                    idx=self.site.get(i)
                    if amino_id==self.amino[idx]:
                        self.label[idx]=1
                        return
        else:
            self.label[idx]=1
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        f = self.feat
        d = self.saprot
        
        if f.shape[0] == d.shape[0] + 2:
            f = f[1:-1, :]
            
        min_len = min(f.shape[0], d.shape[0])
        f = f[:min_len, :]
        d = d[:min_len, :]
        
        try:
            full_feat = torch.cat([f, d], dim=1)
        except RuntimeError:
            print(f"Error at {self.name}: Feat {f.shape} != SAPROT {d.shape}")
            raise

        target_label = self.label[:min_len]
        
        return full_feat, self.adj, target_label
def collate_fn(batch):
    edges = [item['edge'] for item in batch]
    feats = [item['feat'] for item in batch]
    adjs = [item['adj'] for item in batch]
    labels = torch.cat([item['label'] for item in batch],0)
    return feats,edges, adjs, labels

def extract_chain(root,pid,chain,force=False):
    # ensure necessary directories exist to avoid FileNotFoundError on write
    os.makedirs(f'{root}/purePDB', exist_ok=True)
    os.makedirs(f'{root}/PDB', exist_ok=True)

    if not force and os.path.exists(f'{root}/purePDB/{pid}_{chain}.pdb'):
        return True
    if not os.path.exists(f'{root}/PDB/{pid}.pdb'):
        retry=5
        pdb=None
        while retry>0:
            try:
                with rq.get(f'https://files.rcsb.org/download/{pid}.pdb') as f:
                    if f.status_code==200:
                        pdb=f.content
                        break
            except:
                retry-=1
                continue
        if pdb is None:
            print(f'PDB file {pid} failed to download')
            return False
        with open(f'{root}/PDB/{pid}.pdb','wb') as f:
            f.write(pdb)
    lines=[]
    with open(f'{root}/PDB/{pid}.pdb','r') as f:
        for line in f:
            if line[:6]=='HEADER':
                lines.append(line)
            if line[:6].strip()=='TER' and line[21]==chain:
                lines.append(line)
                break
            feats=judge(line,None)
            if feats is not None and feats[1]==chain:
                lines.append(line)
    with open(f'{root}/purePDB/{pid}_{chain}.pdb','w') as f:
        for i in lines:
            f.write(i)
    return True
def process_chain(data,root,pid,model,device):
    
    same={}
    pdb_path = f'{root}/purePDB/{pid}.pdb' 
    
    if not os.path.exists(pdb_path):
        print(f"❌ Không tìm thấy file: {pdb_path}")
        return data
    with open(f'{root}/purePDB/{pid}.pdb','r') as f:
        for line in f:
            if line[:6]=='HEADER':
                date=line[50:59].strip()
                data.date=date
                continue
            feats=judge(line,'CA')
            if feats is None:
                continue
            amino,_,site,x,y,z=feats
            if len(amino)>3:
                if same.get(site) is None:
                    same[site]=amino[0]
                if same[site]!=amino[0]:
                    continue
                amino=amino[-3:]
            data.add(amino,site,[x,y,z])
    data.process()
    get_saprot(data.name, data.sequence, root, device)
    data.get_adj(root)
    data.extract(model,device,root)
    return data
def initial(file,root,model=None,device='cpu',from_native_pdb=True):
    df=pd.read_csv(f'{root}/{file}',header=0,index_col=0)
    prefix=df.index
    labels=df['Epitopes (resi_resn)']
    samples=[]
    with tqdm(prefix) as tbar:
        for i in tbar:
            tbar.set_postfix(protein=i)
            if from_native_pdb:
                state=extract_chain(root,i[:4],i[-1])
                if not state:
                    continue
            data=chain()
            p,c=i.split('_')
            data.protein_name=p
            data.chain_name=c
            data.name=f"{p}_{c}"
            process_chain(data,root,i,model,device)
            label=labels.loc[i].split(', ')
            for j in label:
                site,amino=j.split('_')
                data.update(site,amino)
            samples.append(data)
    with open(f'{root}/total.pkl','wb') as f:
        pk.dump(samples,f)

def initial_epitope3D(file, root, model=None, device='cpu', from_native_pdb=True):
    df = pd.read_csv(f'{root}/{file}', header=0)
    samples = []
    with tqdm(range(len(df))) as tbar:
        for idx in tbar:

            row = df.iloc[idx]
            pdb_id = row['PDB ID']
            label_raw = row['Epitope List (residueid_residuename_chain)']

            if pd.isna(label_raw):
                continue

            # Gom label theo chain
            chain_labels = {}

            labels = label_raw.split(', ')
            for item in labels:
                # 148_GLN_A
                parts = item.split('_')
                if len(parts) != 3:
                    continue

                site, amino, chain_id = parts

                if chain_id not in chain_labels:
                    chain_labels[chain_id] = []

                chain_labels[chain_id].append(f"{site}_{amino}")

            for chain_id, label_list in chain_labels.items():

                name = f"{pdb_id}_{chain_id}"
                tbar.set_postfix(protein=name)

                if from_native_pdb:
                    state = extract_chain(root, pdb_id, chain_id)
                    if not state:
                        continue

                data = chain()

                data.protein_name = pdb_id
                data.chain_name = chain_id
                data.name = name

                process_chain(data, root, name, model, device)

                # ---- đảm bảo lấy date từ HEADER ----
                if data.date == '' or data.date is None:
                    try:
                        with open(f"{root}/PDB/{pdb_id}.pdb", "r") as f:
                            for line in f:
                                if line.startswith("HEADER"):
                                    data.date = line[50:59].strip()
                                    break
                    except:
                        data.date = ''

                for j in label_list:
                    site, amino = j.split('_')
                    data.update(site, amino)

                samples.append(data)

    output_name = file.replace(".csv", ".pkl")
    with open(f'{root}/{output_name}', 'wb') as f:
        pk.dump(samples, f)

    return samples

def export_tabular(root, out_dir="./tabular", split='all'):
    """Export per-residue tabular features for XGBoost.
    Produces: <out_dir>/<split>.npz with arrays: X (N x D), y (N,), names (N,), idx (N,), resn (N,)
    split: 'train' or 'test' or 'all' (default 'all' -> concatenate train+test)
    """
    os.makedirs(out_dir, exist_ok=True)
    pk_map = {
        'train': f'{root}/train.pkl',
        'test': f'{root}/test.pkl',
        'all': None,
    }
    samples = []
    if split in ('train','test'):
        p = pk_map[split]
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run initial() first to build pickles")
        with open(p,'rb') as f:
            samples = pk.load(f)
    else:
        # Ưu tiên total.pkl (BCE_633)
        total_path = os.path.join(root, "total.pkl")

        if os.path.exists(total_path):
            with open(total_path, "rb") as f:
                samples = pk.load(f)
        else:
            # Epitope3D: ghép train + test
            files = [
                os.path.join(root, "train.pkl"),
                os.path.join(root, "test.pkl")
            ]

            for p in files:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        samples += pk.load(f)

    rows_X = []
    rows_y = []
    rows_name = []
    rows_idx = []
    rows_resn = []

    for s in tqdm(samples, desc='Exporting residues'):
        # ensure features loaded
        try:
            s.load_feat(root)
            s.load_saprot(root)
            s.load_adj(root,self_cycle=False)
        except Exception as e:
            print(f"[WARN] Failed to load features for {s.name}: {e}")
            continue

        feat = s.feat.float().cpu().numpy() if isinstance(s.feat, torch.Tensor) else np.array(s.feat)
        saprot = s.saprot.float().cpu().numpy() if isinstance(s.saprot, torch.Tensor) else np.array(s.saprot)

        adj = s.adj.float().cpu().numpy() if isinstance(s.adj, torch.Tensor) else np.array(s.adj)

        edge = s.edge.float().cpu().numpy() if isinstance(s.edge, torch.Tensor) else np.array(s.edge)

        amino_ids = s.amino.cpu().numpy() if isinstance(s.amino, torch.Tensor) else np.array(s.amino)

        feat_len = feat.shape[0]
        saprot_len = saprot.shape[0]
        if feat_len != saprot_len:
            if feat_len == saprot_len + 2:
                feat = feat[1:-1]
            elif saprot_len == feat_len + 2:
                saprot = saprot[1:-1]
            else:
                min_len = min(feat_len, saprot_len)
                feat = feat[:min_len]
                saprot = saprot[:min_len]

        L = min(len(s), feat.shape[0], saprot.shape[0])
        for i in range(L):
            esm_i = feat[i]
            saprot_i = saprot[i]
            deg = float(adj[i].sum())
            neighbors = adj[i] > 0
            if neighbors.any():
                neigh_edge_mean = edge[i, neighbors].mean(axis=0)
            else:
                neigh_edge_mean = np.zeros(edge.shape[2], dtype=np.float32)
            amino_id = float(amino_ids[i])
            x = np.concatenate([esm_i.astype(np.float32), saprot_i.astype(np.float32), np.array([deg], dtype=np.float32), neigh_edge_mean.astype(np.float32), np.array([amino_id], dtype=np.float32)])
            rows_X.append(x)
            rows_y.append(float(s.label[i].item() if isinstance(s.label, torch.Tensor) else s.label[i]))
            rows_name.append(s.name)
            rows_idx.append(i)
            rows_resn.append(s.sequence[i])

    X = np.vstack(rows_X).astype(np.float32)
    y = np.array(rows_y, dtype=np.uint8)
    names = np.array(rows_name, dtype=object)
    idxs = np.array(rows_idx, dtype=np.int32)
    resn = np.array(rows_resn, dtype=object)
    out_path = os.path.join(out_dir, f'{split}.npz')
    np.savez_compressed(out_path, X=X, y=y, names=names, idxs=idxs, resn=resn)
    print(f"[DONE] Exported {X.shape[0]} residues to {out_path}")
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/kaggle/input/dataset/data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='all', choices=['train','test','all'])
    args = parser.parse_args()
    export_tabular(args.root, args.out, args.split)