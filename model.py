#model.py
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import EGAT,AE, HierarchicalPooling
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, hidden_dim=256, 
        exfeat_dim=1280, edge_dim=51, 
        augment_eps=0.05, dropout=0.2, 
        lr=1e-6, metrics=None, result_path=None
    ):
        super().__init__()
        self.metrics=metrics
        self.path=result_path
        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []
        # loss function
        self.loss_fn=nn.BCELoss()
        # Hyperparameters
        self.exfeat_dim=exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias=False
        self.W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)
        self.W_u1 = AE(exfeat_dim,hidden_dim,hidden_dim, bias=bias)
        self.edge_linear=nn.Sequential(
            nn.Linear(edge_dim,hidden_dim//4, bias=True),
            nn.ELU(),
        )
        self.gat=EGAT(hidden_dim,hidden_dim,hidden_dim//4,dropout)
        self.hpool = HierarchicalPooling(hidden_dim=hidden_dim, pool_ratio=4)
        self.feature_gate = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.mlp=nn.Sequential(
            nn.Linear(3*hidden_dim,hidden_dim,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim,1,bias=True),
            nn.Sigmoid()
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge, adj):
        h=[]
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask=V.sum(-1)!=0
        if self.training and self.augment_eps > 0:
            aug=torch.randn_like(V)
            aug[~mask]=0
            V = V+self.augment_eps * aug
        mask=mask.sum(1)
        feats,exfeats=self.W_v(V[:,:,:-self.exfeat_dim]),self.W_u1(V[:,:,-self.exfeat_dim:])
        x_gcns=[]
        for i in range(len(V)):
            E=self.edge_linear(edge[i]).permute(2,0,1)
            A = adj[i].to(V.device).float() # Lấy ma trận kề của đồ thị i
            x1,x2=feats[i,:mask[i]],exfeats[i,:mask[i]]
            x_cat = torch.cat([x1, x2], dim=-1) 
            gate = torch.softmax(self.feature_gate(x_cat), dim=-1)
            x = gate[:, 0:1] * x1 + gate[:, 1:2] * x2
            x_local,E=self.gat(x,E)
            x_gcn_final = self.hpool(x_local, A)
            x_gcns.append(x_gcn_final)
        
        x_attns=torch.cat([feats,exfeats],-1)
        
        x_attns=[x_attns[i,:mask[i]] for i in range(len(x_attns))]
        h=[torch.cat([x_attn,x_gcn],-1) for x_attn,x_gcn in zip(x_attns,x_gcns)]
        h=torch.cat(h,0)
        return self.mlp(h)

    def embed(self, V, edge):
        """Return per-residue embeddings from the model (before final MLP).
        Input:
            V: list of tensors (L_i x D)
            edge: list of tensors (L_i x L_i x edge_dim)
        Output:
            List of tensors [ (L_i x H) ] where H = concat(LSTM_out_dim + GCN_out_dim)
        """
        was_train = self.training
        self.eval()
        with torch.no_grad():
            V = pad_sequence(V, batch_first=True, padding_value=0).float()
            mask=V.sum(-1)!=0
            mask_lens=mask.sum(1)
            feats,exfeats=self.W_v(V[:,:,:-self.exfeat_dim]),self.W_u1(V[:,:,-self.exfeat_dim:])
            x_gcns=[]
            for i in range(len(V)):
                E=self.edge_linear(edge[i]).permute(2,0,1)
                x1,x2=feats[i,:mask_lens[i]],exfeats[i,:mask_lens[i]]
                x_gcn=torch.cat([x1,x2],-1)
                x_gcn,_=self.gat(x_gcn,E)
                x_gcns.append(x_gcn)
            
            x_attns = torch.cat([feats, exfeats], -1)
            x_attns=[x_attns[i,:mask_lens[i]] for i in range(len(x_attns))]
            h_list=[torch.cat([x_attn,x_gcn],-1) for x_attn,x_gcn in zip(x_attns,x_gcns)]
        if was_train:
            self.train()
        return h_list

    def embed_gnn_only(self, V, edge):
        """Return per-residue embeddings produced by GNN only (W_v/W_u1 + EGAT), skipping LSTM.
        Input/Output like embed(): returns a list of tensors [(L_i x H), ...]
        """
        was_train = self.training
        self.eval()
        with torch.no_grad():
            V = pad_sequence(V, batch_first=True, padding_value=0).float()
            mask = V.sum(-1) != 0
            mask_lens = mask.sum(1)
            feats = self.W_v(V[:,:,:-self.exfeat_dim])
            exfeats = self.W_u1(V[:,:,-self.exfeat_dim:])
            gcn_outs = []
            for i in range(len(V)):
                E = self.edge_linear(edge[i]).permute(2,0,1)
                x1 = feats[i,:mask_lens[i]]
                x2 = exfeats[i,:mask_lens[i]]
                x = torch.cat([x1, x2], -1)
                x_gcn, _ = self.gat(x, E)
                gcn_outs.append(x_gcn)
        if was_train:
            self.train()
        return gcn_outs
    def training_step(self, batch, batch_idx): 
        feat, edge, adj, y = batch
        pred = self(feat, edge, adj).squeeze(-1)
        loss=self.loss_fn(pred,y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.metrics is not None:
            result=self.metrics.calc_prc(pred.detach().clone(),y.detach().clone())
            self.log('train_auc', result['AUROC'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train_prc', result['AUPRC'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, edge, adj, y = batch
        pred = self(feat, edge, adj).squeeze(-1)
        self.val_preds.append(pred.detach())
        self.val_labels.append(y.detach())
        # log loss theo step (tùy chọn)
        loss = self.loss_fn(pred, y.float())
        #self.log('val_step_loss', loss.detach().cpu().item(), on_step=False, on_epoch=False)
        return
    

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        pred = torch.cat(self.val_preds, 0)
        y    = torch.cat(self.val_labels, 0)
        # reset bộ đệm
        self.val_preds.clear(); self.val_labels.clear()

        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss.cpu().item(),on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('val_AUROC', result['AUROC'],on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_mcc',   result['MCC'],   on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_f1',    result['F1'],    on_step=False, on_epoch=True, prog_bar=False, logger=True)
        print(
            f"Epoch {self.current_epoch} | "
            f"val_loss {loss:.4f} | "
            f"AUROC {result['AUROC']:.4f} | "
            f"AUPRC {result['AUPRC']:.4f}"
        )

    def test_step(self, batch, batch_idx):
        feat, edge, adj, y = batch
        pred = self(feat, edge, adj).squeeze(-1)
        self.test_preds.append(pred.detach())
        self.test_labels.append(y.detach())
        return
    
    
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        pred = torch.cat(self.test_preds, 0)
        y    = torch.cat(self.test_labels, 0)
        # reset bộ đệm
        self.test_preds.clear(); self.test_labels.clear()

        loss = self.loss_fn(pred, y.float())

        if self.path:
            if not os.path.exists(self.path):
                os.system(f'mkdir -p {self.path}')
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('test_loss',      loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_AUROC',     result['AUROC'],   on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_AUPRC',     result['AUPRC'],   on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_recall',    result['RECALL'],  on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_precision', result['PRECISION'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_f1',        result['F1'],      on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_mcc',       result['MCC'],     on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('test_bacc',      result['BACC'],    on_step=False,on_epoch=True, prog_bar=False, logger=True)
            self.log('test_threshold', result['threshold'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)