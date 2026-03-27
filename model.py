import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from HGNN import HierarchicalGNN


class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, 
        exfeat_dim=1280, 
        hidden_dim=256, 
        num_layers=3,                 # 🔥 thêm
        augment_eps=0.05, 
        dropout=0.2, 
        lr=1e-4, 
        hcl_weight=0.1,
        metrics=None, 
        result_path=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = metrics
        self.path = result_path
        self.lr = lr
        self.hcl_weight = hcl_weight

        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

        # 🔥 dùng BCEWithLogitsLoss ổn định hơn
        self.loss_fn = nn.BCEWithLogitsLoss()

        # ===== Feature projection =====
        self.W_v = nn.Linear(feat_dim, hidden_dim)
        self.W_u1 = nn.Linear(exfeat_dim, hidden_dim)

        # ===== HGNN mới =====
        self.hgnn = HierarchicalGNN(
            in_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ===== Classifier mạnh hơn =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ===== HCL loss cải tiến =====
    def compute_hcl_loss(self, layer_outputs, labels):
        hcl_loss = 0
        labels = labels.float()

        for z in layer_outputs:
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_mean = z[pos_mask].mean(0)
                neg_mean = z[neg_mask].mean(0)

                hcl_loss += -F.mse_loss(pos_mean, neg_mean)

        return hcl_loss / len(layer_outputs)   # 🔥 normalize

    # ===== Forward =====
    def forward(self, x, pos, batch_idx=None):
        esm_feat = x[:, :self.hparams.feat_dim]
        saprot_feat = x[:, self.hparams.feat_dim:]

        v = F.elu(self.W_v(esm_feat))
        u = F.elu(self.W_u1(saprot_feat))

        combined_x = torch.cat([v, u], dim=-1)

        if self.training and self.hparams.augment_eps > 0:
            combined_x = combined_x + torch.randn_like(combined_x) * self.hparams.augment_eps

        layer_feats, final_feat = self.hgnn(combined_x, pos, batch_idx)

        logits = self.classifier(final_feat).squeeze(-1)

        return logits, layer_feats

    # ===== Training =====
    def training_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch

        logits, layer_feats = self(x, pos, batch_ptr)

        loss_bce = self.loss_fn(logits, y.float())
        loss_hcl = self.compute_hcl_loss(layer_feats, y)

        total_loss = loss_bce + self.hcl_weight * loss_hcl

        if self.metrics is not None:
            pred = torch.sigmoid(logits)
            result = self.metrics.calc_prc(pred.detach(), y.detach())
            self.log('train_auc', result['AUROC'], on_epoch=True)

        return total_loss

    # ===== Validation =====
    def validation_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch

        logits, _ = self(x, pos, batch_ptr)
        pred = torch.sigmoid(logits)

        self.val_preds.append(pred.detach())
        self.val_labels.append(y.detach())

        return self.loss_fn(logits, y.float())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        pred = torch.cat(self.val_preds, 0)
        y = torch.cat(self.val_labels, 0)

        self.val_preds.clear()
        self.val_labels.clear()

        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss, on_epoch=True, prog_bar=False)

        if self.metrics is not None:
            result = self.metrics(pred, y)
            for k, v in result.items():
                self.log(f'val_{k}', v, on_epoch=True, prog_bar=False)

            print(
                f"Epoch {self.current_epoch} | "
                f"val_loss {loss:.4f} | "
                f"AUROC {result['AUROC']:.4f} | "
                f"AUPRC {result['AUPRC']:.4f}"
            )

    # ===== Test =====
    def test_step(self, batch, batch_idx):
        x, pos, y, batch_ptr = batch.x, batch.pos, batch.y, batch.batch

        logits, _ = self(x, pos, batch_ptr)
        pred = torch.sigmoid(logits)

        self.test_preds.append(pred.detach())
        self.test_labels.append(y.detach())

    def on_test_epoch_end(self):
        if not self.test_preds:
            return

        pred = torch.cat(self.test_preds, 0)
        y = torch.cat(self.test_labels, 0)

        self.test_preds.clear()
        self.test_labels.clear()

        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        if self.metrics is not None:
            result = self.metrics(pred, y)
            for k, v in result.items():
                self.log(f'test_{k}', v, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)