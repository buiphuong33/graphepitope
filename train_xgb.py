# train_xgb.py
import os
import sys
import argparse
import numpy as np
from utils import export_tabular
from tool import METRICS
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import xgboost as xgb
import torch
import logging
from datetime import datetime
import json


def setup_logging(log_dir, log_name=None, log_level=logging.INFO):
    """Thiết lập logging ghi ra file và console"""
    os.makedirs(log_dir, exist_ok=True)
    
    if log_name is None:
        log_name = f"xgb_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_path = os.path.join(log_dir, log_name)
    
    # Cấu hình logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__), log_path


def load_or_export(root, out_dir, split, use_gnn=False, pca_dim=0, logger=None):
    """Load hoặc export dữ liệu tabular"""
    if logger:
        logger.info("Loading/exporting data...")
    
    # prefer merged file if using GNN embeddings
    merged_path = os.path.join(out_dir, f'{split}_merged.npz')
    if use_gnn and os.path.exists(merged_path):
        if logger:
            logger.info(f"Using merged features from {merged_path}")
        return np.load(merged_path, allow_pickle=True)

    path = os.path.join(out_dir, f'{split}.npz')
    if not os.path.exists(path):
        if logger:
            logger.info(f"{path} not found, exporting via utils.export_tabular...")
        export_tabular(root, out_dir, split='all' if split == 'all' else split)
    
    data = np.load(path, allow_pickle=True)
    if logger:
        logger.info(f"Loaded data from {path}: X={data['X'].shape}, y={data['y'].shape}")

    # if requested and merged not present, try to merge automatically
    if use_gnn:
        from merge_tabular_and_gnn import merge
        if logger:
            logger.info(f'Merging tabular and GNN embeddings (PCA dim: {pca_dim})')
        merged_path, pca_model = merge(out_dir, split, pca_dim)
        data = np.load(merged_path, allow_pickle=True)
        if logger:
            logger.info(f"Loaded merged data: X={data['X'].shape}, y={data['y'].shape}")

    return data


def print_metrics(logger, metrics_dict, prefix=""):
    """Helper function để in metrics một cách an toàn"""
    if not metrics_dict:
        logger.warning(f"{prefix} No metrics available")
        return
    
    # Các metrics có thể có
    metric_names = ['F1', 'MCC', 'Precision', 'Recall', 'Accuracy', 'Sensitivity', 'Specificity', 'threshold']
    
    for name in metric_names:
        if name in metrics_dict:
            logger.info(f"  {prefix}{name}: {metrics_dict[name]:.6f}")


def main(args):
    # 1. SETUP LOGGING
    logger, log_file = setup_logging(args.log_dir, args.log_name, args.log_level)
    
    logger.info("="*70)
    logger.info("STARTING XGBOOST TRAINING")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*70)
    
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_model) if os.path.dirname(args.out_model) else '.', exist_ok=True)
    
    # 2. LOAD DATA
    logger.info("\n" + "-"*50)
    logger.info("STEP 1: LOADING DATA")
    logger.info("-"*50)
    
    data = load_or_export(args.root, args.out, args.split, 
                          use_gnn=args.use_gnn, pca_dim=args.pca_dim, logger=logger)
    
    X = data['X']
    y = data['y']
    names = data['names']
    
    # Thống kê dữ liệu
    total_pos = (y == 1).sum()
    total_neg = (y == 0).sum()
    unique_proteins = len(np.unique(names))
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Unique proteins: {unique_proteins}")
    logger.info(f"Total positive residues: {total_pos} ({total_pos/len(y)*100:.2f}%)")
    logger.info(f"Total negative residues: {total_neg} ({total_neg/len(y)*100:.2f}%)")

    # 3. SPLIT DATA
    logger.info("\n" + "-"*50)
    logger.info("STEP 2: SPLITTING DATA (GroupShuffleSplit by protein)")
    logger.info("-"*50)
    
    if args.split == 'all':
        logger.info("Splitting all data into train (80%) and test (20%)...")
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss_test.split(X, y, groups=names))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        names_train = names[train_idx]
        
        logger.info(f"Train set: {len(X_train)} residues from {len(np.unique(names_train))} proteins")
        logger.info(f"  Positive: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
        logger.info(f"  Negative: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
        logger.info(f"Test set: {len(X_test)} residues")
        logger.info(f"  Positive: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")
        logger.info(f"  Negative: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
        
    elif args.split == 'train':
        logger.info("Using train set from file...")
        X_train, y_train, names_train = X, y, names
        test_path = os.path.join(args.out, 'test_merged.npz' if args.use_gnn else 'test.npz')
        
        if not os.path.exists(test_path):
            logger.warning('test.npz not found! Splitting 20% from train for test')
            gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss_test.split(X_train, y_train, groups=names_train))
            X_train, X_test = X_train[train_idx], X_train[test_idx]
            y_train, y_test = y_train[train_idx], y_train[test_idx]
            names_train = names_train[train_idx]
        else:
            logger.info(f"Loading test set from: {test_path}")
            test = np.load(test_path, allow_pickle=True)
            X_test, y_test = test['X'], test['y']
        
        logger.info(f"Train set: {len(X_train)} residues from {len(np.unique(names_train))} proteins")
        logger.info(f"  Positive: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
        logger.info(f"  Negative: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
        logger.info(f"Test set: {len(X_test)} residues")
        logger.info(f"  Positive: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")
        logger.info(f"  Negative: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
    else:
        raise ValueError('split must be train or all')

    # 4. SPLIT TRAIN/VAL
    logger.info("\n" + "-"*50)
    logger.info("STEP 3: SPLITTING TRAIN/VALIDATION (90/10)")
    logger.info("-"*50)
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, val_idx = next(gss_val.split(X_train, y_train, groups=names_train))
    
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    logger.info(f"Training set: {len(X_tr)} residues")
    logger.info(f"  Positive: {(y_tr==1).sum()} ({(y_tr==1).sum()/len(y_tr)*100:.2f}%)")
    logger.info(f"  Negative: {(y_tr==0).sum()} ({(y_tr==0).sum()/len(y_tr)*100:.2f}%)")
    logger.info(f"Validation set: {len(X_val)} residues")
    logger.info(f"  Positive: {(y_val==1).sum()} ({(y_val==1).sum()/len(y_val)*100:.2f}%)")
    logger.info(f"  Negative: {(y_val==0).sum()} ({(y_val==0).sum()/len(y_val)*100:.2f}%)")
    
    # 5. HANDLE IMBALANCE
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg) / float(pos + 1e-9)
    logger.info(f"\nScale_pos_weight: {spw:.2f} (neg/pos ratio)")

    # 6. TRAIN ENSEMBLE
    logger.info("\n" + "="*70)
    logger.info("STEP 4: TRAINING XGBOOST ENSEMBLE")
    logger.info("="*70)
    
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if len(seeds) != 3:
        logger.warning(f"Expected 3 seeds, got {len(seeds)}. Using: {seeds}")
    else:
        logger.info(f"Seeds: {seeds}")

    eval_set = [(X_val, y_val)]
    
    param_grid = [
        dict(
            max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.65,
            min_child_weight=10,
            reg_lambda=5.0, reg_alpha=1.0,
            gamma=2.0,
            name="Conservative"
        ),
        dict(
            max_depth=6, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=2.0, reg_alpha=0.0,
            gamma=0.0,
            name="Balanced"
        ),
        dict(
            max_depth=7, learning_rate=0.05,
            subsample=0.70, colsample_bytree=0.90,
            min_child_weight=1,
            reg_lambda=1.0, reg_alpha=0.0,
            gamma=0.0,
            name="Complex"
        ),
    ]
    probas = []
    n_models = min(len(seeds), len(param_grid))
    model_results = []

    for i in range(n_models):
        seed = seeds[i]
        hp = param_grid[i]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Model {i+1}/{n_models}: {hp['name']}")
        logger.info(f"Seed: {seed}")
        logger.info("Hyperparameters:")
        for key, value in hp.items():
            if key != 'name':
                logger.info(f"  {key}: {value}")
        logger.info(f"{'='*60}")

        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['auc', 'aucpr'],   
            n_estimators=args.n_estimators,      
            early_stopping_rounds=args.early_stopping,
            learning_rate=hp["learning_rate"],
            max_depth=hp["max_depth"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            min_child_weight=hp["min_child_weight"],
            reg_lambda=hp["reg_lambda"],
            reg_alpha=hp["reg_alpha"],
            gamma=hp["gamma"],
            scale_pos_weight=spw,
            tree_method='hist',      
            random_state=seed,
            verbosity=0,
            n_jobs=max(1, os.cpu_count()-1)
        )

        logger.info("Training...")
        
        # Fit với verbose để log quá trình training
        clf.fit(
            X_tr, y_tr, 
            eval_set=eval_set,
            verbose=args.log_interval
        )

        # Log kết quả
        logger.info(f"\n[{hp['name']}] Training completed!")
        logger.info(f"  Best iteration: {clf.best_iteration}")
        
        # Lấy best score từ evals_result
        evals_result = clf.evals_result()
        if evals_result and 'validation_0' in evals_result:
            val_auc = evals_result['validation_0']['auc']
            val_aucpr = evals_result['validation_0']['aucpr']
            if val_auc:
                best_auroc = max(val_auc)
                best_auprc = max(val_aucpr) if val_aucpr else 0
                logger.info(f"  Best AUROC: {best_auroc:.6f}")
                logger.info(f"  Best AUPRC: {best_auprc:.6f}")

        # Lưu model
        base, ext = os.path.splitext(args.out_model)
        model_path = f"{base}_m{i+1}_seed{seed}{ext}"
        dirn = os.path.dirname(model_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        joblib.dump(clf, model_path)
        logger.info(f"  Model saved to: {model_path}")
        
        # Dự đoán trên validation
        val_pred = clf.predict_proba(X_val)[:, 1]
        metrics_val = METRICS(device='cpu')
        val_res = metrics_val.calc_prc(torch.tensor(val_pred), torch.tensor(y_val))
        val_thr = metrics_val(torch.tensor(val_pred), torch.tensor(y_val))
        
        logger.info(f"  Val AUROC: {val_res['AUROC']:.6f}, Val AUPRC: {val_res['AUPRC']:.6f}")
        logger.info(f"  Val metrics:")
        print_metrics(logger, val_thr, "    ")
        
        # Dự đoán trên test
        probas.append(clf.predict_proba(X_test)[:, 1])
        
        model_results.append({
            'model_idx': i+1,
            'name': hp['name'],
            'seed': seed,
            'best_iteration': clf.best_iteration,
            'val_auroc': val_res['AUROC'],
            'val_auprc': val_res['AUPRC'],
            'val_metrics': val_thr
        })

    # 7. ENSEMBLE EVALUATION
    logger.info("\n" + "="*70)
    logger.info("STEP 5: ENSEMBLE EVALUATION ON TEST SET")
    logger.info("="*70)
    
    logger.info(f"Averaging predictions from {len(probas)} models...")
    proba = np.mean(np.vstack(probas), axis=0)
    logger.info(f"Ensemble predictions shape: {proba.shape}")

    metrics = METRICS(device='cpu')
    pred_t = torch.tensor(proba)
    y_t = torch.tensor(y_test)
    
    logger.info("\n" + "-"*50)
    logger.info("ENSEMBLE PERFORMANCE")
    logger.info("-"*50)
    
    # AUROC & AUPRC
    res = metrics.calc_prc(pred_t, y_t)
    logger.info(f"AUROC: {res['AUROC']:.6f}")
    logger.info(f"AUPRC: {res['AUPRC']:.6f}")
    
    # Thresholded metrics
    thr_metrics = metrics(pred_t, y_t)
    logger.info(f"\nWith optimal threshold (maximizing F1):")
    logger.info(f"  Threshold: {thr_metrics.get('threshold', 'N/A')}")
    
    # In tất cả metrics một cách an toàn
    logger.info("  Metrics:")
    for key, value in thr_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"    {key}: {value:.6f}")
        else:
            logger.info(f"    {key}: {value}")
    
    # 8. SAVE RESULTS
    logger.info("\n" + "-"*50)
    logger.info("STEP 6: SAVING RESULTS")
    logger.info("-"*50)
    
    # Save predictions
    pred_path = os.path.join(args.out, 'xgb_test_preds.npz')
    np.savez_compressed(pred_path, proba=proba, y=y_test)
    logger.info(f"Predictions saved to: {pred_path}")
    
    # Save detailed results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'split': args.split,
        'use_gnn': args.use_gnn,
        'pca_dim': args.pca_dim,
        'seeds': seeds,
        'n_estimators': args.n_estimators,
        'early_stopping': args.early_stopping,
        'log_interval': args.log_interval,
        'train_size': len(X_tr),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'positive_ratio': float(pos / len(y_tr) * 100),
        'scale_pos_weight': float(spw),
        'num_models': n_models,
        'model_results': model_results,
        'ensemble_performance': {
            'auroc': float(res['AUROC']),
            'auprc': float(res['AUPRC']),
            'thresholded_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                    for k, v in thr_metrics.items()}
        }
    }
    
    result_json_path = os.path.join(args.out, 'xgb_results.json')
    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {result_json_path}")
    
    # 9. SUMMARY
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Results: {result_json_path}")
    logger.info(f"Predictions: {pred_path}")
    logger.info(f"Models: {os.path.dirname(args.out_model)}")
    logger.info("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/kaggle/input/dataset/data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'all'])
    parser.add_argument('--out-model', type=str, default='./model/xgb_model.joblib')
    parser.add_argument('--use-gnn', action='store_true', help='Use GNN embeddings merged into features')
    parser.add_argument('--pca-dim', type=int, default=10, help='PCA dim for GNN embeddings (fit on train)')
    parser.add_argument('--seeds', type=str, default='42,202,777',
                        help='Comma-separated random seeds for ensemble (e.g., 42,202,777)')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save log files')
    parser.add_argument('--log-name', type=str, default=None, help='Log file name (auto-generated if None)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--dataset', type=str, default='BCE_633', help='Dataset name for logging')
    
    # XGBoost arguments
    parser.add_argument('--n_estimators', type=int, default=3000, help='Number of boosting rounds')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping rounds')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N epochs (verbose parameter)')
    
    args = parser.parse_args()
    
    # Convert log level string to constant
    log_level = getattr(logging, args.log_level.upper())
    args.log_level = log_level
    
    main(args)