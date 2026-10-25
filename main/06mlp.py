import os, sys, logging, numpy as np, pandas as pd
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, mixed_precision
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import re

# Suppress TensorFlow plugin registration warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# GPU environment flags (intentional)
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_cuda_data_dir=/mnt/UKB1/magnus/scratch/env/ml/nvvm/libdevice')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_tensorflow(require_gpu=True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                logger.warning(f"Failed to set memory growth on {gpu}: {exc}")
        mixed_precision.set_global_policy('mixed_float16')
        logger.info("GPU detected; mixed precision enabled.")
    else:
        msg = "No GPU detected; refusing to run on CPU."
        if require_gpu:
            raise RuntimeError(msg)
        logger.warning(msg)

def make_dataset(x, y=None, batch_size=1024, shuffle=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- helpers ---
def parse_suffix(sfx):
    m = re.match(r"rep(\d+)_r2(\d+)_p(\d+)", sfx)
    if not m:
        return {"rep": None, "r2_tag": None, "pval_tag": None}
    rep, r2_tag, pval_tag = m.groups()
    return {"rep": int(rep), "r2_tag": r2_tag, "pval_tag": pval_tag}

def load_labels(path):
    y = pd.read_csv(path, sep=r'\s+', header=None, names=['label'], engine='python').squeeze()
    y = pd.to_numeric(y, errors='coerce')
    if set(pd.unique(y.dropna())) <= {1, 2}:
        y = y.astype(int) - 1
    return y

def load_header(path):
    with open(path) as f:
        return f.read().strip().split()

def safe_norm_contrib(weights):
    abs_w = np.abs(weights)
    denom = abs_w.sum()
    if denom == 0:
        return np.zeros_like(abs_w)
    return abs_w / denom

def build_model(n_features, lr, l2, n_hidden, hidden_units, dropout_rate, batchnorm):
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    for _ in range(n_hidden):
        model.add(layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2)
        ))
        if batchnorm:
            model.add(layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.add(layers.Lambda(lambda x: tf.squeeze(x, axis=-1)))  # Remove last dimension
    model.add(layers.Activation('sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryCrossentropy(name='bce'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

# --- main pipeline ---
def train_and_evaluate(xtrain, ytrain, header, out_dir, sfx, n_trials, n_splits, rep, test_sets):
    os.makedirs(out_dir, exist_ok=True)
    start_ts = datetime.utcnow().isoformat()
    configure_tensorflow(require_gpu=True)

    # Load data
    y_train = load_labels(ytrain).astype(int)
    classes = np.unique(y_train)
    cw_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
    feature_names = load_header(header)
    X_all = pd.read_csv(xtrain, sep=' ', header=None).fillna(0).values.astype(np.float32)

    # Optuna objective
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        l2 = trial.suggest_float('l2', 1e-6, 1e-3, log=True)
        n_hidden = trial.suggest_int('n_hidden', 1, 10)
        hidden_units = trial.suggest_int('hidden_units', 1, 10)
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
        batchnorm = trial.suggest_categorical('batchnorm', [False, True])
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs, prev_auc = [], None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(y_train)), y_train)):
            model = build_model(X_all.shape[1], lr, l2, n_hidden, hidden_units, dropout_rate, batchnorm)
            
            # Early stopping within CV fold
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                restore_best_weights=True,
                verbose=0
            )
            
            train_ds = make_dataset(
                X_all[train_idx],
                y_train.iloc[train_idx].values,
                batch_size=batch_size,
                shuffle=True
            )
            val_ds = make_dataset(
                X_all[val_idx],
                y_train.iloc[val_idx].values,
                batch_size=batch_size,
                shuffle=False
            )
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=200,
                verbose=0,
                class_weight=class_weight,
                callbacks=[early_stop]
            )
            val_pred_ds = make_dataset(X_all[val_idx], batch_size=1024, shuffle=False)
            probs = model.predict(val_pred_ds, verbose=0).ravel()
            auc_val = roc_auc_score(y_train.iloc[val_idx], probs)
            aucs.append(auc_val)

            mean_auc = float(np.mean(aucs))
            val_bce = history.history['val_bce'][-1] if 'val_bce' in history.history else None
            epochs_trained = len(history.history['loss'])
            
            if prev_auc is None:
                if val_bce is not None:
                    logger.info(f"Trial {trial.number}, Fold {fold_idx}: AUC={auc_val:.4f}, BCE={val_bce:.4f}, Epochs={epochs_trained}")
                else:
                    logger.info(f"Trial {trial.number}, Fold {fold_idx}: AUC={auc_val:.4f}, Epochs={epochs_trained}")
            else:
                delta = auc_val - prev_auc
                if val_bce is not None:
                    logger.info(f"Trial {trial.number}, Fold {fold_idx}: AUC={auc_val:.4f} (Δ={delta:+.4f}), BCE={val_bce:.4f}, Epochs={epochs_trained}")
                else:
                    logger.info(f"Trial {trial.number}, Fold {fold_idx}: AUC={auc_val:.4f} (Δ={delta:+.4f}), Epochs={epochs_trained}")
            prev_auc = auc_val

            trial.report(mean_auc, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr('fold_aucs', aucs)
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=n_trials)

    # Save Optuna trials dataframe
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(out_dir, f"mlp_tf_optuna_trials_{sfx}.csv")
    trials_df.to_csv(trials_csv, index=False)

    # Optuna visualizations (XGB-style)
    try:
        vis.plot_optimization_history(study).write_html(os.path.join(out_dir, f"optuna_history_{sfx}.html"))
        vis.plot_parallel_coordinate(study).write_html(os.path.join(out_dir, f"optuna_parallel_{sfx}.html"))
        vis.plot_param_importances(study).write_html(os.path.join(out_dir, f"optuna_importances_{sfx}.html"))
        vis.plot_slice(study).write_html(os.path.join(out_dir, f"optuna_slice_{sfx}.html"))
        vis.plot_contour(study).write_html(os.path.join(out_dir, f"optuna_contour_{sfx}.html"))
        vis.plot_edf(study).write_html(os.path.join(out_dir, f"optuna_edf_{sfx}.html"))
    except Exception as e:
        logger.warning(f"Optuna visualization failed: {e}")

    best = study.best_params
    best_cv_mean_auc = float(study.best_value)

    # Transparency: recompute per-fold AUCs using best hyperparameters
    skf_best = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    per_fold_aucs = []
    for fold_i, (train_idx, val_idx) in enumerate(skf_best.split(np.arange(len(y_train)), y_train), 1):
        model_fold = build_model(
            X_all.shape[1],
            best['lr'], best['l2'],
            best['n_hidden'], best['hidden_units'],
            best['dropout'], best['batchnorm']
        )
        early_stop_fold = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=0
        )
        fold_train_ds = make_dataset(
            X_all[train_idx],
            y_train.iloc[train_idx].values,
            batch_size=best['batch_size'],
            shuffle=True
        )
        fold_val_ds = make_dataset(
            X_all[val_idx],
            y_train.iloc[val_idx].values,
            batch_size=best['batch_size'],
            shuffle=False
        )
        fold_history = model_fold.fit(
            fold_train_ds,
            validation_data=fold_val_ds,
            epochs=200,
            verbose=0,
            class_weight=class_weight,
            callbacks=[early_stop_fold]
        )
        fold_pred_ds = make_dataset(X_all[val_idx], batch_size=1024, shuffle=False)
        probs_fold = model_fold.predict(fold_pred_ds, verbose=0).ravel()
        per_fold_aucs.append(float(roc_auc_score(y_train.iloc[val_idx], probs_fold)))
        fold_epochs = len(fold_history.history['loss'])
        logger.info(f"Best-params CV Fold {fold_i}: Epochs={fold_epochs}")

    # Final model training
    model = build_model(
        X_all.shape[1],
        best['lr'], best['l2'],
        best['n_hidden'], best['hidden_units'],
        best['dropout'], best['batchnorm']
    )
    final_train_ds = make_dataset(
        X_all,
        y_train.values,
        batch_size=best['batch_size'],
        shuffle=True
    )
    # Use a large number of epochs; early stopping in validation handled during CV
    final_history = model.fit(
        final_train_ds,
        epochs=200,
        verbose=0,
        class_weight=class_weight
    )
    final_epochs = len(final_history.history['loss'])
    logger.info(f"Final model trained for {final_epochs} epochs")

    # Save model (aligned naming)
    model_path = os.path.join(out_dir, f"mlp_tf_model_{sfx}.keras")
    model.save(model_path)
    logger.info(f"Saved trained model to {model_path}")

    # Feature importance from first layer weights
    # First dense layer is layers[1] if an Input layer exists; but Sequential keeps Dense at index 0 after Input
    # We use the first Dense layer with weights (if n_hidden >= 1)
    first_dense_layer = None
    for lyr in model.layers:
        if isinstance(lyr, layers.Dense) and lyr.get_weights():
            first_dense_layer = lyr
            break
    if first_dense_layer is None:
        logger.warning("No dense layer with weights found; feature importance cannot be computed.")
        feature_importance = np.zeros(X_all.shape[1], dtype=np.float32)
    else:
        W = first_dense_layer.get_weights()[0]  # (n_features, units)
        feature_importance = np.sum(np.abs(W), axis=1)

    norm_contrib = safe_norm_contrib(feature_importance)
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'norm_contrib': norm_contrib
    }).sort_values(by='norm_contrib', ascending=False)
    feature_csv = os.path.join(out_dir, f"mlp_tf_feature_importance_{sfx}.csv")
    feature_df.to_csv(feature_csv, index=False)

    # Train AUC
    train_pred_ds = make_dataset(X_all, batch_size=1024, shuffle=False)
    train_probs = model.predict(train_pred_ds, verbose=0).ravel()
    train_auc = roc_auc_score(y_train, train_probs)

    # Evaluate test sets
    results, roc_curves = {}, []
    logger.info(f"Evaluating {len(test_sets)} test sets...")
    for label, xt, yt in test_sets:
        if not (os.path.exists(xt) and os.path.exists(yt)):
            logger.warning(f"Skipping {label}: X={os.path.exists(xt)}, y={os.path.exists(yt)}")
            logger.warning(f"  X path: {xt}")
            logger.warning(f"  y path: {yt}")
            continue
        logger.info(f"Evaluating test set: {label}")

        X_test = pd.read_csv(xt, sep=' ', header=None).fillna(0).values.astype(np.float32)
        y_test = load_labels(yt).astype(int).values
        test_pred_ds = make_dataset(X_test, batch_size=1024, shuffle=False)
        probs = model.predict(test_pred_ds, verbose=0).ravel()
        auc_val = float(roc_auc_score(y_test, probs))
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves.append((fpr, tpr, label, auc_val))
        results[label] = auc_val

    # Save ROC plot (aligned naming)
    plt.figure(figsize=(7,6))
    for fpr, tpr, label, auc_val in roc_curves:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"MLP TF ROC – {sfx}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'mlp_tf_roc_{sfx}.png'))
    plt.close()

    # Wide summary CSV (XGB-style fields)
    parts = parse_suffix(sfx)
    summary = {
        "model_type": "mlp_tf",
        "suffix": sfx,
        "rep": rep,
        "r2_tag": parts["r2_tag"],
        "pval_tag": parts["pval_tag"],
        "n_features": X_all.shape[1],
        "n_trials": n_trials,
        "n_splits": n_splits,
        "lr": best['lr'],
        "l2": best['l2'],
        "n_hidden": best['n_hidden'],
        "hidden_units": best['hidden_units'],
        "dropout": best['dropout'],
        "batchnorm": best['batchnorm'],
        "batch_size": best['batch_size'],
        "cv_auc": best_cv_mean_auc,
        "train_auc": train_auc
    }
    for label, auc_val in results.items():
        summary[f"test_auc_{label}"] = auc_val
    for i, auc in enumerate(per_fold_aucs, 1):
        summary[f"cv_fold{i}_auc"] = auc

    wide_csv = os.path.join(out_dir, "mlp_tf_all_runs_summary.csv")
    pd.DataFrame([summary]).to_csv(
        wide_csv,
        mode="a",
        header=not os.path.exists(wide_csv),
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) < 9:
        print("Usage: python 06mlp.py <X_train> <y_train> <header> <out_dir> <sfx> <n_trials> <n_splits> <rep> <label1> <X_test1> <y_test1> ...")
        sys.exit(1)
    _, X_train_p, y_train_p, header_p, out_dir, sfx, n_trials, n_splits, rep, *test_paths = sys.argv
    # Only process complete triplets (label, X_test, y_test)
    test_sets = [(test_paths[i], test_paths[i+1], test_paths[i+2]) for i in range(0, len(test_paths) - 2, 3)]
    train_and_evaluate(X_train_p, y_train_p, header_p, out_dir, sfx, int(n_trials), int(n_splits), int(rep), test_sets)
