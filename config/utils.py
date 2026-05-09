"""
utils.py — 工具函数
==================
- 方向准确率(DA)计算
- 随机种子设置
- 日志与格式化
"""
import os, sys, random, time
import numpy as np
import pandas as pd

def set_seed(seed):
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def da_score(y_true, y_pred):
    """方向准确率 Directional Accuracy."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def all_metrics(y_true, y_pred):
    """返回完整评估指标字典。"""
    assert len(y_true) == len(y_pred), "长度不匹配"
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'da': da_score(y_true, y_pred),
        'n': len(y_true),
    }

def format_result(model, horizon, seed, metrics, extra=None):
    """格式化单条实验结果。"""
    row = {
        'model': model,
        'horizon': horizon,
        'seed': seed,
        'rmse': round(metrics['rmse'], 6),
        'mae': round(metrics['mae'], 6),
        'da': round(metrics['da'], 4),
        'n': metrics.get('n', 0),
    }
    if extra:
        row.update(extra)
    return row

def timer(func):
    """装饰器：记录函数执行时间。"""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        if isinstance(result, dict):
            result['time_sec'] = round(elapsed, 2)
        return result
    return wrapper

def save_results(df_results, filename, subdir=''):
    """保存结果CSV到 results/ 目录。"""
    from config import RESULTS_DIR
    out_path = os.path.join(RESULTS_DIR, subdir, filename) if subdir else os.path.join(RESULTS_DIR, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    return out_path
