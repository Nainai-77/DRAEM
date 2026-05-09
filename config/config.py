"""
config.py — DRAEM 实验配置
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# === 数据集 ===
DATASET_CSV = os.path.join(BASE_DIR, '..', '05_merged_dataset', 'dataset_forecast_v1_h7.csv')

# === 结构化特征（26个） ===
NUMERIC_COLS = [
    'close', 'open', 'high', 'low', 'volume', 'amount',
    'log_return', 'volatility_20d', 'volatility_60d', 'range_pct',
    'return_lag1', 'return_lag3', 'return_lag5', 'volume_lag1',
    'mkt_hbea_close', 'mkt_bea_close', 'mkt_shea_close',
    'mkt_szea_close', 'mkt_cea_close',
    'hs300_ret', 'eua_ret', 'blt_oil_ret', 'dq_oil_ret', 'coal_ret',
    'cpi', 'm2',
]

# === 文本特征 — 按模块分组 ===
TEXT_MODULES = {
    'policy': [
        'policy_cnt', 'policy_has', 'policy_alen',
        'policy_roll3', 'policy_roll7', 'policy_roll14',
        'policy_lag1', 'policy_lag3', 'policy_lag5',
        'policy_gap', 'policy_sent',
    ],
    'market': [
        'market_cnt', 'market_has', 'market_alen',
        'market_roll3', 'market_roll7', 'market_roll14',
        'market_lag1', 'market_lag3', 'market_lag5',
        'market_gap', 'market_sent',
    ],
    'finance': [
        'finance_cnt', 'finance_has', 'finance_alen',
        'finance_roll3', 'finance_roll7', 'finance_roll14',
        'finance_lag1', 'finance_lag3', 'finance_lag5',
        'finance_gap', 'finance_sent',
    ],
    'trade': [
        'trade_cnt', 'trade_has', 'trade_alen',
        'trade_roll3', 'trade_roll7', 'trade_roll14',
        'trade_lag1', 'trade_lag3', 'trade_lag5',
        'trade_gap', 'trade_sent',
    ],
    'compliance': [
        'compliance_cnt', 'compliance_has', 'compliance_alen',
        'compliance_roll3', 'compliance_roll7', 'compliance_roll14',
        'compliance_lag1', 'compliance_lag3', 'compliance_lag5',
        'compliance_gap', 'compliance_sent',
    ],
}

TEXT_GLOBAL_COLS = [
    'global_cnt', 'global_has', 'quality_score',
    'importance_score', 'global_shock_flag',
    'compliance_window', 'doc_mask',
]

# 展平所有文本列
SELECTED_TEXT_COLS = []
for mcols in TEXT_MODULES.values():
    SELECTED_TEXT_COLS += mcols
SELECTED_TEXT_COLS += TEXT_GLOBAL_COLS

# === 目标变量 ===
TARGET_H1 = 'target_H1'
TARGET_H5 = 'target_H5'
TARGET_H7 = 'target_H7'

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEEDS = [42, 123, 456, 789, 2024]

# === 超参数 ===
HP = {
    'ridge': {'alpha': 1.0},
    'xgboost': {
        'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
    },
    # DRAEM 主模型
    'draem': {
        'd': 32,              # 隐藏层维度
        'n_enc_layers': 2,    # 编码器层数
        'dropout': 0.3,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 500,
        'patience': 40,
        'batch_size': 128,
        'lambda_dir': 0.5,      # 方向损失权重
        'lambda_regime': 0.15,  # Regime辅助任务权重
    },
}
