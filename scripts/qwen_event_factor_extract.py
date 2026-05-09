#!/usr/bin/env python3
"""
qwen_event_factor_extract.py

Aggressive Qwen semantic event-factor extraction for carbon-price forecasting.

This replaces conservative same-day Qwen refinement with structured event factors:
relevance, direction, strength, confidence, persistence, uncertainty, and mechanism scores.

Default input:
  /root/autodl-tmp/DRAEM_paper/llm_experiments/data/daily_texts_nonempty.csv

Outputs:
  results/qwen_event_factor/qwen_event_raw.csv
  results/qwen_event_factor/qwen_event_daily.csv

Long GPU run: write script first; run manually or as a controlled job.
"""
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'Qwen2.5-7B-Instruct'
DEFAULT_IN = Path(__file__).resolve().parent.parent / 'llm_experiments' / 'data' / 'daily_texts_nonempty.csv'
DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'qwen_event_factor'
MODULES = ["policy", "market", "finance", "trade", "compliance"]

MECHANISMS = [
    "policy_tightening",
    "allowance_supply",
    "compliance_pressure",
    "ccer_supply",
    "trading_liquidity",
    "industrial_demand",
    "energy_macro_linkage",
    "market_expectation",
]

FIELDS = [
    "relevance", "impact_direction", "impact_strength", "confidence",
    "decay_days", "uncertainty", *MECHANISMS,
]

SYSTEM_PROMPT = """你是中国碳市场政策、交易与碳价形成机制专家。
你的任务不是直接预测未来价格，而是把文本转成可用于碳价方向预测的结构化“事件因子”。

重要原则：
1. 不要过度保守。只要文本存在合理的碳价传导机制，就应给出非零方向或机制分数。
2. 必须同时识别利多和利空。不要因为文本语气积极就自动判定为利多碳价；要判断其对“碳配额价格/碳价”的供需影响。
3. 中性只用于完全无法判断或与碳价没有实际关系的文本。
4. 方向表示对碳价的潜在影响，不是对新闻情绪的简单判断。
5. 只输出一个JSON对象；不要解释、不要Markdown、不要代码块。

典型利空碳价机制示例：
- CCER项目扩容、CCER供给增加、抵消机制更便利：可能降低配额履约需求，通常对配额价格利空，ccer_supply 应为负向机制。
- 配额供给宽松、企业大量净卖出配额、配额盈余/过剩：通常对价格利空，allowance_supply 应体现供给宽松。
- 碳价指数、买入价/卖出价预期下行，或价格下跌：通常 impact_direction 为负。
- 成交清淡、流动性下降、市场活跃度下降：通常 trading_liquidity 为负，可能利空或增加不确定性。
- 工业需求、能源需求、排放需求走弱：通常 industrial_demand 为负。
- 履约压力下降、清缴压力减轻、政策延期/放松/约束弱化：通常 compliance_pressure 或 policy_tightening 为负。

典型利多碳价机制示例：
- 政策趋严、扩围纳入更多行业、配额收紧、履约压力上升、需求增强、市场预期上行，通常对碳价利多。
"""

USER_TEMPLATE = """请分析以下中文碳市场文本，提取结构化事件因子。

模块：{module}

字段要求：
- relevance: 0/1/2/3，文本与碳价预测的相关性；0=无关，1=弱相关，2=相关，3=高度相关。
- impact_direction: -2/-1/0/1/2，对碳配额价格/碳价方向的潜在影响；负数=利空碳价，正数=利多碳价，绝对值越大影响越强。注意：文本语气积极不等于利多碳价，例如CCER供给扩张、配额大量净卖出、价格指数下行、履约压力下降，都可能是利空。
- impact_strength: 0/1/2/3，经济影响强度；0=无，3=强。
- confidence: 0/1/2/3，判断置信度。
- decay_days: 1/3/5/10/20，影响可能持续的交易日长度。
- uncertainty: 0/1/2/3，是否增加政策或市场不确定性。

机制分数，每项范围 -2/-1/0/1/2。注意：所有机制分数的正负都表示“对碳配额价格/碳价的方向性影响”，不是表示该机制本身增加或减少。正值=通过该机制利多碳价；负值=通过该机制利空碳价；0=无明显影响。
- policy_tightening: 政策趋严/放松对碳价的影响；政策趋严、监管增强、扩围约束增强为正；政策放松、延期、约束弱化为负。
- allowance_supply: 配额供给变化对碳价的影响；配额收紧、供给减少为正；配额宽松、盈余、企业大量净卖出为负。
- compliance_pressure: 履约压力变化对碳价的影响；压力上升、清缴需求增强为正；压力下降、履约压力缓解为负。
- ccer_supply: CCER/抵消供给变化；CCER供给增加、方法学扩容、项目大量入市通常降低配额履约需求，对配额价格偏利空，应给负值；CCER供给受限/暂停/收紧通常偏利多，应给正值。
- trading_liquidity: 交易流动性/活跃度变化；交投活跃、成交放大可为正；成交清淡、流动性下降、市场低迷应为负。
- industrial_demand: 工业生产、排放需求、企业购买需求变化；生产/排放/履约购买需求增强为正，需求走弱、减排放缓但需求不足、企业净卖出压力为负。
- energy_macro_linkage: 能源、宏观、金融市场对碳价的联动影响。
- market_expectation: 市场预期、信心、风险偏好变化对碳价的影响；价格预期上行/买入意愿增强为正；价格指数下行/预期转弱/卖压增强为负。

输出JSON示例：
{{"relevance":2,"impact_direction":1,"impact_strength":2,"confidence":2,"decay_days":5,"uncertainty":1,"policy_tightening":1,"allowance_supply":0,"compliance_pressure":1,"ccer_supply":0,"trading_liquidity":1,"industrial_demand":0,"energy_macro_linkage":0,"market_expectation":1}}

文本：
{text}

JSON："""


def clip_text(text: str, max_chars: int = 2600) -> str:
    """Keep title/front matter plus tail. Daily concatenated texts can be very long."""
    text = str(text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.72)]
    tail = text[-int(max_chars * 0.20):]
    return head + "\n\n[中间长文本已省略，保留首尾关键信息]\n\n" + tail


def extract_json(s: str):
    s = re.sub(r"```json|```", "", str(s).strip()).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def normalize(obj):
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in FIELDS:
        try:
            v = int(round(float(obj.get(k, 0))))
        except Exception:
            v = 0
        if k == "relevance":
            v = max(0, min(3, v))
        elif k == "impact_direction":
            v = max(-2, min(2, v))
        elif k in ["impact_strength", "confidence", "uncertainty"]:
            v = max(0, min(3, v))
        elif k == "decay_days":
            allowed = np.array([1, 3, 5, 10, 20])
            v = int(allowed[np.argmin(np.abs(allowed - v))])
        else:
            v = max(-2, min(2, v))
        out[k] = v

    # If Qwen found strong mechanisms but forgot direction/strength, derive a weak non-zero signal.
    mech_sum = sum(out[m] for m in MECHANISMS)
    if out["relevance"] >= 2 and out["impact_strength"] == 0 and abs(mech_sum) > 0:
        out["impact_strength"] = min(3, max(1, abs(mech_sum) // 2))
    if out["relevance"] >= 2 and out["impact_direction"] == 0 and abs(mech_sum) >= 2:
        out["impact_direction"] = 1 if mech_sum > 0 else -1
    if out["relevance"] >= 2 and out["confidence"] == 0:
        out["confidence"] = 1
    return out


def semantic_signal(row):
    # scaled to [-1, 1]
    return (
        row["impact_direction"] / 2.0
        * row["impact_strength"] / 3.0
        * row["confidence"] / 3.0
        * row["relevance"] / 3.0
    )


def load_model(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading Qwen from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    mod = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        mod = mod.to(dev)
    print(f"Model ready on {dev}")
    return tok, mod, dev


def qwen_call(tok, mod, dev, prompt, max_new_tokens=220):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([txt], return_tensors="pt").to(dev)
    with torch.no_grad():
        ids = mod.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.03,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(ids[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)


def build_daily(raw_df):
    rows = []
    raw_df = raw_df.copy()
    raw_df["qef_signal"] = raw_df.apply(semantic_signal, axis=1)
    raw_df["qef_abs_signal"] = raw_df["qef_signal"].abs()
    raw_df["qef_pos_signal"] = raw_df["qef_signal"].clip(lower=0)
    raw_df["qef_neg_signal"] = (-raw_df["qef_signal"].clip(upper=0))
    raw_df["qef_active"] = ((raw_df["relevance"] >= 1) & (raw_df["impact_strength"] >= 1)).astype(int)

    dates = sorted(raw_df["date"].unique())
    for d in dates:
        sub = raw_df[raw_df["date"] == d]
        row = {"date": d}
        for m in MODULES:
            sm = sub[sub["module"] == m]
            prefix = f"qef_{m}"
            if sm.empty:
                for c in ["signal", "abs_signal", "pos_signal", "neg_signal", "relevance", "strength", "confidence", "uncertainty", "decay_days", "active"]:
                    row[f"{prefix}_{c}"] = 0.0
                for mech in MECHANISMS:
                    row[f"{prefix}_{mech}"] = 0.0
                continue
            # If multiple records/module/day, average semantic intensities; active takes max.
            row[f"{prefix}_signal"] = sm["qef_signal"].mean()
            row[f"{prefix}_abs_signal"] = sm["qef_abs_signal"].mean()
            row[f"{prefix}_pos_signal"] = sm["qef_pos_signal"].mean()
            row[f"{prefix}_neg_signal"] = sm["qef_neg_signal"].mean()
            row[f"{prefix}_relevance"] = sm["relevance"].mean()
            row[f"{prefix}_strength"] = sm["impact_strength"].mean()
            row[f"{prefix}_confidence"] = sm["confidence"].mean()
            row[f"{prefix}_uncertainty"] = sm["uncertainty"].mean()
            row[f"{prefix}_decay_days"] = sm["decay_days"].mean()
            row[f"{prefix}_active"] = sm["qef_active"].max()
            for mech in MECHANISMS:
                row[f"{prefix}_{mech}"] = sm[mech].mean() / 2.0
        sig_cols = [f"qef_{m}_signal" for m in MODULES]
        abs_cols = [f"qef_{m}_abs_signal" for m in MODULES]
        active_cols = [f"qef_{m}_active" for m in MODULES]
        vals = np.array([row[c] for c in sig_cols], dtype=float)
        row["qef_global_signal_mean"] = float(np.nanmean(vals))
        row["qef_global_signal_sum"] = float(np.nansum(vals))
        row["qef_global_abs_signal"] = float(sum(row[c] for c in abs_cols))
        row["qef_global_active_modules"] = float(sum(row[c] for c in active_cols))
        row["qef_global_disagreement"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--modules", nargs="*", default=MODULES)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-chars", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "qwen_event_raw.csv"
    daily_path = outdir / "qwen_event_daily.csv"

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    if args.max_days:
        df = df.head(args.max_days)

    existing = []
    done = set()
    if args.resume and raw_path.exists():
        old = pd.read_csv(raw_path)
        existing = old.to_dict("records")
        done = set(zip(old["date"].astype(str), old["module"].astype(str)))
        print(f"Resume: {len(done)} existing date-module records")

    tasks = []
    for _, r in df.iterrows():
        for m in args.modules:
            if m not in df.columns:
                continue
            text = r.get(m, "")
            if pd.isna(text) or len(str(text).strip()) < args.min_chars:
                continue
            key = (r["date"], m)
            if key not in done:
                tasks.append((r["date"], m, str(text)))
    print(f"Pending date-module tasks: {len(tasks)}")

    tok, mod, dev = load_model(args.model)
    results = existing

    for i, (date, module, text) in enumerate(tqdm(tasks, desc="Qwen event factors"), 1):
        prompt = USER_TEMPLATE.format(module=module, text=clip_text(text))
        norm = None
        raw_text = ""
        for attempt in range(3):
            try:
                raw_text = qwen_call(tok, mod, dev, prompt)
                norm = normalize(extract_json(raw_text))
                if norm is not None:
                    break
            except Exception as e:
                raw_text = f"ERROR: {e}"
        if norm is None:
            norm = {k: 0 for k in FIELDS}
            norm["decay_days"] = 1

        results.append({"date": date, "module": module, **norm, "raw_response": raw_text[:500]})
        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(raw_path, index=False)

    raw = pd.DataFrame(results)
    raw.to_csv(raw_path, index=False)
    daily = build_daily(raw)
    daily.to_csv(daily_path, index=False)

    print(f"Saved raw:   {raw_path} rows={len(raw)}")
    print(f"Saved daily: {daily_path} rows={len(daily)} cols={len(daily.columns)}")
    if len(raw):
        print("\nDistribution summary:")
        for c in ["relevance", "impact_direction", "impact_strength", "confidence", "decay_days"]:
            print(c, raw[c].value_counts(dropna=False).sort_index().to_dict())
        print("active rate", float(((raw["relevance"] >= 1) & (raw["impact_strength"] >= 1)).mean()))


if __name__ == "__main__":
    main()
