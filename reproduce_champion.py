# -*- coding: utf-8 -*-
"""
====================================================================
ESG 永續承諾驗證競賽 2026 — 冠軍配置「完整重訓 + 重現」驅動腳本
====================================================================
用途：競賽交件用。只給 train_2000.json，從零重跑出冠軍 submission：
      訓練 4 組 backbone × 5 折 → ensemble OOF → logit 校正 → 年份規則 → 5 欄輸出 → 驗證。

LB 紀錄：weighted_score = 0.6129
        promise 0.795 / timeline 0.5889 / status 0.6997 / quality 0.4447

【執行前提】
  1. 本腳本是「驅動程式」，模型/訓練/OOF/校正/推論的引擎在 esg_multitask_pro.py，
     兩個檔案要放在同一資料夾（本腳本用 from esg_multitask_pro import ...）。
  2. 交件壓縮包應包含：
       - esg_multitask_pro.py（引擎，本腳本的依賴）
       - reproduce_champion.py（本檔）
       - train_2000.json（訓練資料）
       - 20 個權重 fold0-4 / rb0-4 / lt0-4 / mc0-4 .pt（若要求附權重）
       - 生成式 AI 使用聲明（見最後）
  3. 決定論：CFG.seed=42，run_cv 內部會 set_seed；StratifiedGroupKFold 同 seed →
     折切分完全一致，可重現。

【誠實標註 ⚠️】
  - 「第 1 部分：訓練」的每組 CFG 是依交接紀錄的最佳重建。下列三項我無法百分百
    驗證，請對照你當初的實際訓練設定確認後再正式重跑：
       (a) 每組的 conditional 旗標     (b) 每組的 use_esg_type
       (c) 每組的 backbone 名稱與 epochs
  - 「第 2~5 部分」(OOF→校正→推論→輸出→驗證) 為已驗證的精確流程。
  - 算力警告：4 組 × 5 折 = 20 個 large 級模型，T4 上單折數十分鐘 → 完整重訓需數天。
    交件驗證通常以「程式可重現 + 附權重」為準，未必要評審重跑全部。
"""

import os, sys, glob, json, re
import numpy as np

# 直接從 Drive 載入引擎（依你的工作環境調整）
DRIVE = "/content/drive/MyDrive/esg_final"
sys.path.insert(0, DRIVE)

import esg_multitask_pro as E
from esg_multitask_pro import CFG, run_cv, run_oof, run_infer, TASKS, num_labels

# 路徑
TRAIN_PATH = f"{DRIVE}/train_2000.json"
TEST_PATH  = f"{DRIVE}/vpesg4k_test_2000.json"
CFG.ckpt_dir = DRIVE          # 權重直接落 Drive
CFG.seed     = 42             # 決定論
CFG.amp      = True

# ====================================================================
# 第 1 部分：訓練 4 組 backbone（每組 5 折）
# ====================================================================
# (ckpt_prefix, model_name, conditional, use_esg_type, epochs)
# ⚠️ 確認這四組設定跟你當初實際訓練一致（尤其 fold 的 use_esg_type=True）
GROUPS = [
    ("fold", "hfl/chinese-macbert-large",         False, True,  10),  # ⚠️ 帶 ESG_type 前綴訓的
    ("rb",   "hfl/chinese-roberta-wwm-ext-large", False, False, 10),  # ⚠️
    ("lt",   "hfl/chinese-lert-large",            False, False, 10),  # ⚠️
    ("mc",   "hfl/chinese-macbert-large",         True,  False, 10),  # ⚠️ conditional 條件式架構
]

def train_all_groups():
    """逐組重訓。已存在的權重會被覆寫；想跳過已訓好的組請自行註解。"""
    for prefix, mname, cond, use_esg, ep in GROUPS:
        print(f"\n{'='*60}\n訓練組 {prefix} | {mname} | conditional={cond} | "
              f"use_esg_type={use_esg} | epochs={ep}\n{'='*60}")
        CFG.ckpt_prefix  = prefix
        CFG.model_name   = mname
        CFG.conditional  = cond           # 須在建構模型前設定（影響頭尺寸）
        CFG.use_esg_type = use_esg
        CFG.epochs       = ep
        CFG.max_len      = 384            # 訓練長度（證據幾乎都在前 138 字，384 已足夠）
        CFG.batch_size   = 4
        CFG.grad_accum_steps = 4
        # 進階開關全關（實證：Focal+Sampler+EMA+FGM 同開 → CV 崩到 0.31）
        CFG.use_focal = CFG.use_fgm = CFG.use_ema = CFG.use_sampler = False
        run_cv(data_path=TRAIN_PATH)      # 產生 {prefix}0.pt ... {prefix}4.pt + 印各折分數

# ====================================================================
# 第 2 部分：ensemble OOF（誠實、無洩漏；用對齊官方的 official_macro 計分）
# ====================================================================
OOF_PATH = f"{DRIVE}/oof_ensemble.npz"

def build_oof():
    CFG.use_esg_type = False              # 推論/OOF 一律不加前綴（對齊測試集）
    CFG.max_len      = 512                # 推論長度（冠軍設定）
    CFG.batch_size   = 8
    CFG.model_name   = "hfl/chinese-macbert-large"   # 只是 fallback；每個 ckpt 用自己記錄的 backbone
    CFG.conditional  = False
    run_oof(data_path=TRAIN_PATH,
            prefixes=("fold", "rb", "lt", "mc"),
            save_path=OOF_PATH)           # 內部印「官方口徑」加權 OOF（基準線 ~0.6019）

# ====================================================================
# 第 3 部分：logit 校正參數（冠軍 taus；決定論寫死，不重新搜尋）
# ====================================================================
# 注意：status tau=0.25 是 LB 驗證過的冠軍值。
#       曾試 status tau=0.0（OOF +0.0008）但 LB 反退 -0.0038 → 棄用，鎖回 0.25。
ADJUST_PATH = f"{DRIVE}/adjust_champion.json"
CHAMPION_TAUS = {"promise_status": 0.75, "verification_timeline": 0.0,
                 "evidence_status": 0.25, "evidence_quality": 0.25}

def write_adjust():
    z = np.load(OOF_PATH)                 # priors 由 train_2000 標籤頻率確定性重算
    priors = {}
    for f in TASKS:
        cnt = np.bincount(z[f"gold_{f}"], minlength=num_labels[f]).astype(float)
        priors[f] = np.clip(cnt / cnt.sum(), 1e-6, None).tolist()
    json.dump({"taus": CHAMPION_TAUS, "priors": priors},
              open(ADJUST_PATH, "w", encoding="utf-8"))
    print(f"已寫 {ADJUST_PATH}：taus={CHAMPION_TAUS}")

# ====================================================================
# 第 4 部分：推論 → 年份規則後處理 → 只留 5 欄 → 輸出 submission
# ====================================================================
TMP_CSV   = f"{DRIVE}/_tmp_infer.csv"
FINAL_CSV = f"{DRIVE}/submission_champion.csv"
COLS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

def _future_years(t):
    """文本中 >2024 的未來年份（含民國轉西元：民國年 + 1911）。"""
    ys = [int(m) for m in re.findall(r'(?<!\d)(\d{4})(?!\d)', t) if 2024 < int(m) < 2100]
    ys += [int(m) + 1911 for m in re.findall(r'民國\s*(\d{2,3})', t) if 2024 < int(m) + 1911 < 2100]
    return ys

def infer_and_postprocess():
    import pandas as pd
    CFG.use_esg_type = False
    CFG.max_len      = 512
    CFG.batch_size   = 8
    ckpts = sorted(sum([glob.glob(f"{DRIVE}/{p}*.pt") for p in ("fold", "rb", "lt", "mc")], []))
    assert len(ckpts) == 20, f"權重數應為 20，實際 {len(ckpts)}"

    # 4-1 ensemble 推論 + logit 校正 + LogicFixes（由 run_infer 完成）
    run_infer(test_path=TEST_PATH, out_path=TMP_CSV, ckpts=ckpts, adjust_path=ADJUST_PATH)

    # 4-2 年份規則：promise=Yes 且含未來年份 → 改寫 timeline
    df = pd.read_csv(TMP_CSV, dtype=str, keep_default_na=False)
    txt = {str(r["id"]): r["data"] for r in json.load(open(TEST_PATH, encoding="utf-8"))}
    n = 0
    for i, row in df.iterrows():
        if row["promise_status"] != "Yes":
            continue
        ys = _future_years(txt.get(str(row["id"]), ""))
        if not ys:
            continue
        new = "more_than_5_years" if max(ys) > 2029 else \
              ("between_2_and_5_years" if max(ys) >= 2027 else None)
        if new and df.at[i, "verification_timeline"] != new:
            df.at[i, "verification_timeline"] = new
            n += 1
    print(f"年份規則改寫 {n} 列")

    # 4-3 只留官方規定的 5 欄、固定順序，輸出
    df[COLS].to_csv(FINAL_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"→ {FINAL_CSV}")

# ====================================================================
# 第 5 部分：提交前格式 + 邏輯驗證（全綠才可上傳）
# ====================================================================
def validate():
    import pandas as pd
    df = pd.read_csv(FINAL_CSV, dtype=str, keep_default_na=False)
    ok = True
    def chk(c, m):
        nonlocal ok
        print(("✓" if c else "✗"), m); ok = ok and c

    chk(list(df.columns) == COLS, "5 欄與順序正確")
    chk(len(df) == 2000, f"列數=2000（{len(df)}）")
    chk([int(x) for x in df["id"]] == list(range(12001, 14001)), "id 12001~14000 連續")
    LAB = {"promise_status": {"Yes", "No"},
           "verification_timeline": {"already", "within_2_years", "between_2_and_5_years",
                                     "more_than_5_years", "N/A"},
           "evidence_status": {"Yes", "No", "N/A"},
           "evidence_quality": {"Clear", "Not Clear", "Misleading", "N/A"}}
    for c in LAB:
        chk(set(df[c]) <= LAB[c], f"{c} 值合法")
    chk(not (df.values == "").any() and not df.isna().any().any(), "無空值/NaN")
    no = df["promise_status"] == "No"
    chk((df.loc[no, COLS[2:]] == "N/A").all().all(), "promise=No → 其餘全 N/A")
    chk(((df["evidence_status"] != "Yes") == (df["evidence_quality"] == "N/A")).all(),
        "quality 非 N/A ⟺ evidence=Yes")
    raw = open(FINAL_CSV, "rb").read()
    chk(not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw, "UTF-8 no-BOM + LF")
    print("\n可上傳" if ok else "\n有問題，先別傳")

# ====================================================================
# 主流程
# ====================================================================
if __name__ == "__main__":
    # 完整重現（從零）：四步依序跑
    train_all_groups()      # ← 數天（T4）。已有權重要重現「輸出」可註解此行，直接從 build_oof 開始
    build_oof()
    write_adjust()
    infer_and_postprocess()
    validate()

# ====================================================================
# 生成式 AI 使用聲明（草稿；請貼到交件文件並依實際情況微調）
# ====================================================================
"""
【生成式 AI 使用聲明】

本隊於開發過程中使用 Anthropic Claude 作為程式協作與除錯輔助，涵蓋：
  - 程式碼撰寫、重構與錯誤排除（OOM、AMP、CUDA、資料處理等）
  - 競賽評估指標（Macro-F1 口徑）之釐清與本地評分對齊
  - 實驗設計討論與結果判讀（交叉驗證、logit 校正、後處理規則）

模型方法評估：
  - 最終提交之模型為自行訓練的 MacBERT-large / RoBERTa-wwm-ext-large /
    LERT-large / MacBERT(條件式) 之 5 折交叉驗證 ensemble，未使用任何外部
    大型語言模型之輸出作為提交內容。
  - 曾評估開源模型 Qwen2.5-7B-Instruct 作為 evidence_quality 任務之零樣本
    評審，經驗證其表現未優於自訓模型，故未納入最終提交。

所有提交之預測結果均由上述自訓模型之確定性推論流程產生，程式碼可完整重現，
無任何人為修改預測值之情形。
"""
