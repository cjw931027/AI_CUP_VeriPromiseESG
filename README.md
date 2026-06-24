# AI CUP 2026 — VeriPromiseESG 永續承諾驗證競賽

繁體中文 ESG 永續報告書段落的四任務分類系統。本倉庫為可重現之實作程式碼,供主辦單位查核得獎正當性。

**最終成績(Private Leaderboard):加權分數 0.6434421,Rank 11 / 143。**

| 子任務 | 權重 | macro-F1 |
|--------|------|----------|
| promise_status（承諾識別） | 20% | 0.795 |
| verification_timeline（驗證時機,4 類） | 15% | 0.591 |
| evidence_status（證據連結） | 30% | 0.700 |
| evidence_quality（證據清晰度,3 類） | 35% | 0.445 |

> 書面報告書(Word/PDF)依官方信件指引循另一管道繳交,不在本倉庫內。

---

## 方法概述

四任務具明確的階層邏輯相依(承諾為 No 時其餘欄位皆為 N/A)。系統採「共享編碼器 + 任務專屬輸出頭」之多任務架構,並以異質骨幹集成提升泛化:

- **異質骨幹集成**:`hfl/chinese-macbert-large`、`hfl/chinese-roberta-wwm-ext-large`、`hfl/chinese-lert-large`,外加一組以 MacBERT 訓練的「條件式階層」變體(下游頭不含 N/A、N/A 由機率階層分解還原)。每種骨幹各做 **5 折 StratifiedGroupKFold**,共 **4 × 5 = 20 個模型**,推論時轉回統一輸出空間做機率平均。
- **任務專屬雙路池化**:每任務同時取 `[CLS]` 與該任務專屬的 Attention Pooling,串接為 2H 維特徵。
- **級聯 logits**:上游任務(promise、evidence_status)logits 以縮放係數 0.1 串接進下游輸入(不切斷梯度),編碼任務階層。
- **Multi-Sample Dropout**:輸出頭以 5 次、丟棄率 0.3 取樣平均,提升小資料穩定度。
- **損失**:交叉熵 + 類別權重(上限 5)+ 標籤平滑 0.05;條件式變體用 masked CE。
- **防洩漏切分**:依「公司」分組之 StratifiedGroupKFold(seed=42),確保同公司段落不跨 train/val。僅以段落文本 `data` 為輸入,排除 `promise_string` / `evidence_string` 以防標籤洩漏。
- **後處理**:`LogicFixes` 強制輸出符合官方欄位邏輯;out-of-fold 上的 `log p − τ·log(prior)` τ 校正(緩解稀少類別被先驗壓低);以及年份規則(promise=Yes 且文本含明確未來年份時,依年份改寫 verification_timeline)。τ 校正與年份規則於 `reproduce_champion.py` 中套用。

---

## 執行環境

- Google Colab / Kaggle 雲端環境,Python 3.12
- 單張 NVIDIA T4 或 P100 GPU(16GB)
- 套件見 [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

---

## 資料

本倉庫**不含**官方競賽資料(著作權考量)。重現前請至 AIdea 平台下載 `train_2000.json` 與 `vpesg4k_test_2000.json`,放入 `reproduce_champion.py` 最上方 `DRIVE` 常數所指的資料夾(與權重、`adjust_champion.json` 同處)。系統僅使用主辦單位提供之官方標註資料,**未引入任何外部標註資料**。

---

## 一鍵重現

權威重現腳本為 **`reproduce_champion.py`**(冠軍實際配方);`esg_multitask_pro.py` 為它呼叫的引擎
(模型/訓練/推論/OOF/τ 校正)。腳本依序執行:訓練 4 骨幹 × 5 折 → 集成 OOF → 寫入 τ 校正 →
推論 + 年份規則後處理 → 官方格式驗證。

**步驟**
1. 把以下檔案放進【同一個資料夾】:官方 `train_2000.json`、`vpesg4k_test_2000.json`、
   20 個權重(`fold/rb/lt/mc` 各 5 折)、`adjust_champion.json`。
2. 修改 `reproduce_champion.py` 最上方的 `DRIVE` 常數,指向該資料夾。
3. 執行:
   ```bash
   python reproduce_champion.py
   ```
   - **已有 20 個權重(從下方 Google Drive 下載)→ 免重訓**:把 `__main__` 裡的 `train_all_groups()`
     註解掉,從 `build_oof()` 起跑(省去 3+ 小時),最終輸出 `submission_champion.csv`。

### 冠軍配方關鍵設定(已寫死於 `reproduce_champion.py`)
- **4 骨幹 × 5 折 = 20 模型**:`fold`=MacBERT(訓練時帶 ESG_type 前綴)、`rb`=RoBERTa-wwm、
  `lt`=LERT、`mc`=條件式 MacBERT 變體。推論時各權重自帶 backbone/conditional 旗標,集成自動還原。
- **max_len**:訓練 384、推論 512。
- **τ logit 校正**(LB 驗證之固定值,見 `adjust_champion.json`):
  promise 0.75 / timeline 0 / evidence_status 0.25 / evidence_quality 0.25。
- **年份規則後處理**:promise=Yes 且文本含明確未來年份(西元/民國)時依年份改寫 verification_timeline
  (max 年 > 2029 → more_than_5_years;≥ 2027 → between_2_and_5_years)。
- **決定論**:`CFG.seed=42`、StratifiedGroupKFold 同 seed → 折切分一致。重訓因 GPU/驅動隨機性,
  權重與分數會有微小浮動,屬正常(重現碼僅供查核正當性,不計入排名)。

**重要備註**
- 引擎各函式的預設路徑為 Colab 的 `/content/...`;`reproduce_champion.py` 已用最上方 `DRIVE` 常數統一覆蓋。
- 輸出 CSV 為 UTF-8(無 BOM)、LF 換行、5 欄(id, promise_status, verification_timeline,
  evidence_status, evidence_quality),腳本第 5 部分 `validate()` 會自動檢查。

### 訓練權重(20 個 `.pt`)

權重檔總量過大(遠超 GitHub 單檔 100MB 上限),存放於 Google Drive(連同 `adjust_champion.json`、
官方資料,構成可直接執行 `reproduce_champion.py` 的完整封裝):

> **下載連結（Google Drive,知道連結者可檢視）:**
> https://drive.google.com/drive/folders/1zB5SGvRuUnvfUHTMpmHQfTM8XC4Vgnhd?usp=sharing

---

## 生成式 AI 與外部資源使用揭露

- **冠軍管線未使用任何外部標註資料,亦未使用 LLM 生成訓練資料。**
- 曾評估開源模型 `Qwen2.5-7B-Instruct` 作為證據清晰度任務之零樣本分類,惟表現未優於自訓集成,**未納入最終提交**。
- 開發與除錯過程使用 **Anthropic Claude** 輔助撰寫/除錯程式、釐清評估指標口徑與實驗設計討論。最終所有預測均由本隊自訓模型之確定性推論流程產生,無人工修改預測值。
- 團隊與 AI 之貢獻比例詳見另交之書面報告書。
