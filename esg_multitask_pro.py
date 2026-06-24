# -*- coding: utf-8 -*-
"""
AI CUP 2026 VeriPromiseESG — 繁中 ESG 段落四任務分類【引擎】
（冠軍模型,Private LB 0.6434 / Rank 11;完整重現見 reproduce_champion.py 與 README.md）

本檔提供:模型、訓練(run_cv)、集成推論(run_infer)、OOF 評估(run_oof)、
τ logit 校正(tune_adjustment)等元件,由 reproduce_champion.py 依冠軍配方逐一呼叫
(4 種骨幹 × 5 折 StratifiedGroupKFold = 20 模型;訓練 max_len 384、推論 512;
 年份規則等後處理在 driver 中完成)。

核心設計:MacBERT/RoBERTa-wwm/LERT + 條件式變體、僅用 data 文本(防輸入洩漏)、
雙路池化(CLS⊕AttnPool)、級聯 logits、Multi-Sample Dropout、可微分 Constraint Loss、
防洩漏「依公司」分組切分、LogicFixes 後處理、CSV(UTF-8 無 BOM + LF)。

★ 路徑:所有函式預設路徑為 Colab 的 /content/...;在其他環境執行請顯式傳入
  data_path / test_path / out_path 並設定 CFG.ckpt_dir(見 README.md)。
★ 環境:單張 GPU(T4/P100, 16GB)、Python 3.12;套件見 requirements.txt;
  首次執行會自 Hugging Face 下載骨幹(hfl/chinese-macbert-large 等),須連網。
"""

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score


# ============================================================
# 1. Config
# ============================================================
class CFG:
    model_name = "hfl/chinese-macbert-large"
    max_len = 384
    batch_size = 4
    grad_accum_steps = 4
    epochs = 10
    early_stop_patience = 3
    lr = 1.5e-5
    head_lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    grad_clip = 0.5
    dropout = 0.3
    n_msdo = 5
    cascade_scale = 0.1
    constraint_weight = 0.3
    max_class_weight = 5.0
    use_esg_type = False         # 官方測試集無 ESG_type 欄位 → 關閉前綴，消除 train/test 不一致

    # 進階開關（冠軍未使用,預設全關;曾單獨 A/B 皆無增益）
    n_folds = 5                  # 交叉驗證折數（=ensemble 模型數）
    use_focal = False
    focal_gamma = 2.0
    use_fgm = False              # FGM 對抗訓練
    fgm_eps = 1.0
    use_ema = False              # 權重指數移動平均
    ema_decay = 0.99
    use_sampler = False          # 稀少類別過採樣
    sampler_tasks = ["verification_timeline", "evidence_quality", "evidence_status"]

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    ckpt_dir = "/content"
    ckpt_prefix = "fold"

    # N/A 字串：官方規格列 "N/A" 為類別 → 預設 "N/A"；若正式 JSON 用空字串請改 ""
    output_na_token = "N/A"

    # --- 實驗C：條件式階層訓練 ---
    # True 時：timeline/evidence_status/evidence_quality 的頭【不含 N/A 類】，
    # loss 只在 gating 條件成立的樣本上計算（timeline/ev_status 限 promise=Yes；
    # quality 限 evidence_status=Yes）。N/A 由機率階層分解精確處理，Constraint Loss 停用。
    # 注意：conditional 與舊權重的頭尺寸不同，須在【建構/載入模型前】設好；
    # 權重檔會記錄此旗標，ensemble 可新舊混用（皆轉回輸出空間機率再平均）。
    conditional = False


EVAL_FIELDS = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
    "evidence_status":       ["Yes", "No", "N/A"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
}
# 條件式模式下各任務「頭」的類別空間（無 N/A）
HEAD_FIELDS_COND = {
    "promise_status":        ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
    "evidence_status":       ["Yes", "No"],
    "evidence_quality":      ["Clear", "Not Clear", "Misleading"],
}
FIELD_WEIGHTS = {"promise_status": 0.20, "verification_timeline": 0.15,
                 "evidence_status": 0.30, "evidence_quality": 0.35}
TASKS = list(EVAL_FIELDS.keys())
label2id = {f: {lab: i for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
id2label = {f: {i: lab for i, lab in enumerate(labs)} for f, labs in EVAL_FIELDS.items()}
num_labels = {f: len(labs) for f, labs in EVAL_FIELDS.items()}
NA_IDX = {f: (label2id[f]["N/A"] if "N/A" in label2id[f] else -1) for f in TASKS}

# 頭空間（依 CFG.conditional 動態）：conditional=True 用 HEAD_FIELDS_COND，否則同輸出空間
def head_fields():
    return HEAD_FIELDS_COND if CFG.conditional else EVAL_FIELDS

def head_num_labels():
    return {f: len(labs) for f, labs in head_fields().items()}

def head_label2id():
    return {f: {lab: i for i, lab in enumerate(labs)} for f, labs in head_fields().items()}


def to_output_probs(logits, conditional):
    """
    頭空間 logits → 輸出空間機率（含 N/A 欄），讓新舊模型可在同一空間做 ensemble。
    conditional=True 時用機率階層分解：
      P(timeline=c)   = P(promise=Yes)·P4(c)；P(timeline=N/A)   = P(promise=No)
      P(ev_status=Yes)= P(promY)·Pe(Yes) 等； P(ev_status=N/A)  = P(promise=No)
      P(quality=c)    = P(promY)·Pe(Yes)·P3(c)；P(quality=N/A)  = 1 - P(promY)·Pe(Yes)
    conditional=False 時即各任務 softmax（頭空間=輸出空間）。
    """
    if not conditional:
        return {f: torch.softmax(logits[f].float(), -1) for f in TASKS}

    hl2i = {f: {lab: i for i, lab in enumerate(HEAD_FIELDS_COND[f])} for f in TASKS}
    pP = torch.softmax(logits["promise_status"].float(), -1)            # [B,2] 頭=輸出
    p_yes = pP[:, hl2i["promise_status"]["Yes"]]
    p_no  = pP[:, hl2i["promise_status"]["No"]]
    out = {"promise_status": pP}

    B = pP.size(0)
    # timeline
    p4 = torch.softmax(logits["verification_timeline"].float(), -1)     # [B,4]
    t = torch.zeros(B, num_labels["verification_timeline"])
    for lab, hi in hl2i["verification_timeline"].items():
        t[:, label2id["verification_timeline"][lab]] = p_yes * p4[:, hi]
    t[:, NA_IDX["verification_timeline"]] = p_no
    out["verification_timeline"] = t
    # evidence_status
    pe = torch.softmax(logits["evidence_status"].float(), -1)           # [B,2]
    e = torch.zeros(B, num_labels["evidence_status"])
    for lab, hi in hl2i["evidence_status"].items():
        e[:, label2id["evidence_status"][lab]] = p_yes * pe[:, hi]
    e[:, NA_IDX["evidence_status"]] = p_no
    out["evidence_status"] = e
    # quality
    gate = p_yes * pe[:, hl2i["evidence_status"]["Yes"]]                # P(promY ∧ evY)
    p3 = torch.softmax(logits["evidence_quality"].float(), -1)          # [B,3]
    q = torch.zeros(B, num_labels["evidence_quality"])
    for lab, hi in hl2i["evidence_quality"].items():
        q[:, label2id["evidence_quality"][lab]] = gate * p3[:, hi]
    q[:, NA_IDX["evidence_quality"]] = 1.0 - gate
    out["evidence_quality"] = q
    return out


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def norm_label(field, raw):
    s = ("" if raw is None else str(raw)).strip()
    return "N/A" if s == "" else s


def get_ci(sample, *keys):
    """大小寫/別名容錯取值（官方 ESG_type vs 樣本 esg_type、URL vs pdf_url）。"""
    lower = {k.lower(): v for k, v in sample.items()}
    for k in keys:
        if k in sample and sample[k] not in (None, ""):
            return sample[k]
        if k.lower() in lower and lower[k.lower()] not in (None, ""):
            return lower[k.lower()]
    return ""


def group_key(sample, idx):
    """防洩漏分組單位：company → URL → ticker → company_source → 唯一id。"""
    for keys in (("company",), ("URL", "pdf_url", "url"), ("ticker",), ("company_source",)):
        v = get_ci(sample, *keys)
        if str(v).strip():
            return f"{keys[0]}:{v}"
    return f"__id_{sample.get('id', idx)}"


# ============================================================
# 2. Dataset
# ============================================================
class ESGDataset(Dataset):
    def __init__(self, data, tokenizer, with_labels=True):
        self.data, self.tok, self.with_labels = data, tokenizer, with_labels

    def __len__(self):
        return len(self.data)

    def _build_text(self, s):
        text = s.get("data", "") or ""
        if CFG.use_esg_type:
            et = str(get_ci(s, "ESG_type", "esg_type")).strip()
            if et:
                text = f"[{et}] {text}"
        return text

    def __getitem__(self, idx):
        s = self.data[idx]
        enc = self.tok(self._build_text(s), truncation=True, max_length=CFG.max_len,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.with_labels:
            item["labels"] = {f: torch.tensor(label2id[f][norm_label(f, s.get(f, ""))],
                                               dtype=torch.long) for f in TASKS}
        return item


def collate_fn(batch):
    out = {"input_ids": torch.stack([b["input_ids"] for b in batch]),
           "attention_mask": torch.stack([b["attention_mask"] for b in batch])}
    if "labels" in batch[0]:
        out["labels"] = {f: torch.stack([b["labels"][f] for b in batch]) for f in TASKS}
    return out


# ============================================================
# 3. 模型（雙路池化 + 級聯 logits + Multi-Sample Dropout）
# ============================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, hs, attn_mask):
        scores = self.scorer(hs).squeeze(-1).masked_fill(attn_mask == 0, -1e4)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hs * w).sum(dim=1)


class TaskHead(nn.Module):
    def __init__(self, in_dim, n_class, dropout, n_msdo):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_msdo)])
        self.fc = nn.Linear(in_dim, n_class)

    def forward(self, feat):
        return torch.stack([self.fc(do(feat)) for do in self.dropouts], dim=0).mean(dim=0)


class MultiTaskMacBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.model_name)
        H = self.backbone.config.hidden_size
        self.pools = nn.ModuleDict({f: AttentionPool(H) for f in TASKS})
        feat_dim = 2 * H
        hn = head_num_labels()                       # 頭空間尺寸（conditional 時無 N/A）
        in_dims = {
            "promise_status":        feat_dim,
            "verification_timeline": feat_dim + hn["promise_status"],
            "evidence_status":       feat_dim + hn["promise_status"],
            "evidence_quality":      feat_dim + hn["evidence_status"],
        }
        self.heads = nn.ModuleDict({f: TaskHead(in_dims[f], hn[f],
                                                CFG.dropout, CFG.n_msdo) for f in TASKS})

    def _fuse(self, task, seq_out, cls_vec, attn_mask):
        return torch.cat([cls_vec, self.pools[task](seq_out, attn_mask)], dim=-1)

    def forward(self, input_ids, attention_mask):
        seq_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vec = seq_out[:, 0]; s = CFG.cascade_scale; logits = {}
        logits["promise_status"] = self.heads["promise_status"](
            self._fuse("promise_status", seq_out, cls_vec, attention_mask))
        p = logits["promise_status"] * s
        logits["verification_timeline"] = self.heads["verification_timeline"](
            torch.cat([self._fuse("verification_timeline", seq_out, cls_vec, attention_mask), p], -1))
        logits["evidence_status"] = self.heads["evidence_status"](
            torch.cat([self._fuse("evidence_status", seq_out, cls_vec, attention_mask), p], -1))
        e = logits["evidence_status"] * s
        logits["evidence_quality"] = self.heads["evidence_quality"](
            torch.cat([self._fuse("evidence_quality", seq_out, cls_vec, attention_mask), e], -1))
        return logits


# ============================================================
# 4. Focal Loss + Constraint Loss
# ============================================================
class FocalLoss(nn.Module):
    """多分類 Focal Loss，含類別權重 alpha；gamma 放大難例/稀少類別。"""
    def __init__(self, weight, gamma):
        super().__init__()
        self.weight = weight; self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return ((1.0 - pt) ** self.gamma * ce).mean()


def constraint_loss(logits):
    p = {f: torch.softmax(logits[f], dim=-1) for f in TASKS}
    p_no = p["promise_status"][:, label2id["promise_status"]["No"]]
    es_no = p["evidence_status"][:, label2id["evidence_status"]["No"]]
    gap = lambda f: 1.0 - p[f][:, NA_IDX[f]]
    return (p_no * gap("verification_timeline") + p_no * gap("evidence_status")
            + p_no * gap("evidence_quality") + es_no * gap("evidence_quality")).mean()


def build_class_weights(train_data):
    """頭空間類別權重。conditional 時各任務只統計 gating 子集
    （timeline/ev_status 限 promise=Yes；quality 限 evidence_status=Yes）。"""
    hf = head_fields(); hl2i = head_label2id()
    weights = {}
    for f in TASKS:
        if CFG.conditional and f != "promise_status":
            if f == "evidence_quality":
                sub = [x for x in train_data if norm_label("evidence_status", x.get("evidence_status", "")) == "Yes"]
            else:
                sub = [x for x in train_data if norm_label("promise_status", x.get("promise_status", "")) == "Yes"]
            cnt = Counter(hl2i[f][norm_label(f, x.get(f, ""))] for x in sub
                          if norm_label(f, x.get(f, "")) in hl2i[f])
        else:
            cnt = Counter(hl2i[f][norm_label(f, x.get(f, ""))] for x in train_data
                          if norm_label(f, x.get(f, "")) in hl2i[f])
        n = len(hf[f]); total = max(1, sum(cnt.values())); w = torch.ones(n)
        for i in range(n):
            c = cnt.get(i, 0)
            w[i] = (total / (n * c)) if c > 0 else CFG.max_class_weight
        w = torch.clamp(w, 1.0, CFG.max_class_weight)
        weights[f] = w / w.mean()
    return weights


class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.conditional = CFG.conditional
        if CFG.use_focal:
            self.crit = {f: FocalLoss(class_weights[f].to(CFG.device), CFG.focal_gamma) for f in TASKS}
        else:
            self.crit = {f: nn.CrossEntropyLoss(weight=class_weights[f].to(CFG.device),
                                                label_smoothing=0.05) for f in TASKS}
        if self.conditional:
            # 輸出空間 label id → 頭空間 id（N/A → -1，會被 mask 掉）
            hl2i = head_label2id()
            self.map = {f: torch.tensor([hl2i[f].get(lab, -1) for lab in EVAL_FIELDS[f]],
                                        dtype=torch.long) for f in TASKS}
            self.w = {f: class_weights[f].to(CFG.device) for f in TASKS}

    def forward(self, logits, labels, cstr_w):
        if not self.conditional:
            task_loss = sum(FIELD_WEIGHTS[f] * self.crit[f](logits[f], labels[f]) for f in TASKS)
            cstr = constraint_loss(logits)
            return task_loss + cstr_w * cstr, task_loss.detach(), cstr.detach()

        # --- 條件式：masked CE，N/A 不參與，gating 用「黃金標籤」（teacher forcing）---
        yes_p = label2id["promise_status"]["Yes"]
        yes_e = label2id["evidence_status"]["Yes"]
        masks = {
            "promise_status":        torch.ones_like(labels["promise_status"], dtype=torch.float),
            "verification_timeline": (labels["promise_status"] == yes_p).float(),
            "evidence_status":       (labels["promise_status"] == yes_p).float(),
            "evidence_quality":      (labels["evidence_status"] == yes_e).float(),
        }
        task_loss = 0.0
        for f in TASKS:
            head_y = self.map[f].to(labels[f].device)[labels[f]]      # -1 = masked
            m = masks[f] * (head_y >= 0).float()
            if m.sum() < 1:
                continue
            ce = F.cross_entropy(logits[f], head_y.clamp(min=0), weight=self.w[f],
                                 label_smoothing=0.05, reduction="none")
            task_loss = task_loss + FIELD_WEIGHTS[f] * (ce * m).sum() / m.sum()
        zero = torch.zeros((), device=logits["promise_status"].device)
        return task_loss, (task_loss.detach() if torch.is_tensor(task_loss) else zero), zero


# ============================================================
# 5. FGM 對抗訓練 + EMA
# ============================================================
class FGM:
    """擾動詞向量做對抗訓練：attack→二次forward/backward→restore。"""
    def __init__(self, model, eps):
        self.model = model; self.eps = eps; self.backup = {}
        self.emb_name = None
        for n, _ in model.named_parameters():
            if "word_embeddings" in n:
                self.emb_name = n; break

    def attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n == self.emb_name and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ============================================================
# 6. LogicFixes（可吃 logits 或機率，masked argmax）
# ============================================================
def _masked_argmax(row, allowed):
    m = torch.full_like(row, float("-inf"))
    for i in allowed:
        m[i] = row[i]
    return int(torch.argmax(m).item())


def apply_logic_fixes(scores):
    B = scores["promise_status"].size(0)
    out = {f: torch.zeros(B, dtype=torch.long) for f in TASKS}
    real = {f: [i for i in range(num_labels[f]) if i != NA_IDX[f]] for f in TASKS}
    for b in range(B):
        p = int(torch.argmax(scores["promise_status"][b]).item())
        out["promise_status"][b] = p
        if p == label2id["promise_status"]["No"]:
            out["verification_timeline"][b] = NA_IDX["verification_timeline"]
            out["evidence_status"][b] = NA_IDX["evidence_status"]
            out["evidence_quality"][b] = NA_IDX["evidence_quality"]
            continue
        out["verification_timeline"][b] = _masked_argmax(
            scores["verification_timeline"][b], real["verification_timeline"])
        es = _masked_argmax(scores["evidence_status"][b], real["evidence_status"])
        out["evidence_status"][b] = es
        if es == label2id["evidence_status"]["No"]:
            out["evidence_quality"][b] = NA_IDX["evidence_quality"]
        else:
            out["evidence_quality"][b] = _masked_argmax(
                scores["evidence_quality"][b], real["evidence_quality"])
    return out


# ============================================================
# 7. 訓練元件
# ============================================================
def make_optimizer(model):
    bb, head = [], []
    for n, pr in model.named_parameters():
        if pr.requires_grad:
            (bb if n.startswith("backbone") else head).append(pr)
    return torch.optim.AdamW([
        {"params": bb, "lr": CFG.lr, "weight_decay": CFG.weight_decay},
        {"params": head, "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}])


def build_sampler(train_data, class_weights):
    """依稀少類別過採樣：樣本權重 = 其在指定任務中所屬類別權重的最大值。"""
    ws = []
    for x in train_data:
        m = 1.0
        for f in CFG.sampler_tasks:
            idx = label2id[f][norm_label(f, x.get(f, ""))]
            m = max(m, float(class_weights[f][idx]))
        ws.append(m)
    return WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds = {f: [] for f in TASKS}; golds = {f: [] for f in TASKS}
    for batch in loader:
        ids = batch["input_ids"].to(CFG.device); mask = batch["attention_mask"].to(CFG.device)
        with torch.amp.autocast("cuda", enabled=CFG.amp):
            logits = model(ids, mask)
        fixed = apply_logic_fixes(to_output_probs(logits, CFG.conditional))
        for f in TASKS:
            preds[f].extend(fixed[f].tolist()); golds[f].extend(batch["labels"][f].tolist())
    score, detail = 0.0, {}
    for f in TASKS:
        mf1 = f1_score(golds[f], preds[f], average="macro", zero_division=0)
        detail[f] = mf1; score += FIELD_WEIGHTS[f] * mf1
    return score, detail


def train_one_fold(train_data, val_data, tok, ckpt_path):
    tr_ds = ESGDataset(train_data, tok); va_ds = ESGDataset(val_data, tok)
    cw = build_class_weights(train_data)
    if CFG.use_sampler:
        tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size,
                               sampler=build_sampler(train_data, cw), collate_fn=collate_fn)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskMacBERT().to(CFG.device)
    criterion = MultiTaskLoss(cw)
    optimizer = make_optimizer(model)
    total_steps = math.ceil(len(tr_loader) / CFG.grad_accum_steps) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * CFG.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.amp)
    fgm = FGM(model, CFG.fgm_eps) if CFG.use_fgm else None
    ema = EMA(model, CFG.ema_decay) if CFG.use_ema else None

    best, bad, gstep = -1.0, 0, 0
    for epoch in range(CFG.epochs):
        model.train(); optimizer.zero_grad(); rl = rt = rc = 0.0
        for step, batch in enumerate(tr_loader):
            ids = batch["input_ids"].to(CFG.device); mask = batch["attention_mask"].to(CFG.device)
            labels = {f: batch["labels"][f].to(CFG.device) for f in TASKS}
            cstr_w = 0.0 if CFG.conditional else \
                CFG.constraint_weight * min(1.0, gstep / max(1, int(0.3 * total_steps)))

            with torch.amp.autocast("cuda", enabled=CFG.amp):
                logits = model(ids, mask)
                loss, l_task, l_cstr = criterion(logits, labels, cstr_w)
                loss = loss / CFG.grad_accum_steps
            scaler.scale(loss).backward()
            rl += loss.item() * CFG.grad_accum_steps; rt += l_task.item(); rc += l_cstr.item()

            if fgm is not None:                       # 對抗訓練：擾動 emb 再反傳一次
                fgm.attack()
                with torch.amp.autocast("cuda", enabled=CFG.amp):
                    adv_logits = model(ids, mask)
                    adv_loss, _, _ = criterion(adv_logits, labels, cstr_w)
                    adv_loss = adv_loss / CFG.grad_accum_steps
                scaler.scale(adv_loss).backward()
                fgm.restore()

            if (step + 1) % CFG.grad_accum_steps == 0 or (step + 1) == len(tr_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                # AMP 偵測到 inf 會跳過 optimizer step → 此時不要走 scheduler（消除警告）
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                gstep += 1
                if ema is not None:
                    ema.update(model)

        if ema is not None:
            ema.apply_to(model)                       # 用 EMA 權重評估/存檔
        score, detail = evaluate(model, va_loader)
        nb = len(tr_loader)
        print(f"  [E{epoch+1}/{CFG.epochs}] loss={rl/nb:.4f}(task={rt/nb:.4f} cstr={rc/nb:.4f}) "
              f"valW-F1={score:.4f} | " + " ".join(f"{f[:4]}:{detail[f]:.3f}" for f in TASKS))
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "model_name": CFG.model_name,
                        "conditional": CFG.conditional}, ckpt_path)
            print(f"    ✓ best={best:.4f} → {ckpt_path}")
        else:
            bad += 1
        if ema is not None:
            ema.restore(model)                        # 還原原始權重續訓
        if bad >= CFG.early_stop_patience:
            print(f"    ⏹ early stop"); break
    return best


# ============================================================
# 8. 交叉驗證訓練（產生 fold0..foldK 權重）
# ============================================================
def run_cv(data_path="/content/vpesg_4k_train_1000.json", pseudo_path=None):
    """pseudo_path：make_pseudo() 產生的偽標籤 JSON。偽標籤【只併入各折 train】，
    絕不進 val —— 折內 CV 分數仍然只以真實標籤計算，維持誠實可比。"""
    set_seed(CFG.seed)
    data = json.load(open(data_path, encoding="utf-8"))
    pseudo = []
    if pseudo_path:
        pseudo = json.load(open(pseudo_path, encoding="utf-8"))
        print(f"[pseudo] 載入偽標籤 {len(pseudo)} 筆（只進 train fold，不進 val）")
    tok = AutoTokenizer.from_pretrained(CFG.model_name)
    groups = [group_key(x, i) for i, x in enumerate(data)]
    y = [label2id["promise_status"][norm_label("promise_status", x.get("promise_status", ""))]
         for x in data]
    sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    oof = []
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros(len(data)), y, groups)):
        g_tr = {groups[i] for i in tr_idx}; g_va = {groups[i] for i in va_idx}
        print(f"\n===== Fold {k} | train={len(tr_idx)}+pseudo{len(pseudo)} val={len(va_idx)} "
              f"組交集={len(g_tr & g_va)}(應為0) =====")
        ckpt = os.path.join(CFG.ckpt_dir, f"{CFG.ckpt_prefix}{k}.pt")
        best = train_one_fold([data[i] for i in tr_idx] + pseudo,
                              [data[i] for i in va_idx], tok, ckpt)
        oof.append(best)
    print(f"\n===== CV 完成 | 各折最佳 = {[round(x,4) for x in oof]} | 平均 = {np.mean(oof):.4f} =====")
    return oof


# ============================================================
# 9. 推論（Ensemble：逐一載入 fold，累加 softmax 機率後平均）
# ============================================================
def _load_test(path):
    if path.endswith(".json"):
        recs = json.load(open(path, encoding="utf-8"))
        return recs, pd.DataFrame(recs)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df.to_dict(orient="records"), df


@torch.no_grad()
def _load_ckpt_model(ckpt_path, tok_cache, loader_cache, recs):
    """依權重檔內記錄的 model_name 重建模型與 tokenizer，支援跨 backbone 混合 ensemble。"""
    obj = torch.load(ckpt_path, map_location=CFG.device)
    mname = obj.get("model_name", CFG.model_name)
    cond = obj.get("conditional", False)
    old, old_c = CFG.model_name, CFG.conditional
    CFG.model_name, CFG.conditional = mname, cond   # 頭尺寸/backbone 須與權重一致
    model = MultiTaskMacBERT().to(CFG.device)
    model.load_state_dict(obj["model"]); model.eval()
    model._conditional = cond                        # 推論時轉輸出空間用
    CFG.model_name, CFG.conditional = old, old_c
    if mname not in tok_cache:                  # 不同 backbone 各自 tokenizer / loader
        tok_cache[mname] = AutoTokenizer.from_pretrained(mname)
        loader_cache[mname] = DataLoader(ESGDataset(recs, tok_cache[mname], with_labels=False),
                                         batch_size=CFG.batch_size, shuffle=False,
                                         collate_fn=collate_fn)
    return model, loader_cache[mname]


def _ensemble_probs(ckpts, recs):
    """逐一載入每個權重檔，累加各任務 softmax，回傳平均機率。"""
    N = len(recs)
    prob_sum = {f: torch.zeros(N, num_labels[f]) for f in TASKS}
    tok_cache, loader_cache = {}, {}
    for ckpt in ckpts:
        model, loader = _load_ckpt_model(ckpt, tok_cache, loader_cache, recs)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(CFG.device); mask = batch["attention_mask"].to(CFG.device)
                with torch.amp.autocast("cuda", enabled=CFG.amp):
                    logits = model(ids, mask)
                bs = ids.size(0)
                out_p = to_output_probs(logits, getattr(model, "_conditional", False))
                for f in TASKS:
                    prob_sum[f][offset:offset+bs] += out_p[f].cpu()
                offset += bs
        del model
        if CFG.device.type == "cuda":
            torch.cuda.empty_cache()
    return {f: prob_sum[f] / len(ckpts) for f in TASKS}


def run_infer(test_path="/content/test.json", out_path="/content/submission.csv",
              ckpts=None, adjust_path=None):
    if ckpts is None:
        ckpts = sorted(glob.glob(os.path.join(CFG.ckpt_dir, f"{CFG.ckpt_prefix}*.pt")))
    assert ckpts, "找不到任何 fold*.pt 權重"
    print(f"Ensemble 模型數: {len(ckpts)} → {ckpts}")

    recs, df = _load_test(test_path)
    avg = _ensemble_probs(ckpts, recs)                             # 跨 backbone 平均機率
    if adjust_path:                                                # 套用 OOF 調好的 logit 校正
        a = json.load(open(adjust_path, encoding="utf-8"))
        avg = {f: torch.log(avg[f].clamp_min(1e-9))
                  - a["taus"][f] * torch.tensor(a["priors"][f], dtype=torch.float32).log()
               for f in TASKS}
        print(f"已套用 logit 校正: taus={a['taus']}")
    fixed = apply_logic_fixes(avg)                                 # 對平均機率做硬性後處理

    def to_str(field, idx):
        lab = id2label[field][idx]
        return CFG.output_na_token if lab == "N/A" else lab
    for f in TASKS:
        df[f] = [to_str(f, int(i)) for i in fixed[f].tolist()]

    df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    raw = open(out_path, "rb").read()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw, "格式錯誤(BOM/CRLF)"
    print(f"Saved {len(df)} rows → {out_path}（UTF-8 no-BOM, LF ✓）")
    return out_path


# ============================================================
# 10. 用「有標籤」資料評估 ensemble（量真實分數，對齊排行榜）
# ============================================================
@torch.no_grad()
def run_eval(label_path="/content/vpesg4k_val_1000.json", ckpts=None):
    """對有答案的驗證集做 ensemble 預測 + LogicFixes，印出各任務 macro-F1 與加權總分。"""
    if ckpts is None:
        ckpts = sorted(glob.glob(os.path.join(CFG.ckpt_dir, f"{CFG.ckpt_prefix}*.pt")))
    assert ckpts, "找不到任何 fold*.pt 權重"
    print(f"Ensemble 模型數: {len(ckpts)}")

    recs = json.load(open(label_path, encoding="utf-8")) if label_path.endswith(".json") \
        else pd.read_csv(label_path, dtype=str, keep_default_na=False).to_dict("records")
    fixed = apply_logic_fixes(_ensemble_probs(ckpts, recs))
    gold = {f: [label2id[f][norm_label(f, r.get(f, ""))] for r in recs] for f in TASKS}

    print("\n任務             macro-F1   (權重)")
    total = 0.0
    for f in TASKS:
        mf1 = f1_score(gold[f], fixed[f].tolist(), average="macro", zero_division=0)
        total += FIELD_WEIGHTS[f] * mf1
        print(f"  {f:22s} {mf1:.4f}   ({FIELD_WEIGHTS[f]:.0%})")
    print(f"\n  >>> 加權總分 (≈排行榜估計) = {total:.4f}")
    return total


# ============================================================
# 11. OOF 誠實評估 + 稀少類別 logit 校正（零訓練成本）
# ============================================================
def _make_splits(data):
    """重現 run_cv 的切分（同 seed、同 group_key → 折完全一致）。"""
    groups = [group_key(x, i) for i, x in enumerate(data)]
    y = [label2id["promise_status"][norm_label("promise_status", x.get("promise_status", ""))]
         for x in data]
    sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    return list(sgkf.split(np.zeros(len(data)), y, groups))


@torch.no_grad()
def run_oof(data_path="/content/train_2000.json", prefixes=("fold", "rb"),
            save_path="/content/oof.npz"):
    """
    對訓練資料做 out-of-fold 預測：fold-k 模型只預測第 k 折 val（它沒看過的資料），
    多組 prefix（不同 backbone）的 OOF 機率取平均 → 2000 筆無洩漏的誠實分數。
    這個分數可直接拿來比較後處理設定，不用浪費排行榜上傳。
    """
    data = json.load(open(data_path, encoding="utf-8"))
    splits = _make_splits(data)
    N = len(data)
    prob_sum = {f: torch.zeros(N, num_labels[f]) for f in TASKS}

    for prefix in prefixes:
        for k, (_, va_idx) in enumerate(splits):
            ckpt = os.path.join(CFG.ckpt_dir, f"{prefix}{k}.pt")
            assert os.path.exists(ckpt), f"缺權重 {ckpt}"
            obj = torch.load(ckpt, map_location=CFG.device)
            mname = obj.get("model_name", CFG.model_name)
            cond = obj.get("conditional", False)
            old, old_c = CFG.model_name, CFG.conditional
            CFG.model_name, CFG.conditional = mname, cond
            model = MultiTaskMacBERT().to(CFG.device)
            model.load_state_dict(obj["model"]); model.eval()
            model._conditional = cond
            CFG.model_name, CFG.conditional = old, old_c

            recs = [data[i] for i in va_idx]
            tok = AutoTokenizer.from_pretrained(mname)
            loader = DataLoader(ESGDataset(recs, tok, with_labels=False),
                                batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)
            va = torch.as_tensor(va_idx, dtype=torch.long)
            off = 0
            for batch in loader:
                ids = batch["input_ids"].to(CFG.device)
                mask = batch["attention_mask"].to(CFG.device)
                with torch.amp.autocast("cuda", enabled=CFG.amp):
                    logits = model(ids, mask)
                bs = ids.size(0)
                out_p = to_output_probs(logits, getattr(model, "_conditional", False))
                for f in TASKS:
                    prob_sum[f][va[off:off+bs]] += out_p[f].cpu()
                off += bs
            del model
            if CFG.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[oof] {prefix}{k} 完成（val {len(va_idx)} 筆）")

    avg = {f: prob_sum[f] / len(prefixes) for f in TASKS}
    gold = {f: np.array([label2id[f][norm_label(f, x.get(f, ""))] for x in data]) for f in TASKS}

    fixed = apply_logic_fixes(avg)
    print("\nOOF 誠實分數（未校正）：")
    total = 0.0
    for f in TASKS:
        mf1 = f1_score(gold[f].tolist(), fixed[f].tolist(), average="macro", zero_division=0)
        total += FIELD_WEIGHTS[f] * mf1
        print(f"  {f:22s} {mf1:.4f}")
    print(f"  >>> 加權總分 = {total:.4f}")

    np.savez(save_path,
             **{f"prob_{f}": avg[f].numpy() for f in TASKS},
             **{f"gold_{f}": gold[f] for f in TASKS})
    print(f"OOF 機率已存 → {save_path}")
    return total


def tune_adjustment(oof_path="/content/oof.npz", out_path="/content/adjust.json",
                    taus=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)):
    """
    在 OOF 上做 logit 校正調參：score = log(p) - tau * log(prior)。
    tau>0 會抬升稀少類別（within_2_years / Misleading / Not Clear）被選中的機會，
    直接針對 macro-F1 被先驗壓死的弱點。逐任務座標上升搜尋，全程以加權 F1 為準。
    """
    z = np.load(oof_path)
    probs = {f: torch.tensor(z[f"prob_{f}"]) for f in TASKS}
    gold = {f: z[f"gold_{f}"].tolist() for f in TASKS}
    priors = {}
    for f in TASKS:
        cnt = np.bincount(z[f"gold_{f}"], minlength=num_labels[f]).astype(float)
        priors[f] = np.clip(cnt / cnt.sum(), 1e-6, None)

    def total_score(tau_d):
        adj = {f: torch.log(probs[f].clamp_min(1e-9))
                  - tau_d[f] * torch.tensor(np.log(priors[f]), dtype=torch.float32) for f in TASKS}
        fixed = apply_logic_fixes(adj)
        return sum(FIELD_WEIGHTS[f] * f1_score(gold[f], fixed[f].tolist(),
                                               average="macro", zero_division=0) for f in TASKS)

    best = {f: 0.0 for f in TASKS}
    base = total_score(best)
    for _ in range(2):                       # 兩輪座標上升
        for f in TASKS:
            cur = best[f]
            for t in taus:
                best[f] = t
                s = total_score(best)
                if s > base + 1e-6:
                    base, cur = s, t
                else:
                    best[f] = cur
            best[f] = cur
    print(f"最佳 tau = {best} | 校正後 OOF 加權總分 = {base:.4f}")
    json.dump({"taus": best, "priors": {f: priors[f].tolist() for f in TASKS}},
              open(out_path, "w", encoding="utf-8"))
    print(f"校正參數已存 → {out_path}（run_infer(adjust_path=...) 套用）")
    return best, base



@torch.no_grad()
def make_pseudo(test_path="/content/vpesg4k_test_2000.json",
                out_path="/content/pseudo.json", ckpts=None, conf=0.90):
    """
    信心過濾規則（全部以 ensemble 平均機率計）：
      - promise=No  → 只需 P(promise=No) ≥ conf（其餘任務被邏輯強制為 N/A，不另要求）
      - promise=Yes → P(promise=Yes) 與 timeline/evidence_status/evidence_quality
                      被選類別的機率都 ≥ conf 才收
    產出 JSON 可直接餵 run_cv(pseudo_path=...)。conf 越高越乾淨、筆數越少。
    """
    if ckpts is None:
        ckpts = sorted(glob.glob(os.path.join(CFG.ckpt_dir, f"{CFG.ckpt_prefix}*.pt")))
    assert ckpts, "找不到權重"
    recs, _ = _load_test(test_path)
    avg = _ensemble_probs(ckpts, recs)
    fixed = apply_logic_fixes(avg)

    yes_id = label2id["promise_status"]["Yes"]
    no_id = label2id["promise_status"]["No"]
    kept = []
    for i, r in enumerate(recs):
        p_pred = int(fixed["promise_status"][i])
        ok = False
        if p_pred == no_id:
            ok = float(avg["promise_status"][i, no_id]) >= conf
        else:
            ok = (float(avg["promise_status"][i, yes_id]) >= conf and
                  all(float(avg[f][i, int(fixed[f][i])]) >= conf
                      for f in ["verification_timeline", "evidence_status", "evidence_quality"]))
        if ok:
            item = dict(r)
            for f in TASKS:
                item[f] = id2label[f][int(fixed[f][i])]   # 內部 'N/A' 字串，norm_label 可讀
            kept.append(item)

    json.dump(kept, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    n_yes = sum(1 for x in kept if x["promise_status"] == "Yes")
    print(f"[pseudo] conf≥{conf}: 收 {len(kept)}/{len(recs)} 筆"
          f"（Yes={n_yes}, No={len(kept)-n_yes}）→ {out_path}")
    return out_path


if __name__ == "__main__":
    run_cv()
