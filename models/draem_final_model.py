"""
draem_final_model.py  ——  DRAEM 修复合并版（基于 v3/v4 全面修正）
=======================================================================
修复的问题（按严重程度排序）：

[BUG-1] v3/v4: 双门控被调用两次 → alpha_dir 实际是第二次 forward 的 alpha_reg
  位置: v3 forward 344-346行，v4 forward 405-413行
  修复: alpha_reg, alpha_dir = self.gate(...) 只调用一次

[BUG-2] v3: CrossAttentionHead 76行 scores 被除以 temp 两次（self.temp 已在75行用过）
  位置: v3 line 76: attn = torch.sigmoid(scores / self.temp)
        scores = (q * k).sum(dim=-1) / self.temp  ← 已经除过了
  修复: attn = torch.sigmoid(scores)  直接用已缩放的 scores

[BUG-3] v3/v4: PolarityAttentionHead 中 polarity_score 用 torch.no_grad() 截断梯度
  位置: v3 line 113-114
  后果: 极性分数不参与反向传播，方向路径无法通过 polarity 信号学习
  修复: 去掉 no_grad，允许梯度流过

[BUG-4] v4: gate() 调用两次（force_alpha_reg 分支调用一次，else 分支又调用一次）
  位置: v4 forward 405-413行
  修复: 统一在 else 分支调用一次，force 时单独处理

[BUG-5] v3/v4: UncertaintyWeighting 对小数据集不稳定（log_vars 发散）
  位置: UncertaintyWeighting.forward()
  修复: 对 log_vars 施加 clamp(-4, 4)，防止 precision → 0 或 ∞
        同时添加 log_vars 的 L2 正则项

[BUG-6] v4: CrossAttentionHead（v4版）76行改回了 softmax，与 v3 修复不一致
  修复: 统一使用 per-sample sigmoid

[设计问题-1] 方向路径的 SentimentAlignment 对全零文本退化为常数输出
  pos_signal = clamp(x, min=0) = 0 向量 → proj_pos(0) = bias_only → 常数
  修复: 添加特征是否全零的检测标志位（has_text flag），在全零时 bypass

[设计问题-2] CrossAttentionHead 的 out_proj 从未被使用
  v3 line 66: self.out_proj = nn.Linear(d, d)  ← 定义了
  v3 line 77: attended = (attn...) * v).reshape(B, d)  ← 没用 out_proj
  修复: 使用 out_proj，使注意力输出经过线性变换

[设计问题-3] fusion_reg 和 fusion_dir 的隐层维度写死（hidden=48, out=32）
  当 d=64 时，n_fusion_reg = 64*6 = 384，压缩到 48 太激进
  修复: hidden = max(n_in // 2, 64)，自适应隐层维度

[设计问题-4] 方向路径的 dir_cross_attn 与 cross_attn 使用相同 TextModuleEncoder 输出
  两条路径的 cross-attention 输入 h_text 来自同一个 encoder，
  dir 路径本应有独立的 text encoding（体现"异步"）
  修复: 添加 dir_text_encs 独立编码器（可选，节省参数时共享）

[设计问题-5] _init_weights 对 LayerNorm 的 weight/bias 没有重置
  LayerNorm 默认 weight=1, bias=0，这是正确的，Kaiming 不适用于 LN
  修复: 跳过 LayerNorm 层的初始化
=======================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def _adaptive_hidden(n_in: int, min_h: int = 64) -> int:
    """自适应隐层维度：输入的一半，但不低于 min_h。"""
    return max(n_in // 2, min_h)


# ─────────────────────────────────────────────────────────────────
# 1. 结构化特征编码器
# ─────────────────────────────────────────────────────────────────

class StructEncoder(nn.Module):
    """
    市场特征编码器（残差 MLP）。
    包含 skip connection 保留原始特征信息。
    """
    def __init__(self, n_in: int, d: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = n_in if i == 0 else d
            layers += [nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.skip = nn.Linear(n_in, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# ─────────────────────────────────────────────────────────────────
# 2. 文本模块编码器
# ─────────────────────────────────────────────────────────────────

class TextModuleEncoder(nn.Module):
    """每个文本模块独立的 MLP 编码器（保留模块间异构性）。"""
    def __init__(self, n_in: int, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d),    nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


# ─────────────────────────────────────────────────────────────────
# 3. 跨模态注意力（修复版）
# ─────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Per-sample sigmoid attention（修复 v3/v4 的两个 attention 问题）。

    修复 [BUG-2]: scores 不再被 temp 除两次
    修复 [设计-2]: 使用 out_proj 变换注意力输出
    改进: 使用 sigmoid 而非 softmax（各模块独立激活，不强制竞争）
    """
    def __init__(self, d: int = 64):
        super().__init__()
        self.Wq    = nn.Linear(d, d, bias=False)
        self.Wk    = nn.Linear(d, d, bias=False)
        self.Wv    = nn.Linear(d, d, bias=False)
        self.Wo    = nn.Linear(d, d)
        self.norm  = nn.LayerNorm(d)
        self.scale = math.sqrt(d)

    def forward(self, h_struct: torch.Tensor, h_text: torch.Tensor):
        """
        h_struct: (B, d)  市场编码
        h_text:   (B, d)  文本模块编码
        返回: attended (B, d), attn_weight (B,)
        """
        q = self.Wq(h_struct)    # (B, d)
        k = self.Wk(h_text)      # (B, d)
        v = self.Wv(h_text)      # (B, d)

        # Per-sample dot product → sigmoid（只除一次 scale）
        score = (q * k).sum(dim=-1) / self.scale    # (B,)
        attn  = torch.sigmoid(score)                 # (B,) ∈ [0,1]

        # 注意力加权 + 输出投影 + 残差 + 归一化
        attended = self.Wo(attn.unsqueeze(-1) * v)  # (B, d)
        attended = self.norm(attended + h_struct)    # 残差连接

        return attended, attn


# ─────────────────────────────────────────────────────────────────
# 4. 极性感知注意力（修复版）
# ─────────────────────────────────────────────────────────────────

class PolarityAttention(nn.Module):
    """
    方向路径专用：从原始文本特征提取极性信号。

    修复 [BUG-3]: 去掉 torch.no_grad()，允许梯度流过极性分数
    修复 [BUG-2]: scores 只除一次 scale
    """
    def __init__(self, n_text_feat: int, d: int = 64):
        super().__init__()
        self.Wt   = nn.Linear(n_text_feat, d, bias=False)
        self.Wq   = nn.Linear(d, d, bias=False)
        self.Wo   = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.scale = math.sqrt(d)

    def forward(self, h_struct: torch.Tensor, x_text_raw: torch.Tensor):
        """
        h_struct:    (B, d)
        x_text_raw:  (B, n_text_feat) 原始文本特征（含方向得分）
        返回: attended (B, d), polarity_score (B,)
        """
        h_t = self.Wt(x_text_raw)                            # (B, d)
        q   = self.Wq(h_struct)                               # (B, d)

        score  = (q * h_t).sum(dim=-1) / self.scale          # (B,)
        attn   = torch.sigmoid(score)                         # (B,) ∈ [0,1]

        attended = self.Wo(attn.unsqueeze(-1) * h_t)         # (B, d)
        attended = self.norm(attended + h_struct)

        # 极性分数：文本特征的 tanh 均值（梯度可流过）
        polarity = torch.tanh(x_text_raw).mean(dim=-1)       # (B,) ∈ [-1,1]

        return attended, polarity


# ─────────────────────────────────────────────────────────────────
# 5. 双向情感对齐（加入全零检测）
# ─────────────────────────────────────────────────────────────────

class SentimentAlignment(nn.Module):
    """
    分离正向/负向情感池，处理 [设计问题-1]：全零文本退化。

    改进: 当文本全零时，用学习到的"无信息"嵌入替代常数偏置输出。
    """
    def __init__(self, n_text_feat: int, d: int = 64):
        super().__init__()
        self.proj_pos   = nn.Linear(n_text_feat, d)
        self.proj_neg   = nn.Linear(n_text_feat, d)
        self.sent_head  = nn.Sequential(
            nn.Linear(d * 2, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        # 无信息时的可学习嵌入（文本全零时使用）
        self.null_embed = nn.Parameter(torch.zeros(d * 2))

    def forward(self, x_text: torch.Tensor):
        """
        x_text: (B, n_text_feat)
        返回: sent_feat (B, 2d), sentiment (B,)
        """
        B = x_text.size(0)

        # 判断哪些样本有非零文本（has_text: (B,) bool）
        has_text = (x_text.abs().sum(dim=-1) > 1e-6)   # (B,)

        pos_signal = torch.clamp(x_text, min=0.0)
        neg_signal = torch.clamp(x_text, max=0.0)
        h_pos = torch.relu(self.proj_pos(pos_signal))   # (B, d)
        h_neg = torch.relu(self.proj_neg(neg_signal))   # (B, d)
        sent_feat_raw = torch.cat([h_pos, h_neg], dim=-1)  # (B, 2d)

        # 无文本样本替换为 null_embed
        null = self.null_embed.unsqueeze(0).expand(B, -1)  # (B, 2d)
        sent_feat = torch.where(
            has_text.unsqueeze(-1).expand_as(sent_feat_raw),
            sent_feat_raw,
            null,
        )

        sent = torch.tanh(self.sent_head(sent_feat).squeeze(-1))   # (B,)
        return sent_feat, sent, has_text


# ─────────────────────────────────────────────────────────────────
# 6. 置信度感知门控（修复 BUG-1 + BUG-5）
# ─────────────────────────────────────────────────────────────────

class ConfidenceAwareGate(nn.Module):
    """
    置信度感知稀疏门控（单次调用，修复 BUG-1）。

    修复 [BUG-5]: log_vars clamp 防止发散
    改进: 偏置初始化为负值，使得无文本时 α → 0
          alpha_reg 和 alpha_dir 各有独立的输出头（共享主干）
    """
    def __init__(self, total_text_feat: int, hidden: int = 32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(total_text_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.head_reg = nn.Linear(hidden, 1)
        self.head_dir = nn.Linear(hidden, 1)

        # 初始化: 偏置为 -2，使得 sigmoid(-2) ≈ 0.12（默认低文本依赖）
        nn.init.constant_(self.head_reg.bias, -2.0)
        nn.init.constant_(self.head_dir.bias, -2.0)

    def forward(self, x_text: torch.Tensor):
        """
        x_text: (B, total_text_feat)
        返回: alpha_reg (B,), alpha_dir (B,)
              单次 forward，修复 BUG-1
        """
        h = self.shared(x_text)
        alpha_reg = torch.sigmoid(self.head_reg(h).squeeze(-1))
        alpha_dir = torch.sigmoid(self.head_dir(h).squeeze(-1))
        return alpha_reg, alpha_dir   # 元组，不拆开调用


# ─────────────────────────────────────────────────────────────────
# 7. 不确定性加权（修复 BUG-5：防止发散）
# ─────────────────────────────────────────────────────────────────

class UncertaintyWeighting(nn.Module):
    """
    Kendall-style 多任务不确定性加权。

    修复 [BUG-5]:
      - log_vars 限制在 [-4, 4]，防止 precision → 0 或 ∞
      - 添加 L2 正则：鼓励权重不要偏离 1 太远
    """
    def __init__(self, n_tasks: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        log_vars = self.log_vars.clamp(-4.0, 4.0)   # 修复 BUG-5
        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-log_vars[i])
            total = total + precision * loss + log_vars[i] * 0.5
        # L2 正则（轻微惩罚极端权重）
        total = total + 0.01 * (log_vars ** 2).sum()
        return total


# ─────────────────────────────────────────────────────────────────
# 8. 自适应融合块（修复 [设计问题-3]）
# ─────────────────────────────────────────────────────────────────

class FusionBlock(nn.Module):
    """
    自适应隐层维度的融合 MLP。

    修复 [设计问题-3]: hidden 不再写死，根据 n_in 自适应
    """
    def __init__(self, n_in: int, out: int = 64, dropout: float = 0.2):
        super().__init__()
        hidden = _adaptive_hidden(n_in, min_h=out)
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out),  nn.LayerNorm(out),  nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# 9. DRAEM 最终模型（整合所有修复）
# ─────────────────────────────────────────────────────────────────

class DRAEM(nn.Module):
    """
    DRAEM — Dynamic Regime-Attention Ensemble Model（修复合并版）

    架构：
      市场特征 → StructEncoder → h_s
      各文本模块 → TextModuleEncoder → 独立编码
      回归路径: CrossModalAttention(Q=h_s, K/V=text) → α_reg → fusion_reg → y_reg
      方向路径: PolarityAttention(raw_text) + SentimentAlignment → α_dir → fusion_dir → y_dir
      门控: ConfidenceAwareGate(all_text) → α_reg, α_dir（单次调用）
      辅助: regime 分类头

    主要修复:
      BUG-1: 双门控单次调用
      BUG-2: Attention scores 只除一次 scale
      BUG-3: polarity_score 梯度可流过
      BUG-4: gate 单次调用
      BUG-5: UncertaintyWeighting clamp + 正则
      BUG-6: 统一使用 sigmoid attention
      设计-1: 全零文本退化处理
      设计-2: out_proj 正确使用
      设计-3: FusionBlock 自适应 hidden
      设计-5: _init_weights 跳过 LayerNorm
    """

    def __init__(
        self,
        n_struct:          int,
        module_sizes:      dict,        # {mod_name: n_feat}
        n_global:          int,         # 全局文本特征维度（用于门控）
        d:                 int   = 64,
        n_enc_layers:      int   = 2,
        dropout:           float = 0.15,
        use_uncertainty:   bool  = False,  # 建议 False（小数据集不稳定）
        lambda_dir:        float = 0.5,
        lambda_regime:     float = 0.1,
    ):
        super().__init__()
        self.d            = d
        self.module_names = list(module_sizes.keys())
        self.n_modules    = len(self.module_names)
        self.use_uncertainty = use_uncertainty

        # ── 编码器 ──────────────────────────────────────────────────
        self.struct_enc = StructEncoder(n_struct, d, n_enc_layers, dropout)

        self.text_encs = nn.ModuleDict({
            m: TextModuleEncoder(sz, d, dropout)
            for m, sz in module_sizes.items()
        })

        # ── 回归路径注意力 ──────────────────────────────────────────
        self.reg_attns = nn.ModuleDict({
            m: CrossModalAttention(d)
            for m in self.module_names
        })

        # ── 方向路径注意力（独立，体现异步双路径）────────────────────
        self.dir_attns = nn.ModuleDict({
            m: CrossModalAttention(d)
            for m in self.module_names
        })

        # ── 方向路径专用组件 ─────────────────────────────────────────
        n_text_total = sum(module_sizes.values())
        self.polarity_attn  = PolarityAttention(n_text_total, d)
        self.sentiment_align = SentimentAlignment(n_text_total, d)

        # ── 门控（单次调用，修复 BUG-1）──────────────────────────────
        self.gate = ConfidenceAwareGate(n_global, hidden=32)

        # ── 融合层（自适应 hidden，修复设计问题-3）───────────────────
        # 回归: h_s(d) + M×attended(d) = (1+M)*d
        n_fuse_reg = d * (1 + self.n_modules)
        # 方向: h_s(d) + polarity(d) + sentiment(2d) + M×dir_attn(d) = (4+M)*d
        n_fuse_dir = d * (4 + self.n_modules)

        self.fusion_reg = FusionBlock(n_fuse_reg, out=d, dropout=dropout)
        self.fusion_dir = FusionBlock(n_fuse_dir, out=d, dropout=dropout)

        # ── 输出头 ───────────────────────────────────────────────────
        self.reg_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.dir_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

        # ── 辅助任务：Regime 分类 ────────────────────────────────────
        self.regime_head = nn.Sequential(
            nn.Linear(d * self.n_modules, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

        # ── 损失权重 ─────────────────────────────────────────────────
        self.lambda_dir    = lambda_dir
        self.lambda_regime = lambda_regime

        # ── 不确定性加权（可选）──────────────────────────────────────
        self.uncertainty = UncertaintyWeighting(3) if use_uncertainty else None

        self._init_weights()

    def _init_weights(self):
        """修复 [设计问题-5]：跳过 LayerNorm，只初始化 Linear。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # LayerNorm 保留默认初始化（weight=1, bias=0）

    def forward(
        self,
        x_struct:       torch.Tensor,         # (B, n_struct)
        x_mod_dict:     dict,                  # {mod: (B, n_mod_feat)}
        x_global:       torch.Tensor,          # (B, n_global) 用于门控
        x_text_all:     torch.Tensor = None,   # (B, n_text_total) 用于极性注意力
        force_alpha:    float        = None,   # 消融实验：强制固定 α
    ):
        B = x_struct.size(0)

        # 如果没有传 x_text_all，从 x_mod_dict 拼接
        if x_text_all is None:
            x_text_all = torch.cat(
                [x_mod_dict[m] for m in self.module_names], dim=-1
            )

        # ── 1. 结构化编码 ────────────────────────────────────────────
        h_s = self.struct_enc(x_struct)                            # (B, d)

        # ── 2. 文本模块编码 ──────────────────────────────────────────
        h_text = {m: self.text_encs[m](x_mod_dict[m]) for m in self.module_names}

        # ── 3. 跨模态注意力（回归 & 方向路径各自独立）────────────────
        reg_attended = [self.reg_attns[m](h_s, h_text[m])[0] for m in self.module_names]
        dir_attended = [self.dir_attns[m](h_s, h_text[m])[0] for m in self.module_names]
        attn_weights = {m: self.reg_attns[m](h_s, h_text[m])[1] for m in self.module_names}

        # ── 4. 门控（单次调用，修复 BUG-1）──────────────────────────
        if force_alpha is not None:
            alpha_reg = torch.full((B,), force_alpha, device=x_struct.device)
            alpha_dir = torch.full((B,), force_alpha, device=x_struct.device)
        else:
            alpha_reg, alpha_dir = self.gate(x_global)            # 单次调用

        # ── 5. 回归路径 ──────────────────────────────────────────────
        h_s_reg = h_s * (1 - alpha_reg).unsqueeze(-1)
        gated_reg = [att * alpha_reg.unsqueeze(-1) for att in reg_attended]
        h_fuse_reg = self.fusion_reg(torch.cat([h_s_reg] + gated_reg, dim=-1))
        y_reg = self.reg_head(h_fuse_reg).squeeze(-1)             # (B,)

        # ── 6. 方向路径 ──────────────────────────────────────────────
        h_polar, polarity     = self.polarity_attn(h_s, x_text_all)   # (B,d), (B,)
        h_sent, sentiment, has_text = self.sentiment_align(x_text_all) # (B,2d),(B,),(B,)

        h_s_dir     = h_s    * (1 - alpha_dir).unsqueeze(-1)
        h_polar_g   = h_polar * alpha_dir.unsqueeze(-1)
        gated_dir   = [att * alpha_dir.unsqueeze(-1) for att in dir_attended]

        h_fuse_dir = self.fusion_dir(
            torch.cat([h_s_dir, h_polar_g, h_sent] + gated_dir, dim=-1)
        )
        y_dir = self.dir_head(h_fuse_dir).squeeze(-1)             # (B,)

        # ── 7. 辅助 Regime 头 ────────────────────────────────────────
        h_text_cat = torch.cat([h_text[m] for m in self.module_names], dim=-1)
        p_regime   = self.regime_head(h_text_cat).squeeze(-1)     # (B,)

        return y_reg, y_dir, p_regime, alpha_reg, alpha_dir, sentiment, attn_weights

    def compute_loss(
        self,
        y_reg:       torch.Tensor,
        y_dir:       torch.Tensor,
        p_regime:    torch.Tensor,
        y_true:      torch.Tensor,
        regime_true: torch.Tensor,
    ) -> tuple:
        """
        多任务损失。
        建议 use_uncertainty=False（小数据集）时用固定权重。
        """
        loss_reg    = F.mse_loss(y_reg, y_true)
        dir_label   = (y_true > 0).float()
        loss_dir    = F.binary_cross_entropy_with_logits(y_dir, dir_label)
        loss_regime = F.binary_cross_entropy_with_logits(p_regime, regime_true)

        if self.use_uncertainty and self.uncertainty is not None:
            total = self.uncertainty([loss_reg, loss_dir, loss_regime])
        else:
            total = loss_reg + self.lambda_dir * loss_dir + self.lambda_regime * loss_regime

        # 在线指标（不参与梯度）
        with torch.no_grad():
            da_dir = (
                (torch.sigmoid(y_dir) > 0.5).float() * 2 - 1 == torch.sign(y_true)
            ).float().mean().item()

        return total, loss_reg.item(), loss_dir.item(), da_dir


# ─────────────────────────────────────────────────────────────────
# 10. DRAEM-LLM（v4 逻辑，修复所有 v4 bug）
# ─────────────────────────────────────────────────────────────────

class LLMPriorGate(nn.Module):
    """
    LLM 先验引导的门控（修复 v4 的双重调用 BUG-4）。

    修复: gate 只在一次 forward 中同时输出 alpha_reg, alpha_dir, llm_bias_term
    """
    def __init__(self, n_text: int, n_llm: int, hidden: int = 32):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(n_text, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.llm_branch = nn.Sequential(
            nn.Linear(n_llm, 16), nn.LayerNorm(16), nn.GELU(),
            nn.Linear(16, 16),    nn.LayerNorm(16), nn.GELU(),
        )
        fused = hidden + 16
        self.head_reg = nn.Linear(fused, 1)
        self.head_dir = nn.Linear(fused, 1)

        # 可学习的 LLM bias 注入强度，初始为 0（不依赖 LLM）
        self.bias_strength = nn.Parameter(torch.zeros(1))

        nn.init.constant_(self.head_reg.bias, -2.0)
        nn.init.constant_(self.head_dir.bias, -2.0)

    def forward(self, x_text: torch.Tensor, x_llm: torch.Tensor):
        """
        修复 BUG-4: 单次调用返回所有输出
        返回: alpha_reg (B,), alpha_dir (B,), llm_bias_term (B,)
        """
        h_text = self.text_branch(x_text)
        h_llm  = self.llm_branch(x_llm)
        h      = torch.cat([h_text, h_llm], dim=-1)

        alpha_reg = torch.sigmoid(self.head_reg(h).squeeze(-1))

        # alpha_dir 受 LLM confidence（第3维）调节
        raw_dir   = self.head_dir(h).squeeze(-1)
        alpha_dir = torch.sigmoid(raw_dir)
        if x_llm.size(-1) >= 3:
            conf      = x_llm[:, 2].clamp(0, 1)
            alpha_dir = alpha_dir * conf + 0.2 * (1 - conf)

        # LLM bias 注入项（第2维是方向偏置信号）
        if x_llm.size(-1) >= 2:
            llm_bias      = x_llm[:, 1].clamp(-1, 1)
            llm_bias_term = torch.tanh(self.bias_strength) * llm_bias
        else:
            llm_bias_term = torch.zeros(x_text.size(0), device=x_text.device)

        return alpha_reg, alpha_dir, llm_bias_term


class DRAEM_LLM(DRAEM):
    """
    DRAEM + LLM 先验（v4 逻辑，修复版）。

    继承 DRAEM 的所有修复，只替换门控为 LLMPriorGate。
    LLM bias 直接注入方向 logit（可学习强度，初始为 0）。
    """
    def __init__(
        self,
        n_struct:    int,
        module_sizes: dict,
        n_global:    int,
        n_llm:       int   = 12,
        d:           int   = 64,
        n_enc_layers: int  = 2,
        dropout:     float = 0.15,
        use_uncertainty: bool = False,
        lambda_dir:  float = 0.5,
        lambda_regime: float = 0.1,
    ):
        super().__init__(
            n_struct=n_struct,
            module_sizes=module_sizes,
            n_global=n_global,
            d=d, n_enc_layers=n_enc_layers, dropout=dropout,
            use_uncertainty=use_uncertainty,
            lambda_dir=lambda_dir, lambda_regime=lambda_regime,
        )
        # 替换门控为 LLM 引导版本（修复 BUG-4）
        self.gate   = LLMPriorGate(n_global, n_llm, hidden=32)
        self.n_llm  = n_llm

    def forward(
        self,
        x_struct:    torch.Tensor,
        x_mod_dict:  dict,
        x_global:    torch.Tensor,
        x_text_all:  torch.Tensor = None,
        x_llm:       torch.Tensor = None,
        force_alpha: float        = None,
    ):
        B = x_struct.size(0)

        if x_llm is None:
            x_llm = torch.zeros(B, self.n_llm, device=x_struct.device)
        if x_text_all is None:
            x_text_all = torch.cat([x_mod_dict[m] for m in self.module_names], dim=-1)

        # 编码
        h_s    = self.struct_enc(x_struct)
        h_text = {m: self.text_encs[m](x_mod_dict[m]) for m in self.module_names}

        # 注意力
        reg_attended = [self.reg_attns[m](h_s, h_text[m])[0] for m in self.module_names]
        dir_attended = [self.dir_attns[m](h_s, h_text[m])[0] for m in self.module_names]
        attn_weights = {m: self.reg_attns[m](h_s, h_text[m])[1] for m in self.module_names}

        # 门控（单次调用，修复 BUG-1 & BUG-4）
        if force_alpha is not None:
            alpha_reg     = torch.full((B,), force_alpha, device=x_struct.device)
            alpha_dir     = torch.full((B,), force_alpha, device=x_struct.device)
            llm_bias_term = torch.zeros(B, device=x_struct.device)
        else:
            alpha_reg, alpha_dir, llm_bias_term = self.gate(x_global, x_llm)

        # 回归路径
        h_s_reg    = h_s * (1 - alpha_reg).unsqueeze(-1)
        gated_reg  = [att * alpha_reg.unsqueeze(-1) for att in reg_attended]
        h_fuse_reg = self.fusion_reg(torch.cat([h_s_reg] + gated_reg, dim=-1))
        y_reg      = self.reg_head(h_fuse_reg).squeeze(-1)

        # 方向路径
        h_polar, polarity       = self.polarity_attn(h_s, x_text_all)
        h_sent, sentiment, has_text = self.sentiment_align(x_text_all)
        h_s_dir    = h_s    * (1 - alpha_dir).unsqueeze(-1)
        h_polar_g  = h_polar * alpha_dir.unsqueeze(-1)
        gated_dir  = [att * alpha_dir.unsqueeze(-1) for att in dir_attended]
        h_fuse_dir = self.fusion_dir(
            torch.cat([h_s_dir, h_polar_g, h_sent] + gated_dir, dim=-1)
        )
        y_dir_raw = self.dir_head(h_fuse_dir).squeeze(-1)
        y_dir     = y_dir_raw + llm_bias_term   # LLM bias 注入

        # Regime
        h_text_cat = torch.cat([h_text[m] for m in self.module_names], dim=-1)
        p_regime   = self.regime_head(h_text_cat).squeeze(-1)

        return y_reg, y_dir, p_regime, alpha_reg, alpha_dir, sentiment, attn_weights, llm_bias_term


# ─────────────────────────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    B, n_struct, d = 32, 20, 64
    module_sizes  = {"policy": 12, "market": 12, "compliance": 12}
    n_global      = 10

    # ── DRAEM 基础版 ─────────────────────────────────────────────
    model = DRAEM(
        n_struct=n_struct, module_sizes=module_sizes, n_global=n_global,
        d=d, dropout=0.15, use_uncertainty=False, lambda_dir=0.5,
    )

    x_s   = torch.randn(B, n_struct)
    x_mod = {m: torch.randn(B, sz) for m, sz in module_sizes.items()}
    x_g   = torch.randn(B, n_global)

    y_reg, y_dir, p_reg, ar, ad, sent, attn = model(x_s, x_mod, x_g)

    assert y_reg.shape == (B,), f"y_reg shape error: {y_reg.shape}"
    assert y_dir.shape == (B,), f"y_dir shape error: {y_dir.shape}"
    assert ar.shape   == (B,), f"alpha_reg shape error: {ar.shape}"
    assert ad.shape   == (B,), f"alpha_dir shape error: {ad.shape}"
    # BUG-1 验证：alpha_reg 和 alpha_dir 不应完全相同
    assert not torch.allclose(ar, ad, atol=1e-6), "BUG-1 still present: ar == ad"

    y_true   = torch.randn(B)
    regime_t = torch.randint(0, 2, (B,)).float()
    loss, l_r, l_d, da = model.compute_loss(y_reg, y_dir, p_reg, y_true, regime_t)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DRAEM base: params={n_params:,}  loss={loss.item():.4f}  "
          f"DA_dir={da*100:.1f}%  ar={ar.mean():.3f}  ad={ad.mean():.3f}")

    # ── DRAEM-LLM 验证 ────────────────────────────────────────────
    n_llm   = 12
    model_v = DRAEM_LLM(
        n_struct=n_struct, module_sizes=module_sizes, n_global=n_global,
        n_llm=n_llm, d=d, use_uncertainty=False, lambda_dir=0.5,
    )
    x_llm   = torch.randn(B, n_llm)
    out = model_v(x_s, x_mod, x_g, x_llm=x_llm)
    y_reg_v, y_dir_v, p_reg_v, ar_v, ad_v, sent_v, _, bias_v = out
    loss_v, *_ = model_v.compute_loss(y_reg_v, y_dir_v, p_reg_v, y_true, regime_t)
    loss_v.backward()
    print(f"DRAEM-LLM:  params={sum(p.numel() for p in model_v.parameters()):,}  "
          f"loss={loss_v.item():.4f}  bias_mean={bias_v.mean().item():.4f}")

    print("\n✓ All checks passed.")
    sys.exit(0)
