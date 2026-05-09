"""
draem_v3_model.py — DRAEM v3: Asymmetric Dual-Pathway Architecture
===================================================================
核心改进 vs v1/v2:
1. 异步双路径: 回归路径 + 方向路径彻底解耦
2. 双任务门控: α_reg (幅度) + α_dir (方向) 分别学习
3. 极性感知注意力: 方向路径直接从原始文本提取情感极性
4. 不确定性加权损失: Kendall-style multi-task learning
5. 双向情感对齐: 正向/负向情感池分别建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# 1. Encoders
# ─────────────────────────────────────────────────────────────────
class DiffTemporalEncoder(nn.Module):
    """结构化特征编码器。"""
    def __init__(self, n_in, d=32, n_layers=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(d if i > 0 else n_in, d))
            layers.append(nn.LayerNorm(d))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)
        self.skip_proj = nn.Linear(n_in, d) if n_in != d else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip_proj(x)


class TextModuleEncoder(nn.Module):
    """文本模块编码器（per-module MLP）。"""
    def __init__(self, n_in, d=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────
# 2. Cross-Attention
# ─────────────────────────────────────────────────────────────────
class CrossAttentionHead(nn.Module):
    """标准 dot-product cross-attention（用于回归路径）。"""
    def __init__(self, d=32, n_heads=2):
        super().__init__()
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        assert d % n_heads == 0
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.temp = nn.Parameter(torch.ones(1) * math.sqrt(d))

    def forward(self, h_struct, h_text):
        B, h, hd = h_struct.size(0), self.n_heads, self.head_dim
        q = self.W_q(h_struct).view(B, h, hd)
        k = self.W_k(h_text).view(B, h, hd)
        v = self.W_v(h_text).view(B, h, hd)
        scores = (q * k).sum(dim=-1) / self.temp
        attn = torch.sigmoid(scores)
        attended = (attn.unsqueeze(-1) * v).reshape(B, self.d)
        attended = self.norm(attended + h_struct)
        attended = self.out_proj(self.norm(attended + h_struct))

        return attended, attn.mean(dim=1)
class PolarityAttentionHead(nn.Module):
    """
    极性感知注意力（用于方向路径）。
    直接用原始文本（不过 encoder），保留情感极性信息。
    """
    def __init__(self, n_text_feat, d=32, n_heads=2):
        super().__init__()
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        assert d % n_heads == 0
        self.W_t = nn.Linear(n_text_feat, d)
        self.W_q = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.temp = nn.Parameter(torch.ones(1) * math.sqrt(d))

    def forward(self, h_struct, x_text_raw):
        """
        h_struct: (B, d)
        x_text_raw: (B, n_text_feat) — 原始文本特征
        Returns: attended (B, d), polarity_score (B,)
        """
        B, h, hd = h_struct.size(0), self.n_heads, self.head_dim
        h_text_proj = self.W_t(x_text_raw)
        q = self.W_q(h_struct).view(B, h, hd)
        k = h_text_proj.view(B, h, hd)
        v = h_text_proj.view(B, h, hd)
        scores = (q * k).sum(dim=-1) / self.temp
        attn = torch.sigmoid(scores)
        attended = (attn.unsqueeze(-1) * v).reshape(B, self.d)
        attended = self.norm(attended + h_struct)
        # 极性分数：基于原始文本特征的 sign-weighted mean
        with torch.no_grad():
            polarity = torch.tanh(x_text_raw).mean(dim=1)  # ∈ [-1, 1]
        return attended, polarity


# ─────────────────────────────────────────────────────────────────
# 3. 双任务门控
# ─────────────────────────────────────────────────────────────────
class TaskSpecificGate(nn.Module):
    """任务特定门控：α_reg (幅度) 和 α_dir (方向) 分别预测。"""
    def __init__(self, total_text_feat, hidden=16):
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

    def forward(self, x_text_combined):
        h_shared = self.shared(x_text_combined)
        alpha_reg = torch.sigmoid(self.head_reg(h_shared).squeeze(-1))
        alpha_dir = torch.sigmoid(self.head_dir(h_shared).squeeze(-1))
        return alpha_reg, alpha_dir


class SentimentAlignment(nn.Module):
    """
    双向情感对齐层（方向路径专用）。
    分别建模正向/负向情感池，捕捉完整情感谱。
    """
    def __init__(self, n_text_feat, d=32):
        super().__init__()
        self.proj_pos = nn.Linear(n_text_feat, d)
        self.proj_neg = nn.Linear(n_text_feat, d)
        self.sent_head = nn.Sequential(
            nn.Linear(d * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_text):
        """
        x_text: (B, n_text_feat)
        Returns: sent_feat (B, 2d), sentiment (B,)
        """
        pos_signal = torch.clamp(x_text, min=0.0)
        neg_signal = torch.clamp(x_text, max=0.0)
        h_pos = torch.relu(self.proj_pos(pos_signal))
        h_neg = torch.relu(self.proj_neg(neg_signal))
        sent_feat = torch.cat([h_pos, h_neg], dim=-1)
        sent = torch.tanh(self.sent_head(sent_feat).squeeze(-1))
        return sent_feat, sent


# ─────────────────────────────────────────────────────────────────
# 4. 融合块
# ─────────────────────────────────────────────────────────────────
class FusionBlock(nn.Module):
    def __init__(self, n_in, hidden=48, out=32, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
            nn.LayerNorm(out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# 5. 不确定性加权 (Kendall et al., 2018)
# ─────────────────────────────────────────────────────────────────
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, l in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * l.mean() + self.log_vars[i] * 0.5
        return total


# ─────────────────────────────────────────────────────────────────
# 6. DRAEM v3 主模型
# ─────────────────────────────────────────────────────────────────
class DRAEMv3(nn.Module):
    """
    DRAEM v3 — Asymmetric Dual-Pathway Architecture

    两条路径：
    - 回归路径: struct_enc → text_encs → cross_attn → α_reg → fusion_reg → reg_head
    - 方向路径: struct_enc → polarity_attn(raw) → sentiment_align → α_dir → fusion_dir → dir_head

    训练策略：
    - Uncertainty weighting: 自动平衡 reg/dir/regime 三个任务
    - 双门控: α_reg 学习文本何时对幅度可信，α_dir 学习文本何时对方向可信
    """
    def __init__(self, n_struct, text_module_sizes, total_text_feat,
                 d=32, n_enc_layers=2, dropout=0.25,
                 use_uncertainty=True, horizon_lambda=None,
                 use_gate=True, use_cross_attn=True):
        super().__init__()
        self.d = d
        self.use_uncertainty = use_uncertainty
        self.use_gate = use_gate
        self.use_cross_attn = use_cross_attn
        self.module_names = list(text_module_sizes.keys())
        self.n_modules = len(self.module_names)
        self.module_feat_dims = text_module_sizes

        # Encoders
        self.struct_enc = DiffTemporalEncoder(n_struct, d, n_enc_layers)
        self.text_encs = nn.ModuleDict()
        for mname, nsz in text_module_sizes.items():
            self.text_encs[mname] = TextModuleEncoder(nsz, d)

        # 回归路径: 标准 cross-attention
        self.cross_attns = nn.ModuleDict()
        for mname in self.module_names:
            self.cross_attns[mname] = CrossAttentionHead(d, n_heads=max(1, d // 16))

        # 方向路径: 极性感知 attention（直接用原始文本）
        self.polarity_attn = PolarityAttentionHead(total_text_feat, d, n_heads=max(1, d // 16))

        # 方向路径: 双向情感对齐
        self.sentiment_align = SentimentAlignment(total_text_feat, d)

        # 方向路径: 文本线索 cross-attention（与回归路径独立）
        self.dir_cross_attns = nn.ModuleDict()
        for mname in self.module_names:
            self.dir_cross_attns[mname] = CrossAttentionHead(d, n_heads=max(1, d // 16))

        # 双任务门控
        self.gate = TaskSpecificGate(total_text_feat, hidden=16)

        # 融合层
        # 回归: d*(1+M)
        # 方向: d(struct_res) + d(polarity) + 2d(sentiment) + M*d(dir_att) = (4+M)*d
        n_fusion_reg = d * (1 + self.n_modules)
        # 方向: 4d (残差+极性+2d情感) + M*d(dir_att) = (4+M)*d
        # 无 CrossAttn 时 = 4d
        n_fusion_dir = (4 + self.n_modules) * d if use_cross_attn else 4 * d

        self.fusion_reg = FusionBlock(n_fusion_reg, hidden=48, out=32, dropout=dropout)
        self.fusion_dir = FusionBlock(n_fusion_dir, hidden=64, out=32, dropout=dropout)

        # 输出头
        self.reg_head = nn.Linear(32, 1)
        self.dir_head = nn.Linear(32, 1)

        # 辅助任务: regime 分类
        self.regime_classifier = nn.Sequential(
            nn.Linear(d * self.n_modules, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

        # 不确定性加权
        self.uncertainty = UncertaintyWeighting(n_tasks=3) if use_uncertainty else None

        # Horizon-specific λ
        self.horizon_lambda = horizon_lambda or {'reg': 1.0, 'dir': 0.5, 'regime': 0.1}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_struct, x_text_dict, x_text_combined=None,
                force_alpha_reg=None, force_alpha_dir=None):
        """
        x_struct: (B, n_struct)
        x_text_dict: dict {module_name: (B, n_module_feat)}
        x_text_combined: (B, total_text_feat)
        """
        B = x_struct.size(0)

        # 1. 编码结构化特征
        h_s = self.struct_enc(x_struct)  # (B, d)

        # 2. 文本编码（两条路径共享）
        attended_list = []      # 回归路径用
        dir_attended_list = []  # 方向路径用
        h_text_encoded_list = []
        for mname in self.module_names:
            h_text = self.text_encs[mname](x_text_dict[mname])
            h_text_encoded_list.append(h_text)
            # 回归 cross-attn
            att_reg, _ = self.cross_attns[mname](h_s, h_text)
            attended_list.append(att_reg)
            # 方向 cross-attn（独立）
            att_dir, _ = self.dir_cross_attns[mname](h_s, h_text)
            dir_attended_list.append(att_dir)

        # 3. 文本拼接
        if x_text_combined is None:
            if x_text_dict and len(x_text_dict) > 0:
                x_text_combined = torch.cat(list(x_text_dict.values()), dim=-1)
            else:
                x_text_combined = torch.zeros(B, 0, device=x_struct.device)
        x_tc = x_text_combined

        # 4. 双任务门控（或强制 0.5）
        if not self.use_gate:
            alpha_reg = torch.full((B,), 0.5, device=x_struct.device)
            alpha_dir = torch.full((B,), 0.5, device=x_struct.device)
        else:
            alpha_reg, alpha_dir = self.gate(x_tc)  # 只调用一次

        # 5. ── 回归路径 ─────────────────────────────────────
        gated_list_reg = [att * alpha_reg.unsqueeze(-1) for att in attended_list]
        h_s_gated_reg = h_s * (1.0 - alpha_reg).unsqueeze(-1)
        fused_reg_in = torch.cat([h_s_gated_reg] + gated_list_reg, dim=-1)  # (B, d*(1+M))
        h_reg = self.fusion_reg(fused_reg_in)
        y_reg = self.reg_head(h_reg).squeeze(-1)  # (B,)

        # 6. ── 方向路径 ─────────────────────────────────────
        # 6a. 极性注意力（原始文本，不过 encoder）
        h_polarity, polarity_score = self.polarity_attn(h_s, x_tc)  # (B, d), (B,)

        # 6b. 双向情感对齐
        h_sent_feat, sent_score = self.sentiment_align(x_tc)  # (B, 2d), (B,)
        sentiment = polarity_score + 0.5 * sent_score  # 融合两个情感信号

        # 方向路径融合
        # h_sent_feat = 2d, 所以 No-CrossAttn 时总维度 = d(残差) + d(极性) + 2d(情感) = 4d
        # 有 CrossAttn 时 = 4d + M*d = (4+M)*d
        h_s_gated_dir = h_s * (1.0 - alpha_dir).unsqueeze(-1)  # (B, d)
        gated_polarity = h_polarity * alpha_dir.unsqueeze(-1)  # (B, d)
        if self.use_cross_attn and len(dir_attended_list) > 0:
            gated_dir_att = [att * alpha_dir.unsqueeze(-1) for att in dir_attended_list]  # M × (B, d)
            fused_dir_in = torch.cat(
                [h_s_gated_dir, gated_polarity, h_sent_feat] + gated_dir_att,
                dim=-1
            )  # (B, (4+M)*d)
        else:
            fused_dir_in = torch.cat(
                [h_s_gated_dir, gated_polarity, h_sent_feat],
                dim=-1
            )  # (B, 4d)
        h_dir = self.fusion_dir(fused_dir_in)
        y_dir = self.dir_head(h_dir).squeeze(-1)  # (B,)

        # 7. 辅助任务
        if len(h_text_encoded_list) > 0:
            h_text_cat = torch.cat(h_text_encoded_list, dim=-1)
            p_regime = self.regime_classifier(h_text_cat).squeeze(-1)
        else:
            p_regime = torch.zeros(B, device=x_struct.device)

        return y_reg, y_dir, p_regime, alpha_reg, alpha_dir, sentiment

    def compute_loss(self, y_reg, y_dir, p_regime, y_true,
                     alpha_regime=None, alpha_reg=None, alpha_dir=None):
        """多任务损失。"""
        loss_reg = F.mse_loss(y_reg, y_true)
        dir_label = (y_true > 0).float()
        loss_dir = F.binary_cross_entropy_with_logits(y_dir, dir_label)

        if alpha_regime is not None and alpha_regime.numel() > 0:
            loss_regime = F.binary_cross_entropy_with_logits(p_regime, alpha_regime)
        else:
            loss_regime = torch.tensor(0.0, device=y_reg.device)

        if self.use_uncertainty and self.uncertainty is not None:
            losses = [loss_reg, loss_dir, loss_regime]
            total_loss = self.uncertainty(losses)
        else:
            lw = self.horizon_lambda
            total_loss = (lw['reg'] * loss_reg +
                          lw['dir'] * loss_dir +
                          lw['regime'] * loss_regime)

        # Metrics
        dir_prob = torch.sigmoid(y_dir)
        da_dir = (torch.sign(y_true) == (dir_prob > 0.5).float() * 2 - 1).float().mean()
        reg_sign = (torch.sign(y_reg) == torch.sign(y_true)).float().mean()
        metrics = {
            'loss_reg': loss_reg.item(),
            'loss_dir': loss_dir.item(),
            'loss_regime': loss_regime.item(),
            'da_dir': da_dir.item(),
            'reg_sign_acc': reg_sign.item(),
        }
        return total_loss, metrics
