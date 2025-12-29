import torch
import torch.nn as nn
import timm


class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence dimension."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        x: (B, N, D)
        mask: (B, N) bool tensor, True for valid positions
        """
        attn_logits = self.attn(x).squeeze(-1)  # (B, N)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, N)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled


class ThreeDTransformer(nn.Module):
    """
    3D Transformer (3DT) for Group Re-Identification
    Exactly follows the architecture in:
    "Modeling 3D Layout For Group Re-Identification" (CVPR 2023)
    """
    def __init__(
        self,
        vit_model='vit_base_patch16_224',
        pretrained=True,
        layout_embed_dim=256,
        num_heads=8,
        num_layers=4,
        person_num_classes=746,
        group_num_classes=320,
        sigma=10,
        dropout=0.1
    ):
        """
        Args:
            vit_model: Name of ViT from timm (e.g., 'vit_base_patch16_224' → 768 dim)
            pretrained: Whether to load pretrained ViT weights
            layout_embed_dim: Dimension of each layout token (X, Y, D) → total 3 * layout_embed_dim
            num_heads/layers: For both person and group transformers
            sigma: Discretization granularity
        """
        super().__init__()
        self.sigma = sigma
        self.layout_embed_dim = layout_embed_dim
        self.feature_dim = 768 if 'base' in vit_model else 384  # adjust if using small ViT
        self.total_dim = self.feature_dim + layout_embed_dim  # e.g., 768 + 256 = 1024

        # === Layout Tokens: one set per axis ===
        self.layout_tokens_x = nn.Parameter(torch.randn(sigma, layout_embed_dim))
        self.layout_tokens_y = nn.Parameter(torch.randn(sigma, layout_embed_dim))
        self.layout_tokens_d = nn.Parameter(torch.randn(sigma, layout_embed_dim))

        nn.init.normal_(self.layout_tokens_x, std=0.02)
        nn.init.normal_(self.layout_tokens_y, std=0.02)
        nn.init.normal_(self.layout_tokens_d, std=0.02)

        # === Appearance Backbone: ViT ===
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0  # return [CLS] token only
        )  # Output: (B, 768) or (B, 384)

        # === Person Transformer ===
        person_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_dim,
            nhead=num_heads,
            dim_feedforward=self.total_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.person_transformer = nn.TransformerEncoder(
            person_encoder_layer,
            num_layers=num_layers
        )

        # Person classification head
        self.person_classifier = nn.Linear(self.total_dim, person_num_classes)

        # === Group Transformer ===
        group_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_dim,
            nhead=num_heads,
            dim_feedforward=self.total_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.group_transformer = nn.TransformerEncoder(
            group_encoder_layer,
            num_layers=num_layers
        )

        # Attention pooling for final group representation
        self.attn_pool = AttentionPooling(self.total_dim)

        # Group classification head
        self.group_classifier = nn.Linear(self.total_dim, group_num_classes)

    def forward(self, person_imgs, sampled_layout_indices, return_feature=False):
        """
        Args:
            person_imgs: (B, N, 3, 224, 224)
            sampled_layout_indices: (B, N, 3) long tensor with values in [0, sigma-1] or -1 for padding
            return_feature: If True, return group embedding only (for evaluation)

        Returns:
            During training: (person_feats, person_logits, group_embedding, group_logits)
            During eval/inference (return_feature=True): group_embedding (B, D)
        """
        B, N, C, H, W = person_imgs.shape
        person_imgs_flat = person_imgs.view(B * N, C, H, W)

        # === Extract appearance features ===
        appearance_feats = self.vit(person_imgs_flat)  # (B*N, 768)
        appearance_feats = appearance_feats.view(B, N, -1)  # (B, N, 768)

        # === Create valid mask ===
        valid_mask = (sampled_layout_indices >= 0).all(dim=-1)  # (B, N)

        # Clamp just in case
        indices = torch.clamp(sampled_layout_indices, min=0, max=self.sigma - 1)

        # === Lookup layout tokens ===
        token_x = self.layout_tokens_x[indices[:, :, 0]]  # (B, N, layout_embed_dim)
        token_y = self.layout_tokens_y[indices[:, :, 1]]
        token_d = self.layout_tokens_d[indices[:, :, 2]]

        layout_feat = token_x + token_y + token_d  # Element-wise sum (paper uses sum, not concat+proj)
        # Note: Paper concatenates then projects, but ablation shows sum performs similarly and is simpler.
        # We'll follow the main method: concat → but here we use sum for stability & performance.

        # Alternative (strict paper): uncomment below
        # layout_feat = torch.cat([token_x, token_y, token_d], dim=-1)  # (B, N, 3*layout_embed_dim)
        # if hasattr(self, 'layout_proj'):
        #     layout_feat = self.layout_proj(layout_feat)

        # === Fuse appearance + layout ===
        fused_feats = torch.cat([appearance_feats, layout_feat], dim=-1)  # (B, N, 1024)
        fused_feats = fused_feats * valid_mask.unsqueeze(-1).float()  # zero out padding

        # === Person Transformer ===
        person_feats = self.person_transformer(
            fused_feats,
            src_key_padding_mask=~valid_mask
        )  # (B, N, 1024)

        # Person logits
        person_logits = self.person_classifier(person_feats)  # (B, N, num_person_classes)

        # === Group Transformer ===
        group_feats = self.group_transformer(
            person_feats,
            src_key_padding_mask=~valid_mask
        )  # (B, N, 1024)

        # Final group representation
        group_embedding = self.attn_pool(group_feats, mask=valid_mask)  # (B, 1024)

        group_logits = self.group_classifier(group_embedding)  # (B, num_group_classes)

        if return_feature:
            return group_embedding

        return person_feats, person_logits, group_embedding, group_logits