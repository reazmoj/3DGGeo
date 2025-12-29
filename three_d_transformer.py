# import torch 
# import torch.nn as nn 
# import timm

# class AttentionPooling(nn.Module):
#     def __init__(self, embed_dim):
#         super(AttentionPooling, self).__init__()
#         self.attn = nn.Linear(embed_dim, 1)

#     def forward(self, x, mask=None):
#         # x: (B, N, D)
#         attn_weights = self.attn(x).squeeze(-1)  # (B, N)

#         if mask is not None:
#             mask = mask.bool()  # ensure it's bool type
#             attn_weights = attn_weights.masked_fill(~mask, float('-inf'))  # safe masking

#         attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)  # (B, N, 1)
#         pooled = torch.sum(x * attn_weights, dim=1)  # (B, D)
#         return pooled

# class ThreeDTransformer(nn.Module): 
#     def __init__(self, embed_dim=128, num_heads=8, num_layers=4, 
#                  person_num_classes=1652, group_num_classes=860, sigma=10, dropout=0.1, alpha=0.5): 
#         """
#         Args:
#             embed_dim (int): Embedding dimension for both appearance and layout features.
#             num_heads (int): Number of heads in transformer layers.
#             num_layers (int): Number of transformer layers.
#             person_num_classes (int): Number of person classes.
#             group_num_classes (int): Number of group classes.
#             sigma (int): Discretization parameter for layout tokens.
#             dropout (float): Dropout rate for transformer layers.
#         """
#         super(ThreeDTransformer, self).__init__() 
#         self.sigma = sigma
#         self.person_num_classes = person_num_classes
#         self.group_num_classes = group_num_classes
#         self.alpha = alpha

#         # Layout Tokens Initialization for X, Y, D axes.
#         self.layout_tokens_x = nn.Parameter(torch.randn(sigma, embed_dim))
#         self.layout_tokens_y = nn.Parameter(torch.randn(sigma, embed_dim))
#         self.layout_tokens_d = nn.Parameter(torch.randn(sigma, embed_dim))
        
#         # Appearance Backbone: ViT model.
#         self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
#         self.appearance_proj = nn.Linear(768, embed_dim)
        
#         # Layout projection from concatenated tokens.
#         self.layout_proj = nn.Linear(3 * embed_dim, embed_dim)
        
#         # Person Transformer.
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dropout=dropout,
#             batch_first=True
#         ) # new editation

#         self.person_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Person Classification Head.
#         self.person_fc = nn.Linear(embed_dim, self.person_num_classes)
        
#         # Group Transformer.
#         group_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dropout=dropout,
#             batch_first=True
#         ) # new editation

#         self.group_transformer = nn.TransformerEncoder(group_encoder_layer, num_layers=num_layers)

#         # Add a projection layer after concatenation to reduce back to embed_dim
#         self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)  # 64+64 → 64
        
#         self.attn_pool = AttentionPooling(embed_dim)  # <-- Attention pooling added

#         # Final classification layer for group-level output.
#         self.fc = nn.Linear(embed_dim, self.group_num_classes)

#     def forward(self, person_imgs, sampled_layout_features, return_feature=False):
#         """
#         Args:
#             person_imgs (torch.Tensor): Input person images with shape (B, N, 3, H, W).
#             sampled_layout_features (torch.Tensor): Layout indices with shape (B, N, 3) with values in [0, sigma).
#             return_feature (bool): If True, return the aggregated feature embeddings for evaluation.
#         Returns:
#             If return_feature is False, returns (person_logits, group_logits).
#             If True, returns the aggregated features from the group transformer.
#         """
#         B, N, C, H, W = person_imgs.shape
#         person_imgs = person_imgs.view(B * N, C, H, W)
#         # Extract appearance features.
#         vit_features = self.vit(person_imgs)
#         appearance_features = self.appearance_proj(vit_features)  # (B*N, embed_dim)
#         raw_person_embeddings = appearance_features.view(B, N, -1)

#         # Create valid mask for layout features.
#         valid_mask = (sampled_layout_features >= 0).all(dim=-1)  # Shape: (B, N)

#         clipped_indices = torch.clamp(sampled_layout_features, min=0)
        
#         # Layout token extraction.
#         token_x = self.layout_tokens_x[clipped_indices[:, :, 0]]
#         token_y = self.layout_tokens_y[clipped_indices[:, :, 1]]
#         token_d = self.layout_tokens_d[clipped_indices[:, :, 2]]
#         layout_tokens_cat = torch.cat([token_x, token_y, token_d], dim=-1)
#         layout_embeddings = self.layout_proj(layout_tokens_cat)
#         # layout_embeddings = layout_embeddings * valid_mask.float()
#         layout_embeddings = layout_embeddings * valid_mask.unsqueeze(-1).float()
        
#         # Fusion: here we simply add appearance and layout features.
        
#         # combined_features = torch.cat([raw_person_embeddings, layout_embeddings], dim=-1)
#         combined_features = raw_person_embeddings + self.alpha * layout_embeddings

#         combined_features = self.fusion_proj(combined_features)  # Reduce to 64-dim
        
#         person_features = self.person_transformer(
#             combined_features,
#             src_key_padding_mask=~valid_mask
#         )   
        
#         if return_feature:
#             group_features = self.group_transformer(
#                 person_features,
#                 src_key_padding_mask=~valid_mask
#             )
#             aggregated_features = self.attn_pool(group_features, valid_mask)
#             return aggregated_features


#         # Person classification.
#         person_logits = self.person_fc(person_features.reshape(B * N, -1)).reshape(B, N, -1)
        
#         # Group transformer and classification.
#         group_features = self.group_transformer(
#             person_features,
#             src_key_padding_mask=~valid_mask
#         )
#         aggregated_features = self.attn_pool(group_features, valid_mask)
#         group_logits = self.fc(aggregated_features)
        
#         return person_features, person_logits, aggregated_features, group_logits

# new version of model
import torch
import torch.nn as nn
import timm


# -------------------------------------------------------
# Attention Pooling (Mask-aware)
# -------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        """
        x:    (B, N, D)
        mask: (B, N)  -> True for valid person
        """
        attn_score = self.attn(x).squeeze(-1)  # (B, N)
        attn_score = attn_score.masked_fill(~mask, float("-inf"))
        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        pooled = torch.sum(x * attn_weight, dim=1)
        return pooled


# -------------------------------------------------------
# Main Model
# -------------------------------------------------------
class ThreeDTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        person_num_classes=1652,
        group_num_classes=860,
        sigma=10,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sigma = sigma

        # -----------------------------
        # Layout Tokens (X, Y, Depth)
        # -----------------------------
        self.layout_x = nn.Embedding(sigma, embed_dim)
        self.layout_y = nn.Embedding(sigma, embed_dim)
        self.layout_d = nn.Embedding(sigma, embed_dim)

        self.layout_proj = nn.Linear(3 * embed_dim, embed_dim)

        # -----------------------------
        # Appearance Backbone (ViT)
        # -----------------------------
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        self.appearance_proj = nn.Linear(768, embed_dim)

        # -----------------------------
        # Fusion
        # -----------------------------
        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)

        # -----------------------------
        # Person Transformer
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.person_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.person_classifier = nn.Linear(embed_dim, person_num_classes)

        # -----------------------------
        # Group Transformer
        # -----------------------------
        group_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.group_transformer = nn.TransformerEncoder(
            group_layer, num_layers=num_layers
        )

        self.attn_pool = AttentionPooling(embed_dim)
        self.group_classifier = nn.Linear(embed_dim, group_num_classes)

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, person_imgs, layout_indices, return_feature=False):
        """
        person_imgs:    (B, N, 3, H, W)
        layout_indices:(B, N, 3)   values in [0, sigma) or -1
        """

        B, N, C, H, W = person_imgs.shape

        # -----------------------------
        # Valid mask
        # -----------------------------
        valid_mask = (layout_indices >= 0).all(dim=-1)  # (B, N)

        # -----------------------------
        # Appearance
        # -----------------------------
        person_imgs = person_imgs.view(B * N, C, H, W)
        app_feat = self.vit(person_imgs)
        app_feat = self.appearance_proj(app_feat)
        app_feat = app_feat.view(B, N, -1)

        # -----------------------------
        # Layout Embedding
        # -----------------------------
        layout_idx = torch.clamp(layout_indices, min=0)

        lx = self.layout_x(layout_idx[..., 0])
        ly = self.layout_y(layout_idx[..., 1])
        ld = self.layout_d(layout_idx[..., 2])

        layout_feat = torch.cat([lx, ly, ld], dim=-1)
        layout_feat = self.layout_proj(layout_feat)

        layout_feat = layout_feat * valid_mask.unsqueeze(-1)

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = torch.cat([app_feat, layout_feat], dim=-1)
        fused = self.fusion_proj(fused)

        # -----------------------------
        # Person Transformer
        # -----------------------------
        person_feat = self.person_transformer(
            fused,
            src_key_padding_mask=~valid_mask
        )

        # -----------------------------
        # Person logits
        # -----------------------------
        person_logits = self.person_classifier(
            person_feat.reshape(B * N, -1)
        ).view(B, N, -1)

        # -----------------------------
        # Group Transformer
        # -----------------------------
        group_tokens = self.group_transformer(
            person_feat,
            src_key_padding_mask=~valid_mask
        )

        group_feat = self.attn_pool(group_tokens, valid_mask)

        if return_feature:
            return group_feat

        group_logits = self.group_classifier(group_feat)

        return  person_feat, person_logits, group_feat, group_logits

