"""
BiLSTM+MLP DTI Architecture

Drug-Target Interaction prediction via:
- Protein: BiLSTM → mean pool → (B, 128)
- Drug: MLP encoder → (B, 128)
- Fusion: concat (B, 256) → MLP → binary logit

Key design decisions:
- BiLSTM hidden_size = d_model // 2 → bidirectional output = d_model (matches Mamba width)
- Ablation position: MeanPool (no seq modeling) < BiLSTM (recurrent) < Mamba (SSM)
- Weight init: Xavier for Linear, Orthogonal for LSTM hidden-to-hidden
"""

import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    """
    Maps PsePSSM vectors to fixed-size embedding via BiLSTM.

    Architecture:
      1. Project scalar features to d_model dimension
      2. LayerNorm (pre-normalization)
      3. BiLSTM with num_layers stacked layers
      4. LayerNorm (post-normalization)
      5. Mean pool over sequence → fixed embedding

    Hidden size tuned so bidirectional output matches d_model:
      hidden_size = d_model // 2 → forward + backward = d_model

    Input: (B, 220) PsePSSM features
    Output: (B, 220, d_model) → after pooling: (B, d_model)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Project each scalar feature to d_model dimension
        self.project = nn.Linear(1, d_model)
        self.pre_norm = nn.LayerNorm(d_model)

        # BiLSTM: hidden_size = d_model // 2
        # Bidirectional concatenates forward + backward → output = d_model
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 220) PsePSSM features
        Returns:
            (B, 220, d_model) LSTM output sequence
        """
        # Reshape for projection: treat each feature independently
        x = x.unsqueeze(-1)  # (B, 220, 1)
        x = self.project(x)  # (B, 220, d_model)

        # Pre-norm before LSTM
        x = self.pre_norm(x)

        # BiLSTM processes sequence
        x, _ = self.bilstm(x)  # (B, 220, d_model)

        # Post-norm after LSTM
        x = self.post_norm(x)
        return x


class DrugEncoder(nn.Module):
    """
    Projects Morgan fingerprints (fixed-length bit-vectors) to dense embedding.

    Identical to Mamba version — fingerprints don't need sequential modeling.
    Two-layer MLP learns feature interactions before compression.

    Input: (B, 2048) binary {0., 1.} Morgan fingerprint vectors
    Output: (B, 128)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048) Morgan fingerprints
        Returns:
            (B, 128) drug embedding
        """
        return self.net(x)


class BiLSTMDTI(nn.Module):
    """
    Binary DTI classifier combining BiLSTM protein encoding and MLP drug encoding.

    Architecture:
      1. ProteinEncoder: PsePSSM (220) → BiLSTM → mean pool → (128)
      2. DrugEncoder: Morgan FP (2048) → MLP → (128)
      3. Concatenate: (256)
      4. Decoder: MLP (256 → 128 → 1)
      5. Output: binary logit (raw, not sigmoid-ed)

    Loss: BCEWithLogitsLoss (applies sigmoid + BCE internally)

    Ablation context:
      - MeanPool baseline: no sequential modeling
      - BiLSTM (this): recurrent sequential modeling
      - Mamba: SSM sequential modeling
      Compares whether sequential structure + recurrent vs. SSM inductive bias helps.
    """

    def __init__(
        self,
        drug_input_dim: int = 2048,
        d_model: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.1,
        embed_dim: int = 128,
        decoder_hidden: int = 128,
        decoder_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Protein encoder (BiLSTM) ───────────────────────────────────────
        self.protein_encoder = ProteinEncoder(
            d_model=d_model,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        # Project BiLSTM output to embedding dimension
        self.protein_proj = nn.Linear(d_model, embed_dim)
        self.protein_norm = nn.LayerNorm(embed_dim)

        # ── Drug encoder (MLP) ────────────────────────────────────────────
        self.drug_encoder = DrugEncoder(
            input_dim=drug_input_dim,
            hidden_dim=256,
            out_dim=embed_dim,
            dropout=decoder_dropout,
        )

        # ── Fusion decoder (MLP) ──────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(decoder_dropout),
            nn.Linear(decoder_hidden, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Xavier for Linear, Orthogonal for LSTM."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein: (B, 220) PsePSSM features
            drug: (B, 2048) Morgan fingerprints
        Returns:
            (B,) binary logits (pre-sigmoid)
        """
        # Protein: BiLSTM → mean pool → projection
        p = self.protein_encoder(protein)  # (B, 220, d_model)
        p = p.mean(dim=1)  # (B, d_model) mean pooling over sequence
        p = self.protein_proj(p)  # (B, embed_dim)
        p = self.protein_norm(p)  # (B, embed_dim)

        # Drug: MLP encoding
        d = self.drug_encoder(drug)  # (B, embed_dim)

        # Fusion: concatenate and decode
        x = torch.cat([p, d], dim=-1)  # (B, embed_dim * 2)
        logits = self.decoder(x)  # (B, 1)
        return logits.squeeze(-1)  # (B,)
