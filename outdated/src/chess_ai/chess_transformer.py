import torch
import torch.nn as nn
from torch import optim
from lightning import LightningModule


class ChessTransformer(LightningModule):
    def __init__(
        self,
        vocab_size: int = 13,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super(ChessTransformer, self).__init__()
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(65, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.policy_head = nn.Linear(embed_dim, 4096)
        self.value_head = nn.Linear(embed_dim, 1)
        self.loss_fn_policy = nn.CrossEntropyLoss()
        self.loss_fn_value = nn.MSELoss()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.token_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        cls_representation = x[:, 0, :]
        policy_logits = self.policy_head(cls_representation)
        value = self.value_head(cls_representation)
        return policy_logits, value

    def training_step(self, batch, batch_idx):
        boards = batch["board"]
        policy_target = batch["policy_target"]
        value_target = batch["value_target"]
        policy_logits, value_pred = self(boards)
        loss_policy = self.loss_fn_policy(policy_logits, policy_target)
        loss_value = self.loss_fn_value(value_pred, value_target)
        loss = loss_policy + loss_value
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_policy_loss", loss_policy, prog_bar=True)
        self.log("train_value_loss", loss_value, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
