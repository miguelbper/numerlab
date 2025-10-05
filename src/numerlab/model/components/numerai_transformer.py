from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor, nn


class Tokenizer(nn.Module):
    """Tokenization layer for Numerai features.

    This module converts integer feature values into learned embeddings.
    It uses a learnable embedding table to map feature values to dense vectors.

    Args:
        num_tokens: Number of features (tokens) in the input sequence
        dim: Dimension of the token embeddings
        num_values: Number of possible values each feature can take
    """

    def __init__(self, num_tokens: int, dim: int, num_values: int) -> None:
        """Initialize the Tokenizer with learnable embeddings.

        Args:
            num_tokens: Number of features (tokens) in the input sequence
            dim: Dimension of the token embeddings
            num_values: Number of possible values each feature can take
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.num_values = num_values
        self.embedding = nn.Parameter(torch.randn(num_values, num_tokens, dim))  # (N, T, D)

    def forward(self, x: Tensor) -> Tensor:
        """Convert integer feature values to embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing integer feature values

        Returns:
            Token embeddings of shape (batch_size, seq_len, dim)

        Raises:
            ValueError: If x contains non-integer values or values outside [0, num_values)
            ValueError: If x has incorrect sequence length
        """
        # x.shape = (batch_size, seq_len) = (B, T)
        # y.shape = (batch_size, seq_len, dim) = (B, T, D)
        # self.embedding.shape = (num_values, num_tokens, dim) = (N, T, D)
        x_is_integers = x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        x_is_bounded = torch.all(x >= 0) and torch.all(x < self.num_values)
        if not (x_is_integers and x_is_bounded):
            raise ValueError("x must be integers between 0 and num_values")

        if not x.shape[1] == self.num_tokens:
            raise ValueError(f"x should have {self.num_tokens} tokens, but got {x.shape[1] = }")

        _, _, D = self.embedding.shape
        return torch.gather(input=self.embedding, dim=0, index=repeat(x.to(torch.int64), "b t -> b t d", d=D))


class RegressionHead(nn.Module):
    """Regression head for the Numerai transformer.

    This module flattens the transformer output and applies a linear layer
    to produce final predictions.

    Args:
        num_tokens: Number of tokens from the transformer
        dim: Dimension of each token embedding
        num_targets: Number of target values to predict
    """

    def __init__(self, num_tokens: int, dim: int, num_targets: int) -> None:
        """Initialize the regression head.

        Args:
            num_tokens: Number of tokens from the transformer
            dim: Dimension of each token embedding
            num_targets: Number of target values to predict
        """
        super().__init__()
        self.linear = nn.Linear(num_tokens * dim, num_targets)

    def forward(self, x: Tensor) -> Tensor:
        """Apply regression head to transformer output.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, dim)

        Returns:
            Predictions of shape (batch_size, num_targets)
        """
        return self.linear(x.flatten(start_dim=1))


class NumeraiTransformer(nn.Module):
    """Transformer model for Numerai prediction.

    This model uses a transformer architecture to predict Numerai targets.
    It consists of a tokenizer to embed features, a transformer encoder,
    and a regression head for final predictions.

    Args:
        num_features: Number of input features
        num_targets: Number of target values to predict (default: 1)
        num_values: Number of possible values each feature can take (default: 5)
        d_model: Dimension of the transformer model (default: 512)
        nhead: Number of attention heads (default: 8)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: F.gelu)
        layer_norm_eps: Epsilon for layer normalization (default: 1e-5)
        batch_first: Whether batch dimension comes first (default: False)
        norm_first: Whether to apply normalization first (default: False)
        bias: Whether to use bias in linear layers (default: True)
        device: Device to place the model on
        dtype: Data type for the model
        num_layers: Number of transformer layers (default: 12)
        enable_nested_tensor: Whether to enable nested tensors (default: True)
        mask_check: Whether to check for padding masks (default: False)
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int = 1,
        num_values: int = 5,
        d_model: int = 64,  # Original: 512
        nhead: int = 4,  # Original: 8
        dim_feedforward: int = 256,  # Original: 2048
        dropout: float = 0.0,  # Original: 0.1
        activation: str | Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        num_layers: int = 6,  # Original: 12
        enable_nested_tensor: bool = True,
        mask_check: bool = False,
    ) -> None:
        """Initialize the Numerai transformer model.

        Args:
            num_features: Number of input features
            num_targets: Number of target values to predict (default: 1)
            num_values: Number of possible values each feature can take (default: 5)
            d_model: Dimension of the transformer model (default: 512)
            nhead: Number of attention heads (default: 8)
            dim_feedforward: Dimension of feedforward network (default: 2048)
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: F.gelu)
            layer_norm_eps: Epsilon for layer normalization (default: 1e-5)
            batch_first: Whether batch dimension comes first (default: False)
            norm_first: Whether to apply normalization first (default: False)
            bias: Whether to use bias in linear layers (default: True)
            device: Device to place the model on
            dtype: Data type for the model
            num_layers: Number of transformer layers (default: 12)
            enable_nested_tensor: Whether to enable nested tensors (default: True)
            mask_check: Whether to check for padding masks (default: False)
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.net = nn.Sequential(
            Tokenizer(
                num_tokens=num_features,
                dim=d_model,
                num_values=num_values,
            ),
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=enable_nested_tensor,
                mask_check=mask_check,
            ),
            RegressionHead(
                num_tokens=num_features,
                dim=d_model,
                num_targets=num_targets,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the transformer model.

        Args:
            x: Input tensor of shape (batch_size, num_features) containing integer feature values

        Returns:
            Predictions of shape (batch_size, num_targets)
        """
        return self.net(x)
