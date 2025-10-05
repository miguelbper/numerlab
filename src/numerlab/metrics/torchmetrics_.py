import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from numerlab.metrics.erawise import annual_return, corr, mmc, payoff


class Corr(Metric):
    """PyTorch Lightning metric for Numerai correlation.

    Accumulates target, predictions, meta model, and era data across
    batches and computes the Numerai correlation metric at the end of an
    epoch.
    """

    is_differentiable: bool | None = None
    higher_is_better: bool | None = True
    full_state_update: bool = True

    def __init__(self, **kwargs):
        """Initialize the Corr metric with state variables for accumulation."""
        super().__init__(**kwargs)
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("meta", default=[], dist_reduce_fx="cat")
        self.add_state("eras", default=[], dist_reduce_fx="cat")

    def update(self, target: Tensor, preds: Tensor, meta: Tensor, eras: Tensor) -> None:
        """Accumulate batch data for correlation computation.

        Args:
            target: Target values for the batch
            preds: Prediction values for the batch
            meta: Meta model predictions for the batch
            eras: Era identifiers for the batch
        """
        self.target.append(target)
        self.preds.append(preds)
        self.meta.append(meta)
        self.eras.append(eras)

    def compute(self) -> Tensor:
        """Compute the Numerai correlation metric from accumulated data.

        Returns:
            Correlation score as a tensor
        """
        target = dim_zero_cat(self.target).cpu().numpy()
        preds = dim_zero_cat(self.preds).cpu().numpy()
        meta = dim_zero_cat(self.meta).cpu().numpy()
        eras = dim_zero_cat(self.eras).cpu().numpy()
        return torch.tensor(corr(target, preds, meta, eras), dtype=torch.float32)


class MMC(Metric):
    """PyTorch Lightning metric for Meta Model Contribution (MMC).

    Accumulates target, predictions, meta model, and era data across
    batches and computes the MMC metric at the end of an epoch.
    """

    is_differentiable: bool | None = None
    higher_is_better: bool | None = True
    full_state_update: bool = True

    def __init__(self, **kwargs):
        """Initialize the MMC metric with state variables for accumulation."""
        super().__init__(**kwargs)
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("meta", default=[], dist_reduce_fx="cat")
        self.add_state("eras", default=[], dist_reduce_fx="cat")

    def update(self, target: Tensor, preds: Tensor, meta: Tensor, eras: Tensor) -> None:
        """Accumulate batch data for MMC computation.

        Args:
            target: Target values for the batch
            preds: Prediction values for the batch
            meta: Meta model predictions for the batch
            eras: Era identifiers for the batch
        """
        self.target.append(target)
        self.preds.append(preds)
        self.meta.append(meta)
        self.eras.append(eras)

    def compute(self) -> Tensor:
        """Compute the MMC metric from accumulated data.

        Returns:
            MMC score as a tensor
        """
        target = dim_zero_cat(self.target).cpu().numpy()
        preds = dim_zero_cat(self.preds).cpu().numpy()
        meta = dim_zero_cat(self.meta).cpu().numpy()
        eras = dim_zero_cat(self.eras).cpu().numpy()
        return torch.tensor(mmc(target, preds, meta, eras), dtype=torch.float32)


class Payoff(Metric):
    """PyTorch Lightning metric for Numerai payoff.

    Accumulates target, predictions, meta model, and era data across
    batches and computes the payoff metric at the end of an epoch.
    """

    is_differentiable: bool | None = None
    higher_is_better: bool | None = True
    full_state_update: bool = True

    def __init__(self, **kwargs):
        """Initialize the Payoff metric with state variables for
        accumulation."""
        super().__init__(**kwargs)
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("meta", default=[], dist_reduce_fx="cat")
        self.add_state("eras", default=[], dist_reduce_fx="cat")

    def update(self, target: Tensor, preds: Tensor, meta: Tensor, eras: Tensor) -> None:
        """Accumulate batch data for payoff computation.

        Args:
            target: Target values for the batch
            preds: Prediction values for the batch
            meta: Meta model predictions for the batch
            eras: Era identifiers for the batch
        """
        self.target.append(target)
        self.preds.append(preds)
        self.meta.append(meta)
        self.eras.append(eras)

    def compute(self) -> Tensor:
        """Compute the payoff metric from accumulated data.

        Returns:
            Payoff score as a tensor
        """
        target = dim_zero_cat(self.target).cpu().numpy()
        preds = dim_zero_cat(self.preds).cpu().numpy()
        meta = dim_zero_cat(self.meta).cpu().numpy()
        eras = dim_zero_cat(self.eras).cpu().numpy()
        return torch.tensor(payoff(target, preds, meta, eras), dtype=torch.float32)


class AnnualReturn(Metric):
    """PyTorch Lightning metric for annualized return.

    Accumulates target, predictions, meta model, and era data across
    batches and computes the annualized return at the end of an epoch.
    """

    is_differentiable: bool | None = None
    higher_is_better: bool | None = True
    full_state_update: bool = True

    def __init__(self, **kwargs):
        """Initialize the AnnualReturn metric with state variables for
        accumulation."""
        super().__init__(**kwargs)
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("meta", default=[], dist_reduce_fx="cat")
        self.add_state("eras", default=[], dist_reduce_fx="cat")

    def update(self, target: Tensor, preds: Tensor, meta: Tensor, eras: Tensor) -> None:
        """Accumulate batch data for annual return computation.

        Args:
            target: Target values for the batch
            preds: Prediction values for the batch
            meta: Meta model predictions for the batch
            eras: Era identifiers for the batch
        """
        self.target.append(target)
        self.preds.append(preds)
        self.meta.append(meta)
        self.eras.append(eras)

    def compute(self) -> Tensor:
        """Compute the annualized return from accumulated data.

        Returns:
            Annualized return rate as a tensor
        """
        target = dim_zero_cat(self.target).cpu().numpy()
        preds = dim_zero_cat(self.preds).cpu().numpy()
        meta = dim_zero_cat(self.meta).cpu().numpy()
        eras = dim_zero_cat(self.eras).cpu().numpy()
        return torch.tensor(annual_return(target, preds, meta, eras), dtype=torch.float32)
