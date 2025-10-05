import json
from typing import Literal

import polars as pl
import polars.selectors as cs
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from lightning.fabric.utilities.data import suggested_max_num_workers
from rootutils import find_root
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from numerlab.utils.splits import TRAIN_ERA_1

N = 5


class AllowTF32:
    """Context manager for enabling/disabling TF32 for CUDA matrix
    multiplications.

    TF32 (Tensor Float 32) can provide faster matrix multiplications on
    Ampere GPUs (RTX 30/40 series) while maintaining sufficient
    precision for many use cases.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the context manager.

        Args:
            enabled: Whether to enable TF32 (True) or disable it (False)
        """
        self.enabled = enabled
        self.old_value = None

    def __enter__(self):
        """Enter the context, saving the old value and setting the new one."""
        self.old_value = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.enabled
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring the old value."""
        if self.old_value is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.old_value


def count_X(X: torch.Tensor) -> torch.Tensor:
    return torch.sum(F.one_hot(X.to(torch.int64), num_classes=N), dim=0)


def count_XY(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """count_XY[j, x, y] = #{i | X[i, j] = x and Y[i] = y}"""
    _, J = X.shape
    X_one_hot = F.one_hot(X.to(torch.int64), num_classes=N).to(torch.float32)
    Y_one_hot = F.one_hot(Y.to(torch.int64), num_classes=N).to(torch.float32)
    X_rearranged = rearrange(X_one_hot, "I J N -> (J N) I")
    C = rearrange(X_rearranged @ Y_one_hot, "(J X) N -> J X N", J=J).to(torch.int64)
    return C


def count_XX(X: torch.Tensor) -> torch.Tensor:
    _, J = X.shape
    X_one_hot = F.one_hot(X.to(torch.int64), num_classes=N).to(torch.float32)
    XL = rearrange(X_one_hot, "I J U -> (J U) I")
    XR = rearrange(X_one_hot, "I K V -> I (K V)")
    C = rearrange(XL @ XR, "(J U) (K V) -> J K U V", J=J, K=J).to(torch.int64)
    return C


def count_XXY(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    _, J = X.shape
    C = torch.zeros((J, J, N, N, N), dtype=torch.int64, device=X.device)

    for y in range(N):
        mask = y == Y
        if torch.any(mask):
            C[..., y] = count_XX(X[mask])

    return C


def count_XY_from_XXY(C_XXY: torch.Tensor) -> torch.Tensor:
    return torch.sum(C_XXY[0], dim=1)


def count_XX_from_XXY(C_XXY: torch.Tensor) -> torch.Tensor:
    return torch.sum(C_XXY, dim=-1)


def count_XXY_streaming(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> torch.Tensor:
    _, J = X.shape
    C = torch.zeros((J, J, N, N, N), dtype=torch.int64, device=device)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    for x, y in tqdm(dataloader, desc="Counting XXY"):
        x = x.to(device=device, non_blocking=True)  # (B, J)
        y = y.to(device=device, non_blocking=True)  # (B,)

        x_ohe = F.one_hot(x.to(torch.int64), num_classes=N).to(torch.float32)  # (B, J, U)

        for y_value in range(N):
            mask = y == y_value  # (B,)
            if torch.any(mask):
                x_mask = x_ohe[mask]  # (I, J, U)
                x_view = rearrange(x_mask, "I J U -> I (J U)")
                S = x_view.T @ x_view  # (J U, J U)
                C[..., y_value] += rearrange(S, "(J U) (K V) -> J K U V", J=J, K=J).to(torch.int64)

    return C


def get_MI_XY(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    C_XY = count_XY(X, Y).to(torch.float32)
    return get_MI_XY_from_counts(C_XY)


def get_MI_XY_from_counts(C_XY: torch.Tensor) -> torch.Tensor:
    J = torch.sum(C_XY[0])
    P_XY = C_XY / J
    P_X = reduce(P_XY, "J X Y -> J X 1", "sum")
    P_Y = reduce(P_XY[0], "X Y -> 1 1 Y", "sum")

    Z = torch.zeros_like(C_XY)
    A = P_XY * torch.log(P_XY / (P_X * P_Y))
    B = torch.where(C_XY == 0, Z, A)
    return reduce(B, "J X Y -> J", "sum")


def get_MI_XX(X: torch.Tensor) -> torch.Tensor:
    C_XX = count_XX(X).to(torch.float32)
    return get_MI_XX_from_counts(C_XX)


def get_MI_XX_from_counts(C_XX: torch.Tensor) -> torch.Tensor:
    J = torch.sum(C_XX[0, 0])
    P_XX = C_XX / J
    P_X = reduce(P_XX[0], "K U V -> K V", "sum")

    Z = torch.zeros_like(C_XX)
    A = P_XX * torch.log(P_XX / (rearrange(P_X, "J U -> J 1 U 1") * rearrange(P_X, "K V -> 1 K 1 V")))
    B = torch.where(C_XX == 0, Z, A)
    return reduce(B, "J K U V -> J K", "sum")


def get_MI_XXY(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    C_XXY = count_XXY(X, Y).to(torch.float32)
    return get_MI_XXY_from_counts(C_XXY)


def get_MI_XXY_from_counts(C_XXY: torch.Tensor) -> torch.Tensor:
    J = torch.sum(C_XXY[0, 0])
    P_XXY = C_XXY / J
    P_XY = reduce(P_XXY[0], "K U V Y -> K V Y", "sum")
    P_Y = reduce(P_XY[0], "V Y -> Y", "sum")

    Z = torch.zeros_like(C_XXY)
    A = P_XXY * torch.log(P_Y * P_XXY / (rearrange(P_XY, "J U Y -> J 1 U 1 Y") * rearrange(P_XY, "K V Y -> 1 K 1 V Y")))
    B = torch.where(C_XXY == 0, Z, A)
    return reduce(B, "J K U V Y -> J K", "sum")


def sort_features(
    features: list[str],
    MI_XY: torch.Tensor,
    MI_XX: torch.Tensor,
    MI_XX_Y: torch.Tensor,
    method: Literal["CIFE", "JMI", "CMIM", "JMIM", "DMIM"] = "DMIM",
    device: torch.device | None = None,
) -> list[str]:
    device = device or torch.device("cpu")
    n = len(features)

    # order[feature_idx] = order_idx, in the sorted list, the feature with feature_idx will be at position order_idx
    order = -torch.ones(n, dtype=torch.int32, device=device)

    for order_idx in tqdm(range(n), desc="Sorting features"):
        S = order != -1  # S = mask / set of already selected features

        MI_XY_S = rearrange(MI_XY, "K -> 1 K")[:, S]
        MI_XX_S = MI_XX[:, S]
        MI_XX_Y_S = MI_XX_Y[:, S]

        if not torch.any(S):
            objective = MI_XY
        else:
            match method:
                case "CIFE":
                    objective = MI_XY - reduce(MI_XX_S - MI_XX_Y_S, "J K -> J", "sum")
                case "JMI":
                    objective = MI_XY - reduce(MI_XX_S - MI_XX_Y_S, "J K -> J", "mean")
                case "CMIM":
                    objective = MI_XY - reduce(MI_XX_S - MI_XX_Y_S, "J K -> J", "max")
                case "JMIM":
                    objective = MI_XY - reduce(MI_XX_S - MI_XX_Y_S - MI_XY_S, "J K -> J", "max")
                case "DMIM":
                    objective = MI_XY - reduce(MI_XX_S, "J K -> J", "max") + reduce(MI_XX_Y_S, "J K -> J", "max")

        objective = torch.where(~S, objective, -torch.inf)
        feature_idx = torch.argmax(objective)
        order[feature_idx] = order_idx

    sorted_features = [None for _ in range(n)]
    for feature_idx, order_idx in enumerate(order):
        sorted_features[order_idx] = features[feature_idx]

    return sorted_features


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True

    root_dir = find_root()
    data_dir = root_dir / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pl.scan_parquet(data_dir / "numerai.parquet").filter(pl.col("era") < TRAIN_ERA_1)
    features = df.select(cs.starts_with("feature")).collect_schema().names()

    X = df.select(features).collect().to_torch().to(dtype=torch.int8)
    Y = df.select(["target"]).collect().to_torch().to(dtype=torch.float32).squeeze()
    Y = (4 * Y).to(dtype=torch.int8)

    num_workers = suggested_max_num_workers(1)

    with AllowTF32(True):
        C_XXY = count_XXY_streaming(
            X,
            Y,
            device=device,
            batch_size=2**12,
            num_workers=num_workers,
            pin_memory=True,
        )

    C_XX = count_XX_from_XXY(C_XXY)
    C_XY = count_XY_from_XXY(C_XXY)

    MI_XY = get_MI_XY_from_counts(C_XY)
    MI_XX = get_MI_XX_from_counts(C_XX)
    MI_XX_Y = get_MI_XXY_from_counts(C_XXY)

    for method in ["CIFE", "JMI", "CMIM", "JMIM", "DMIM"]:
        print(f"Sorting features with {method}")
        sorted_features = sort_features(features, MI_XY, MI_XX, MI_XX_Y, method, device)

        print(f"Writing {method} features to {data_dir}/extra_feature_sets.json")

        try:
            with open(data_dir / "extra_feature_sets.json") as f:
                feature_sets = json.load(f)
        except FileNotFoundError:
            feature_sets = {}

        feature_sets[method.lower()] = sorted_features

        with open(data_dir / "extra_feature_sets.json", "w") as f:
            json.dump(feature_sets, f, indent=2)


if __name__ == "__main__":
    main()
