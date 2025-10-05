from abc import ABC, abstractmethod

from lightning import LightningDataModule

from numerlab.utils.types import Data


class SKLearnDataModule(ABC):
    """Abstract base class for data modules in the sklearn framework.

    This class defines the interface for data modules that provide access to
    training, validation, and test datasets. Each dataset is returned as a
    tuple containing era indices, features, targets, and meta model predictions.

    The Data type is defined as:
        Data = tuple[NDArray[np.int8], NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]
        where the elements are: (features, targets, meta_model, eras)
    """

    @abstractmethod
    def train_dataset(self) -> Data:
        """Get the training dataset.

        Returns:
            Data: A tuple containing:
                - era: Era indices for training
                - features: Input features for training
                - target: Target values for training
                - meta: Meta model predictions for training
        """
        pass  # pragma: no cover

    @abstractmethod
    def val_dataset(self) -> Data:
        """Get the validation dataset.

        Returns:
            Data: A tuple containing:
                - era: Era indices for validation
                - features: Input features for validation
                - target: Target values for validation
                - meta: Meta model predictions for validation
        """
        pass  # pragma: no cover

    @abstractmethod
    def test_dataset(self) -> Data:
        """Get the test dataset.

        Returns:
            Data: A tuple containing:
                - era: Era indices for testing
                - features: Input features for testing
                - target: Target values for testing
                - meta: Meta model predictions for testing
        """
        pass  # pragma: no cover


DataModule = LightningDataModule | SKLearnDataModule
