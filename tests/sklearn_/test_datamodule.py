import numpy as np
import pytest

from numerlab.sklearn_.datamodule import SKLearnDataModule
from numerlab.utils.types import Data

NUM_TRAIN_SAMPLES = 10
NUM_VAL_SAMPLES = 2
NUM_TEST_SAMPLES = 2
NUM_FEATURES = 5
NUM_CLASSES = 2
NUM_ERAS = 2

X_train = np.random.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES)
y_train = np.random.randint(0, NUM_CLASSES, NUM_TRAIN_SAMPLES)
m_train = np.random.rand(NUM_TRAIN_SAMPLES)
e_train = np.random.randint(0, NUM_ERAS, NUM_TRAIN_SAMPLES)

X_val = np.random.rand(NUM_VAL_SAMPLES, NUM_FEATURES)
y_val = np.random.randint(0, NUM_CLASSES, NUM_VAL_SAMPLES)
m_val = np.random.rand(NUM_VAL_SAMPLES)
e_val = np.random.randint(0, NUM_ERAS, NUM_VAL_SAMPLES)

X_test = np.random.rand(NUM_TEST_SAMPLES, NUM_FEATURES)
y_test = np.random.randint(0, NUM_CLASSES, NUM_TEST_SAMPLES)
m_test = np.random.rand(NUM_TEST_SAMPLES)
e_test = np.random.randint(0, NUM_ERAS, NUM_TEST_SAMPLES)


class CompleteDataModule(SKLearnDataModule):
    def train_dataset(self) -> Data:
        return X_train, y_train, m_train, e_train

    def val_dataset(self) -> Data:
        return X_val, y_val, m_val, e_val

    def test_dataset(self) -> Data:
        return X_test, y_test, m_test, e_test


class IncompleteDataModule(SKLearnDataModule):
    def train_dataset(self) -> Data:
        return X_train, y_train, m_train, e_train

    def val_dataset(self) -> Data:
        return X_val, y_val, m_val, e_val

    # test_set method is intentionally missing


class TestDataModule:
    def test_complete_datamodule(self):
        dm = CompleteDataModule()

        X_train, y_train, m_train, e_train = dm.train_dataset()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(m_train, np.ndarray)
        assert isinstance(e_train, np.ndarray)
        assert X_train.shape == (NUM_TRAIN_SAMPLES, NUM_FEATURES)
        assert y_train.shape == (NUM_TRAIN_SAMPLES,)
        assert m_train.shape == (NUM_TRAIN_SAMPLES,)
        assert e_train.shape == (NUM_TRAIN_SAMPLES,)

        X_val, y_val, m_val, e_val = dm.val_dataset()
        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert isinstance(m_val, np.ndarray)
        assert isinstance(e_val, np.ndarray)
        assert X_val.shape == (NUM_VAL_SAMPLES, NUM_FEATURES)
        assert y_val.shape == (NUM_VAL_SAMPLES,)
        assert m_val.shape == (NUM_VAL_SAMPLES,)
        assert e_val.shape == (NUM_VAL_SAMPLES,)

        X_test, y_test, m_test, e_test = dm.test_dataset()
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(m_test, np.ndarray)
        assert isinstance(e_test, np.ndarray)
        assert X_test.shape == (NUM_TEST_SAMPLES, NUM_FEATURES)
        assert y_test.shape == (NUM_TEST_SAMPLES,)
        assert m_test.shape == (NUM_TEST_SAMPLES,)
        assert e_test.shape == (NUM_TEST_SAMPLES,)

    def test_incomplete_datamodule(self):
        with pytest.raises(TypeError):
            IncompleteDataModule()

    def test_cannot_instantiate_base_datamodule(self):
        with pytest.raises(TypeError):
            SKLearnDataModule()
