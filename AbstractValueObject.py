from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union


class AbstractValueObject(ABC):
    def __init__(self, value: np.ndarray, expected_shape: Union[Tuple[int, ...], Tuple[None, int]]):
        self.value           = value
        self._expected_shape = expected_shape
        self.validate()

    def validate(self):
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy ndarray")

        # 形状の長さが一致するかをまず確認
        if len(self._expected_shape) != len(self.value.shape):
            raise ValueError(f"Expected shape {self._expected_shape}, but got {self.value.shape}")

        # 各次元のチェック、None は任意のサイズとして許可
        for exp_dim, actual_dim in zip(self._expected_shape, self.value.shape):
            if exp_dim is not None and exp_dim != actual_dim:
                raise ValueError(f"Expected shape {self._expected_shape}, but got {self.value.shape}")

    @abstractmethod
    def __repr__(self):
        pass
