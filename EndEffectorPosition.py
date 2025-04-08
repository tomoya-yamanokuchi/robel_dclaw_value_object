from dataclasses import dataclass
import numpy as np
from .AbstractValueObject import AbstractValueObject


@dataclass(frozen=True)
class EndEffectorPosition(AbstractValueObject):
    value : np.ndarray

    def __post_init__(self):
        # 入力データの形状を常に (N, 6) に変換する
        if self.value.shape == (9,):
            value = np.expand_dims(self.value, axis=0)
        elif len(self.value.shape) == 2 and self.value.shape[1] == 9:
            value = self.value
        else:
            raise ValueError(f"Expected shape (9,) or (N, 9), but got {self.value.shape}")

        object.__setattr__(self, 'value', value)
        object.__setattr__(self, '_expected_shape', (None, 9))
        self.validate()


if __name__ == '__main__':
    import numpy as np

    tp1 = EndEffectorPosition(value=np.random.randn(9))
    print(tp1)
