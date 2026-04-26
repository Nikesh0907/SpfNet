import numpy as np


def unfold(tensor, mode=0):
    tensor = np.asarray(tensor)
    tensor = np.moveaxis(tensor, mode, 0)
    return tensor.reshape(tensor.shape[0], -1)


class _TenAlg:
    @staticmethod
    def mode_dot(tensor, matrix, mode=0):
        tensor = np.asarray(tensor)
        matrix = np.asarray(matrix)
        tensor = np.moveaxis(tensor, mode, 0)
        result = np.tensordot(matrix, tensor, axes=(1, 0))
        return np.moveaxis(result, 0, mode)


tenalg = _TenAlg()
