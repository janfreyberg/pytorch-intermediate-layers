import torch
from pytorch_intermediate_layers.recursive_getattr import (
    _recursive_getattr,
    _is_integer,
)


def test_is_integer():
    assert _is_integer("5") is True
    assert _is_integer("layer_2") is False


def test_that_getting_named_layer_works():
    class SubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(5, 3)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(10, 5)
            self.l2 = SubModel()

    model = Model()

    assert _recursive_getattr(model, "l1") is model.l1
    assert _recursive_getattr(model, "l2.l1") is model.l2.l1


def test_that_getting_integer_layers_works():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Sequential(
                torch.nn.Linear(5, 3), torch.nn.Linear(3, 1)
            )

    model = Model()
    assert _recursive_getattr(model, "l1.0") is model.l1[0]
