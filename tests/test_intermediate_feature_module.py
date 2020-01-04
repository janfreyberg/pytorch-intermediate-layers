import torch

from pytorch_intermediate_layers import IntermediateFeatureModule


@torch.no_grad()
def test_linear_layer_shapes_match():

    nn = torch.nn.Sequential(
        torch.nn.Linear(48, 24),
        torch.nn.Linear(24, 12),
        torch.nn.Linear(12, 6),
        torch.nn.Linear(6, 3),
        torch.nn.Linear(3, 1),
    )

    layers = ["0", "1", "2"]
    expected_output_size = {k: v for k, v in zip(layers, (24, 12, 6))}
    nn = IntermediateFeatureModule(nn, layers, dummy_input_size=(48,))
    test_input = torch.randn(4, 48)
    test_output = nn(test_input)

    for layer in layers:
        assert test_output[layer].shape[0] == 4
        assert test_output[layer].shape[1] == expected_output_size[layer]


@torch.no_grad()
def test_linear_layer_values_match():

    nn = torch.nn.Sequential(
        torch.nn.Linear(48, 24),
        torch.nn.Linear(24, 12),
        torch.nn.Linear(12, 6),
        torch.nn.Linear(6, 3),
        torch.nn.Linear(3, 1),
    )

    layers = ["0", "1", "2"]
    nn = IntermediateFeatureModule(nn, layers, dummy_input_size=(48,))
    test_input = torch.randn(4, 48)
    test_output = nn(test_input)

    x = test_input
    for layer in layers:
        x = nn.base_model[int(layer)](x)
        assert (test_output[layer] == x).all()


@torch.no_grad()
def test_different_return_types():

    base_nn = torch.nn.Sequential(
        torch.nn.Linear(48, 24),
        torch.nn.Linear(24, 12),
        torch.nn.Linear(12, 6),
        torch.nn.Linear(6, 3),
        torch.nn.Linear(3, 1),
    )

    layers = ["0", "1", "2"]
    test_input = torch.randn(4, 48)

    nn = IntermediateFeatureModule(
        base_nn, layers, dummy_input_size=(48,), return_type="dict"
    )
    test_output = nn(test_input)
    assert isinstance(test_output, dict) and set(layers) == set(
        test_output.keys()
    )

    nn = IntermediateFeatureModule(
        base_nn, layers, dummy_input_size=(48,), return_type="list"
    )
    test_output = nn(test_input)
    assert isinstance(test_output, list)

    nn = IntermediateFeatureModule(
        base_nn, layers, dummy_input_size=(48,), return_type="tensor"
    )
    test_output = nn(test_input)
    assert isinstance(test_output, torch.Tensor)
    assert test_output.size(1) == (24 + 12 + 6)


@torch.no_grad()
def test_single_intermediate_layer():
    base_nn = torch.nn.Sequential(
        torch.nn.Linear(48, 24),
        torch.nn.Linear(24, 12),
        torch.nn.Linear(12, 6),
        torch.nn.Linear(6, 3),
        torch.nn.Linear(3, 1),
    )
    layer = "0"
    test_input = torch.randn(4, 48)
    nn = IntermediateFeatureModule(
        base_nn, layer, dummy_input_size=test_input.shape[1:]
    )
    test_output = nn(test_input)
    assert isinstance(test_output, torch.Tensor)
    assert test_output.size(1) == 24
    assert nn.return_type == "tensor"


@torch.no_grad()
def test_convolutional_models_and_inputs():
    base_nn = torch.nn.Sequential(
        torch.nn.Conv2d(3, 12, 3, 1),
        torch.nn.Conv2d(12, 12, 3, 1),
        torch.nn.Conv2d(12, 12, 3, 1),
    )
    layer = "0"
    test_input = torch.randn(4, 3, 512, 512)
    nn = IntermediateFeatureModule(
        base_nn, layer, dummy_input_size=test_input.shape[1:]
    )
    test_output = nn(test_input)

    assert nn.output_channels == base_nn[0].out_channels
    assert (test_output == base_nn[0](test_input)).all()
