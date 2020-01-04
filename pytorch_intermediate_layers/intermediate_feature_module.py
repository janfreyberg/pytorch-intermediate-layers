from typing import Union, Tuple, List, Dict
from typing_extensions import Literal

from collections import OrderedDict

import torch

from .recursive_getattr import _recursive_getattr


ReturnContainerTypes = Union[
    Literal["dict"], Literal["list"], Literal["tensor"]
]


class IntermediateFeatureModule(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        feature_layers: Union[str, List[str]],
        dummy_input_size: Tuple[int],
        return_type: ReturnContainerTypes = "dict",
    ):
        """A module that extracts intermediate output from another module.

        If you have someone else's pytorch module and want to extract inter-
        mediate output (e.g. to make a Feature Pyramid), use this class.

        Parameters
        ----------
        base_model : torch.nn.Module
            The module you want to extract intermediate data from.
        feature_layers : Union[str, Sequence[str]]
            The layer(s) that you want to extract. Either a single layer name,
            or a list of layer names. This can be specified hierarchically, eg.
            'layer4.relu' or 'backbone.feature_extractor.0' - specify the names
            of layers either as the name it has as an attribute of the parent,
            or (for e.g. Sequential models or ModelLists) as an integer.
        dummy_input_size : Tuple[int], optional
            In order to determine the output of this module, a dummy input will
            be run through the base_model. Specify the size here (without the
            batch dimension, as this will always be 1).

            For example, for images, you may want to choose (3, 512, 512).

        return_type : str, optional
            How you want the data to be returned - one of:
            'dict' - a dictionary like {layer_name: <output_tensor>, ...}
            'list' - a list in order of the names passed in, e.g.
                [<layer1_output>, ...]
        """
        super().__init__()
        self.base_model = base_model
        self.cache: Dict[str, torch.Tensor] = OrderedDict()
        self.return_type = return_type

        if isinstance(feature_layers, str):
            self.feature_layers = [feature_layers]
            self.return_type = "tensor"
        else:
            self.feature_layers = list(feature_layers)

        for layer_name in self.feature_layers:

            layer = _recursive_getattr(self.base_model, layer_name)
            # register the hook to cache this layer:
            self._cache_activations(layer, layer_name)

        # run a dummy input through the network:
        self.eval()  # in case this model behaves weirdly during training
        with torch.no_grad():
            dummy_input = torch.randn(*(1, *dummy_input_size))
            dummy_output = self.forward(dummy_input)
            if self.return_type == "list":
                dummy_output_shapes = [t.shape[1:] for t in dummy_output]
            elif self.return_type == "dict":
                dummy_output_shapes = [
                    t.shape[1:] for t in dummy_output.values()
                ]
            elif self.return_type == "tensor":
                dummy_output_shapes = [dummy_output.shape[1:]]

        if len(dummy_input_size) >= 2:
            # 2 / 3 / 4D input, probably convolutional
            self.input_channels = dummy_input_size[0]
            self.output_channels: List[int] = [
                shape[0] for shape in dummy_output_shapes
            ]
            self.size_factors: List[float] = []
            input_size = dummy_input_size[-1]
            for shape in dummy_output_shapes:
                self.size_factors.append(shape[-1] / input_size)
                input_size = shape[-1]
            if self.return_type == "tensor":
                self.output_channels = self.output_channels[0]
        elif len(dummy_input_size) == 1:
            # 1D or linear case
            self.in_features = dummy_input_size[0]
            self.out_features: List[int] = [
                shape[0] for shape in dummy_output_shapes
            ]
            if self.return_type == "tensor":
                self.out_features = self.out_features[0]

        self.train()

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Union[Dict[str, torch.Tensor], List[torch.Tensor]]:

        cache_keys = self.cache_keys()
        if not all(key in self.cache for key in cache_keys):
            # run forward pass of base_model
            self.base_model(x)

        output = OrderedDict(
            [
                (feature_layer, self.cache[cache_key])
                for feature_layer, cache_key in zip(
                    self.feature_layers, cache_keys
                )
            ]
        )

        # remove only this device's cached results:
        for cache_key in cache_keys:
            self.cache.pop(cache_key)

        output = self._format_output(output)
        return output

    def _cache_activations(self, layer: torch.nn.Module, layer_name: str):
        """take in a layer, and add a hook to it that caches its output"""

        def hook(model, input, output):
            device = output.device
            self.cache[f"{layer_name}_{device}"] = output

        layer.register_forward_hook(hook)

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    def cache_keys(self) -> List[str]:
        # keys for the cache need to be unique across devices
        device = self.device
        return [
            f"{feature_layer}_{device}"
            for feature_layer in self.feature_layers
        ]

    def _format_output(self, output: Dict[str, torch.Tensor]):
        if self.return_type == "list":
            return list(output.values())
        elif self.return_type == "dict":
            return output
        elif self.return_type == "tensor":
            return torch.cat(tuple(output.values()), 1)
