from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn


class RNN:
    def __init__(self, num_neurons: int, vision_resolution: int) -> None:
        """Parallelised RNN
        Args:
            num_neurons (int): The number of neurons-per-RNN
            vision_resolution (int): The resolution used for vision
        """
        self.num_neurons = num_neurons
        self.vision_resolution = vision_resolution

        # Pre-compute some values about parameter dimensions
        # Input matrix has shape [vision, number of neurons]
        self.input_shape = (self.vision_resolution, self.num_neurons)
        # Recurrent connection has shape [number of neurons, number of neurons]
        self.recurrent_shape = (self.num_neurons, self.num_neurons)
        # Hidden layer's bias has shape [number of neurons]
        self.hidden_bias_shape = (self.num_neurons,)
        # Output matrix has shape [number of neurons, 1]
        self.output_shape = (self.num_neurons, 1)
        # Output layer's bias is a single variable
        self.output_bias_shape = (1,)

        self.variable_shapes = [
            self.input_shape,
            self.recurrent_shape,
            self.hidden_bias_shape,
            self.output_shape,
            self.output_bias_shape,
        ]

        # Initialise the activation
        self.act = nn.Sequential(nn.ELU(), nn.LayerNorm((num_neurons,), elementwise_affine=False))

        self.num_parallel: int = 0

    def step(
        self, vision: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step the network
        Args:
            vision (torch.Tensor): A tensor of shape [num_parallel, self.vision_resolution] of vision inputs
            hidden_state (Optional[torch.Tensor]): A tensor of shape [num_parallel, self.num_neurons] of hidden state variables, or None (no current hidden state)
        Args:
            actions (torch.Tensor): A tensor of shape [num_parallel] of actions
            hidden_state (torch.Tensor): A tensor of shape [num_parallel, self.num_neurons] of hidden state variables
        """

        num_parallel, res = vision.shape
        assert num_parallel == self.num_parallel
        assert res == self.vision_resolution

        # Instantiate hidden state
        if hidden_state is None:
            hidden_state = self.act(torch.randn((self.num_parallel, self.num_neurons), device=vision.device))

        num_parallel, num_neurons = hidden_state.shape
        assert num_parallel == self.num_parallel
        assert num_neurons == self.num_neurons

        # Add gaussian noise
        network_input = vision

        # Apply network
        input_hidden = torch.bmm(network_input.unsqueeze(1), self.input)[:, 0]
        rec_hidden = torch.bmm(hidden_state.unsqueeze(1), self.rec)[:, 0]
        hidden_pre_act = input_hidden + rec_hidden + self.bias
        # Apply hidden act
        hidden_state = self.act(hidden_pre_act)

        # Apply output
        hidden_out = torch.bmm(hidden_state.unsqueeze(1), self.out)[:, 0]
        out_pre_act = hidden_out + self.out_bias

        actions = torch.tanh(out_pre_act[:, 0])

        return actions, hidden_state

    @property
    def variable_sizes(self) -> List[int]:
        """Total number of parameters of variable shapes"""
        return [int(np.prod(shape)) for shape in self.variable_shapes]

    @property
    def num_parameters(self) -> int:
        """Total number of parameters of policy"""
        return int(np.sum(self.variable_sizes))

    def extract_variables(self, parameters: torch.Tensor) -> List[torch.Tensor]:
        """Extract variables from a given set of parameters
        Args:
            parameters (parameters): A tensor of shape [num_parallel, self.num_parameters] of parameters to extract from
        Returns:
            variables (List[torch.Tensor]): A list of tensors of shape [num_parallel, *shape] for each variable shape in self.variable_shapes
        """
        # Iterate over the parameters to group sub-sets into variables
        num_parallel = parameters.shape[0]
        up_to_param = 0
        variables = []
        for shape, size in zip(self.variable_shapes, self.variable_sizes):
            variables.append(parameters[:, up_to_param : up_to_param + size].reshape(num_parallel, *shape))
            up_to_param += size
        return variables

    def load_parameters(self, parameters: torch.Tensor) -> None:
        """Load parameters from a tensor
        Args:
            parameters (parameters): A tensor of shape [num_parallel, self.num_parameters] of parameters to load
        """
        # Clone the parameters
        parameters = parameters.clone()

        # Get the number of parallel networks implied by the parameters tensor
        self.num_parallel = parameters.shape[0]
        # Ensure that the parameters second dimension is equal to the number of parameters of the RNN
        assert parameters.shape[1] == self.num_parameters

        # Separate the parameters into individual sets of variables
        variables = self.extract_variables(parameters)

        # Store the tensors
        self.input = variables[0]
        self.rec = variables[1]
        self.bias = variables[2]
        self.out = variables[3]
        self.out_bias = variables[4]
