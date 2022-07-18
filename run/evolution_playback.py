""" Playback learned policies and save traces of the learned behaviours
Usage:
    python run/evolution_playback.py [SELF_RECOGNISING_AGENT] [SEPARATE_AGENTS]
Where
    [SELF_RECOGNISING_AGENT] is the path to a .pkl file containing an agent trained to self-recognise
    [SEPARATE_AGENTS] is a path to a .pkl file containing separate agents for each environment
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import torch

from self_recog_as_optim import network, simulation


def get_actions(hidden_states: Tuple[torch.Tensor, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Get the actions for the current simulation state and hidden states using the globally defined mirror policy, non-mirror policy and sim
    Args:
        hidden_states (Tuple[torch.Tensor, torch.Tensor]): Current hidden states of networks
    Returns:
        Tuple[torch.Tensor, torch.Tensor] consisting of actions and hidden states for each policy
    """
    global mirror_policy, non_mirror_policy, sim
    policies = [mirror_policy, non_mirror_policy]
    vision_render = sim.vision
    return [pol.step(vision_render, hidden_state) for pol, hidden_state in zip(policies, hidden_states)]


def evaluate(parameters: torch.Tensor) -> torch.Tensor:
    """Evaluate the given parameters
    Args:
        parameters (torch.Tensor): The parameters to evaluate
    Returns:
        fitnesses (torch.Tensor): The fitness values
    """
    global separate_policies
    # If desired, separate parameters by policy
    if separate_policies:
        mirror_param = parameters[:, 0 : mirror_policy.num_parameters]
        non_mirror_param = parameters[:, mirror_policy.num_parameters : mirror_policy.num_parameters * 2]
    else:
        mirror_param = parameters
        non_mirror_param = parameters

    # If using self-evaluation we just have mirror and non-mirror in same ordering
    non_mirror_indices = torch.randperm(parameters.shape[0], device=device)

    # Load parameters to policies
    mirror_policy.load_parameters(mirror_param)
    non_mirror_policy.load_parameters(non_mirror_param[non_mirror_indices])

    # Initial hidden states are None
    hidden_states = [None, None]

    # Reset the simulation
    sim.reset()

    positions = []

    # Run to termination
    while not sim.terminal:
        # Get the actions
        (mirror_actions, mirror_hidden), (
            non_mirror_actions,
            non_mirror_hidden,
        ) = get_actions(hidden_states)
        # Step the sim
        sim.step(mirror_actions, non_mirror_actions)

        # Update hidden state
        hidden_states = [mirror_hidden, non_mirror_hidden]

        positions.append(torch.stack([sim.mirror_sim.positions, sim.non_mirror_sim.positions], dim=-1))

    return torch.cat(positions, dim=0)


if __name__ == "__main__":

    with torch.no_grad():

        playback_self = ""

        network_dim = 64
        popsize = 1

        playback_self = sys.argv[1]
        playback_sep = sys.argv[2]
        all_positions = []

        for separate_policies in [False, True]:

            if separate_policies:
                file = torch.load(playback_sep)
            else:
                file = torch.load(playback_self)

            # Device -- always CPU for playback
            device = "cpu"

            file = file.to(device)
            print(f"Experiment running on device {device}")

            vision_resolution = 32

            # Initialise the simulation
            sim = simulation.SelfRecSim(
                num_parallel=1,
                vision_resolution=vision_resolution,
                render=True,
                device=device,
            )

            mirror_policy = network.RNN(network_dim, vision_resolution)
            non_mirror_policy = network.RNN(network_dim, vision_resolution)
            # Initialise the policy
            if separate_policies:
                policy_dim = mirror_policy.num_parameters * 2
            else:
                policy_dim = mirror_policy.num_parameters

            # Evaluate the parameters saved in the file
            positions = evaluate(file)

            # Append the positional trace of the file
            all_positions.append(positions)

        # Generate plots for the 2 runs
        if not os.path.exists("plots"):
            os.mkdir("plots")

        import matplotlib.pyplot as plt

        iter = np.arange(200)

        plt.figure(figsize=(8, 1.82), dpi=200)
        plt.set_cmap("cividis")
        plt.plot(iter, all_positions[0][:, 0].cpu().numpy(), label="Self-recognising")
        plt.plot(iter, all_positions[1][:, 0].cpu().numpy(), label="Independent Policy")
        plt.legend()
        plt.ylabel("$x_1(t)$")
        plt.xlabel("$t$")
        plt.savefig("plots/env_1.png")

        plt.figure(figsize=(8, 1.82), dpi=200)
        plt.set_cmap("cividis")
        plt.plot(iter, all_positions[0][:, 1].cpu().numpy(), label="Self-recognising")
        plt.plot(iter, all_positions[1][:, 1].cpu().numpy(), label="Independent Policy")
        plt.legend()
        plt.ylabel("$x_2(t)$")
        plt.xlabel("$t$")
        plt.savefig("plots/env_2.png")
