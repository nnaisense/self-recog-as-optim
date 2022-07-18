""" Run the evolutionary experiments
Usage:
    python run/evolution.py [EXPERIMENT_REPEATS] [SEPARATE_POLICIES]
Where
    [EXPERIMENT_REPEATS] is a positive integer indicating the number of evolutionary experiments to run
    [SEPARATE_POLICIES] is a boolean (e.g. 'True' or 'False') indicating whether to use separate policies (the comparison benchmark)
                        or a single policy (self-recognition)
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

    # Assign a random pairing of mirror and non-mirror policies
    non_mirror_indices = torch.randperm(parameters.shape[0], device=device)

    # We'll also need a mapping to go backwards w.r.t. rewards
    reverse_indices = torch.zeros_like(non_mirror_indices)
    reverse_indices[non_mirror_indices] = torch.arange(parameters.shape[0], device=device)

    # Load parameters to policies
    mirror_policy.load_parameters(mirror_param)
    non_mirror_policy.load_parameters(non_mirror_param[non_mirror_indices])

    # Initial hidden states are None
    hidden_states = [None, None]

    # Reset the simulation
    sim.reset()

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

    # Get the rewards
    global mirror_reward, non_mirror_reward
    mirror_reward = sim.mirror_reward
    non_mirror_reward = sim.non_mirror_reward[reverse_indices]
    all_reward = mirror_reward + non_mirror_reward

    all_reward[mirror_reward <= -200.0] = -400.0
    global mirror_success, non_mirror_success
    mirror_success = torch.abs(sim.mirror_sim.positions - np.pi / 6) < 0.1
    non_mirror_success = torch.abs(sim.non_mirror_sim.positions + 0.5) < 0.1
    return all_reward


if __name__ == "__main__":

    with torch.no_grad():

        ### EXPERIMENT CONFIGURATION ###

        num_repeats = int(sys.argv[1])
        # Hidden dimension of RNN
        network_dim = 64
        # Total population size
        popsize = 15000
        # Number of generations to run evolution for
        generations = 4000

        # Whether to evolve separate policies for the mirror and non-mirror environments
        separate_policies = sys.argv[2].lower() == "true"

        print(
            f'Interpreted command line input "{sys.argv[2]}" as {separate_policies} for whether to use separate policies'
        )

        # Make the experiment_logs directory
        if not os.path.exists("experiment_logs"):
            os.mkdir("experiment_logs")

        # Define the experiment name according to the network dimension, popsize, and whether to evolve separate policies for the mirror and non-mirror environments
        experiment_name = f'{network_dim}_{popsize}_{"sep" if separate_policies else "com"}'
        if not os.path.exists(f"experiment_logs/{experiment_name}"):
            os.mkdir(f"experiment_logs/{experiment_name}")

        print("Base experiment name", experiment_name)

        # Device -- defaults to using CUDA
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Experiment running on device {device}")

        ### SET UP ENVIRONMENT ###

        # Discrete resolution of vision
        vision_resolution = 32

        # Initialise the simulation
        sim = simulation.SelfRecSim(
            num_parallel=popsize,
            vision_resolution=vision_resolution,
            render=False,
            device=device,
        )

        # Create RNNs for both the mirror and non-mirror policies
        mirror_policy = network.RNN(network_dim, vision_resolution)
        non_mirror_policy = network.RNN(network_dim, vision_resolution)

        # Initialise the policy -- note that we need 2x as many parameters when evolving an RNN for each scenario
        if separate_policies:
            policy_dim = mirror_policy.num_parameters * 2
        else:
            policy_dim = mirror_policy.num_parameters

        # ClipUp parameters
        # Initial search radius
        r = 4.5
        if separate_policies:
            # Rescale search radius if evolving 2x as many parameters
            r = r * np.sqrt(2)
        # Initial stdev scales with initial radius + problem dimension
        sigma_init = r / np.sqrt(policy_dim)
        # Max velocity and alpha are inferred from initial radius according to general ClipUp rules-of-thumb
        v_max = r / 15
        alpha = v_max * 0.5
        momentum = 0.9

        # Using centered ranking: Note that you can also use 'none' e.g. just centered fitnesses, or 'nes' e.g. the fitness shaping used in NES literature
        ranking = "centered"

        # Pre compute utilities for the popsize, rather than computing in each iteration
        # Simulate ranks 1 ... popsize
        pre_ranks = torch.arange(popsize, device=device).to(torch.float) + 1
        # Compute the numerator of NES utilities max(log(popsize / 2 + 1) - log(r_i), 0)
        numerator = np.log(popsize / 2 + 1) - torch.log(pre_ranks)
        numerator[numerator <= 0.0] = 0.0
        # Compute the denominator of NES utilities sum of j = 1 .. n [ utilities max(log(lambda / 2 + 1) - log(r_j), 0) ]
        denominator = torch.sum(numerator)
        # NES utility is numerator / denominator -  1 / popsize
        pre_utilities_nes = popsize * (numerator / denominator - (1.0 / popsize))
        # Centered utility is simply (popsize - r_i) / (popsize - 1) - 0.5
        pre_utilities_centered = (popsize - pre_ranks) / (popsize - 1) - 0.5

        # Run the evolution the specified number of times
        for repeat in range(num_repeats):

            # Trace trajectories of population fitness
            fitness_trace = []
            # Trace trajectories of center of populations' fitness
            center_fitness_trace = []

            # Create a sub-directory for this repeat
            experiment_path = f"experiment_logs/{experiment_name}/{repeat}"
            if not os.path.exists(experiment_path):
                os.mkdir(experiment_path)

            print(f"Experiment repeat {repeat} logging to {experiment_path}")

            # Initialize the center mu at zeros
            center = torch.zeros((1, policy_dim), device=device)

            # Create a buffer for standard normal noise
            noise = torch.zeros((popsize, policy_dim), device=device)

            # Initialize the ClipUp velocity at zero
            velocity = torch.zeros_like(center)

            for generation in range(1, generations + 1):

                # Sample noise
                half_noise = torch.randn(
                    (popsize // 2, policy_dim), device=device
                )  # < ---- comment out to disable symmetric sampling
                noise = torch.cat([half_noise, -half_noise], dim=0)
                # noise = torch.randn((popsize, policy_dim), device = device)   # < ---- uncomment out to disable symmetric sampling

                # Make population
                pop = center + sigma_init * noise
                # Evaluate the population
                fitnesses = evaluate(pop)

                # Keep track of the mean fitness of the population
                fitness_trace.append(np.mean(fitnesses.cpu().numpy()))

                # Rank the fitness values
                order = torch.argsort(fitnesses, descending=True)
                ranks = torch.zeros_like(order)
                ranks[order] = torch.arange(popsize, device=device)

                # Apply rank based fitness shaping
                if ranking == "none":
                    utilities = fitnesses - torch.mean(fitnesses)
                elif ranking == "nes":
                    utilities = pre_utilities_nes[ranks]
                elif ranking == "centered":
                    utilities = pre_utilities_centered[ranks]

                # Center gradient is simply sigma * (u_i * epsilon_i) where epsilon_i is the ith noise
                center_grad = sigma_init * (utilities.unsqueeze(-1) * noise).mean(dim=0)

                # ClipUp
                # Normalize center gradient
                norm_grad = center_grad / (torch.norm(center_grad))
                # Accumulate center gradient into velocity
                velocity = alpha * norm_grad + momentum * velocity
                # Clip velocity if it exceeds the maximum velocity
                if torch.norm(velocity) > v_max:
                    velocity = v_max * velocity / torch.norm(velocity)

                # Update
                old_center = center.clone()
                center[:] = center[:] + velocity

                # Do some logging and stdout
                print("Center movement", torch.norm(center - old_center))

                if generation % 10 == 0:
                    center_eval = center.clone().repeat(popsize, 1).to(device)
                    print("Generation", generation, "Mean fitness", torch.mean(fitnesses))
                    print(
                        "Train Mirror reward",
                        torch.mean(mirror_reward),
                        torch.max(mirror_reward),
                    )
                    print(
                        "Train Non-mirror reward",
                        torch.mean(non_mirror_reward),
                        torch.max(non_mirror_reward),
                    )
                    print(
                        "Success rate",
                        (torch.sum(mirror_success.to(torch.float)) + torch.sum(non_mirror_success.to(torch.float)))
                        / (popsize * 2),
                    )
                    center_fit = torch.mean(evaluate(center_eval))
                    print("Mean test fitness", center_fit)
                    center_fitness_trace.append(center_fit)
                    print(
                        "Test Mirror reward",
                        torch.mean(mirror_reward),
                        torch.max(mirror_reward),
                    )
                    print(
                        "Test Non-mirror reward",
                        torch.mean(non_mirror_reward),
                        torch.max(non_mirror_reward),
                    )

                    torch.save(center.cpu(), f"{experiment_path}/generation_{generation}.pkl")

            # Save the fitness trajectories
            data = {
                "train": fitness_trace,
                "test": center_fitness_trace,
            }
            np.save(f"{experiment_path}/data.np", data)

        # Finally, leave a rendered evaluation (of the last evolutionary run) running until the session is closed
        sim = simulation.SelfRecSim(1, vision_resolution=vision_resolution, render=True, device="cpu")

        while True:
            evaluate(center.cpu())
