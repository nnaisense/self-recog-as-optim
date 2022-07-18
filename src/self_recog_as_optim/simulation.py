import time

import numpy as np
import pygame
import torch


class MountainCarSim:
    def __init__(
        self,
        num_parallel: int,
        device: torch.device = "cpu",
    ) -> None:
        """Modified mountain car environment from
        Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
        Args:
            num_parallel (int): The number of simulations to run in parallel
            device (torch.device): The device on which to run the simulation
        """
        self.num_parallel = num_parallel
        self.device = device
        # First reset instantiates variables
        self.reset()

    def reset(self) -> None:
        """Reset the simulations"""
        self.positions = -0.5 * torch.ones((self.num_parallel,), device=self.device)
        self.velocities = torch.zeros((self.num_parallel,), device=self.device)

    def integrate(self, actions: torch.Tensor) -> None:
        """Integrate actions into the simulations
        Args:
            actions (torch.Tensor): A tensor of shape [self.num_parallel] of actions to apply
        """
        # Safety -- ensure actions are within valid range
        actions = torch.clamp(actions, -1.0, 1.0)

        # Update velocity
        self.velocities = self.velocities + 0.001 * actions - 0.0025 * torch.cos(3 * self.positions)

        # Bound velocity
        self.velocities = torch.clamp(self.velocities, -0.07, 0.07)

        # Positional bounds also mean zero velocity
        self.velocities[torch.logical_and(self.positions <= -1.2, self.velocities < 0)] = 0.0
        self.velocities[torch.logical_and(self.positions >= 0.6, self.velocities > 0)] = 0.0
        # Bound position
        self.positions = torch.clamp(self.positions, -1.2, 0.6)

        # Update position
        self.positions = self.positions + self.velocities


class SelfRecSim:
    def __init__(
        self,
        num_parallel: int,
        vision_resolution: int = 32,
        render: bool = False,
        dt: float = 0.05,
        device: torch.device = "cpu",
    ) -> None:
        """Self recognition simulation described in 'Self-recognition as an Optimisation Problem
        Args:
            num_parallel (int): The number of simulations to run in parallel
            vision_resolution (int): The resolution to use for vision
            render (bool): Whether to render the simulation
            dt (float): Rate (hz) to render simulation -- DOES NOT AFFECT DYNAMICS
            device (torch.device): The device on which to run the simulation
        device: torch.device = 'cpu',
        """
        # Instantiate simulations
        self.mirror_sim = MountainCarSim(num_parallel, device=device)
        self.non_mirror_sim = MountainCarSim(num_parallel, device=device)

        # Create buffer variable for vision
        self.vision_buffer = torch.zeros((num_parallel, vision_resolution), device=device)
        # Helper indices for vision
        self.sim_indices = torch.arange(num_parallel, device=device)

        # Store system variables
        self.num_parallel = num_parallel
        self.vision_resolution = vision_resolution
        self.device = device

        # Call reset
        self.reset()

        # Set up any rendering
        self.dt = dt
        self.render = render
        if render:
            pygame.init()
            # pygame.display.list_modes()
            pygame.font.init()
            self._action_font = pygame.font.SysFont("Comic Sans MS", 30)
            self._screen = pygame.display.set_mode([1200, 500])

        # Create some variables for rendering the curve
        self._xs = []
        self._ys = []
        for x in range(600):
            xs = (1.8 * x / 600) - 1.2
            ys = self._height(xs)
            xs, ys = self._scale(xs, ys)
            self._xs.append(x)
            self._ys.append(ys)

    def reset(self) -> None:
        """Reset the simulations"""
        self.mirror_sim.reset()
        self.non_mirror_sim.reset()
        # Instantiate reward buffers
        self.mirror_reward = torch.zeros((self.num_parallel,), device=self.device)
        self.non_mirror_reward = torch.zeros((self.num_parallel,), device=self.device)

        # Track time
        self.iter = 0

    def integrate(self, mirror_actions: torch.Tensor, non_mirror_actions) -> None:
        """Integrate actions into the simulations
        Args:
            mirror_actions (torch.Tensor): A tensor of shape [self.num_parallel] of actions to apply to the mirror simulation
            non_mirror_actions (torch.Tensor): A tensor of shape [self.num_parallel] of actions to apply to the non-mirror simulation
        """
        self.mirror_sim.integrate(mirror_actions)
        self.non_mirror_sim.integrate(non_mirror_actions)

    def update_rewards(self) -> None:
        """Update the reward variables"""
        # Target positions
        mirror_target = np.pi / 6
        non_mirror_target = -0.5
        # Reward is negative distance to target positions
        mirr_reward = -torch.abs(self.mirror_sim.positions - (mirror_target)) / (
            (mirror_target - non_mirror_target) / 2
        )
        non_mirr_reward = -torch.abs(self.non_mirror_sim.positions - (non_mirror_target)) / (
            (mirror_target - non_mirror_target) / 2
        )

        self.mirror_reward += torch.clamp(mirr_reward, -1.0, 0.0)
        self.non_mirror_reward += torch.clamp(non_mirr_reward, -1.0, 0.0)

    def step(self, mirror_actions: torch.Tensor, non_mirror_actions) -> None:
        """Step the simulations
        Args:
            mirror_actions (torch.Tensor): A tensor of shape [self.num_parallel] of actions to apply to the mirror simulation
            non_mirror_actions (torch.Tensor): A tensor of shape [self.num_parallel] of actions to apply to the non-mirror simulation
        """
        self.integrate(mirror_actions, non_mirror_actions)
        self.update_rewards()
        # Step time
        self.iter += 1

        if self.render:
            self.render_sim()
            time.sleep(self.dt)

    @property
    def terminal(self) -> bool:
        """Whether the simulations have reached a terminal state"""
        return self.iter >= 200

    @property
    def vision(self) -> torch.Tensor:
        """Compute visual input for mirror simulation
        Returns:
            vision (torch.Tensor): A [self.num_parallel, self.vision_resolution] tensor of vision
        """
        # Small epsilon prevents overflow
        vision_low = -1.2 - 1e-5
        vision_high = 0.6 + 1e-5
        vision_step = (vision_high - vision_low) / self.vision_resolution

        # Rescale positions to normalize in vision step
        norm_position = (self.mirror_sim.positions - vision_low) / vision_step
        # Convert to indices
        norm_low_position = torch.floor(norm_position).to(torch.long)
        norm_high_position = torch.ceil(norm_position).to(torch.long)

        norm_low_position = torch.clamp(norm_low_position, 0, self.vision_resolution - 1)
        norm_high_position = torch.clamp(norm_high_position, 0, self.vision_resolution - 1)

        # Consider distance to low and high
        low_position = (norm_low_position * vision_step) + vision_low
        high_position = (norm_high_position * vision_step) + vision_low
        dist_to_low = torch.abs(low_position - self.mirror_sim.positions) / vision_step
        dist_to_high = torch.abs(high_position - self.mirror_sim.positions) / vision_step

        # Clear existing vision
        self.vision_buffer.zero_()

        # Store new info
        self.vision_buffer[self.sim_indices, norm_high_position] = 1.0 - dist_to_high
        self.vision_buffer[self.sim_indices, norm_low_position] = 1.0 - dist_to_low

        # Normalise
        mean = torch.mean(self.vision_buffer, dim=-1, keepdim=True)
        std = torch.std(self.vision_buffer, dim=-1, keepdim=True)

        self.vision_buffer = (self.vision_buffer - mean) / std

        return self.vision_buffer

    """ Some helper functions for rendering"""

    def _height(self, xs):
        return np.sin(3.0 * xs) * 0.45 + 0.55

    def _scale(self, x, y):
        scale = 600
        x = (1.2 + x) / (1.8)
        return (x * scale), 400 - (y * 300)

    def render_sim(self) -> None:
        """Render the environment."""
        if self.render:
            # Fill the background with blue (sky)
            self._screen.fill((100, 150, 200))

            # Render mirror

            for xs, ys in zip(self._xs, self._ys):
                pygame.draw.rect(
                    self._screen,
                    (155, 155, 155),
                    pygame.Rect(xs, (ys + 5), 1, 400 - ys),
                )

            car_x = float(self.mirror_sim.positions[0])
            car_y = self._height(car_x)

            car_x, car_y = self._scale(car_x, car_y)
            pygame.draw.rect(self._screen, (0, 0, 0), pygame.Rect(car_x - 10, car_y - 10, 20, 20))

            obj_x = np.pi / 6
            obj_y = self._height(obj_x)
            obj_x, obj_y = self._scale(obj_x, obj_y)
            pygame.draw.rect(self._screen, (0, 200, 50), pygame.Rect(obj_x - 5, obj_y - 75, 10, 80))

            # Draw a white  background for bottom text
            pygame.draw.rect(self._screen, (255, 255, 255), pygame.Rect(0, 400, 600, 100))
            # Render non-mirror
            x_offset = 600

            for xs, ys in zip(self._xs, self._ys):
                pygame.draw.rect(
                    self._screen,
                    (155, 155, 155),
                    pygame.Rect(xs + x_offset, (ys + 5), 1, 400 - ys),
                )

            car_x = float(self.non_mirror_sim.positions[0])
            car_y = self._height(car_x)

            car_x, car_y = self._scale(car_x, car_y)
            pygame.draw.rect(
                self._screen,
                (0, 0, 0),
                pygame.Rect(car_x - 10 + x_offset, car_y - 10, 20, 20),
            )

            obj_x = -0.5
            obj_y = self._height(obj_x)
            obj_x, obj_y = self._scale(obj_x, obj_y)
            pygame.draw.rect(
                self._screen,
                (0, 200, 50),
                pygame.Rect(obj_x - 5 + x_offset, obj_y - 75, 10, 80),
            )

            # Draw a white background for bottom text
            pygame.draw.rect(self._screen, (255, 255, 255), pygame.Rect(0 + x_offset, 400, 600, 100))

            # Draw vision array
            vision_center_x = 600
            vision_center_y = 425
            vision_width = 400
            vision_x_low = vision_center_x - (vision_width / 2)
            vision_height = vision_width / self.vision_resolution
            vision_start_y = vision_center_y - vision_height / 2.0
            vision = self.vision[0].cpu()
            for vis_index in range(self.vision_resolution):
                vis_value = float(vision[vis_index])
                vis_value = np.clip(vis_value, 0.0, 1.0)
                # Convert to color
                vis_col = (255 * vis_value, 255 * vis_value, 255 * vis_value)
                vision_start_x = vision_x_low + (vis_index * vision_height)

                # Draw
                pygame.draw.rect(
                    self._screen,
                    vis_col,
                    pygame.Rect(vision_start_x, vision_start_y, vision_height, vision_height),
                )

            # Add reward text
            mirror_rew = self._action_font.render(f"Reward: {float(self.mirror_reward[0].cpu())}", False, (0, 0, 0))
            non_mirror_rew = self._action_font.render(
                f"Reward: {float(self.non_mirror_reward[0].cpu())}", False, (0, 0, 0)
            )
            self._screen.blit(
                mirror_rew,
                (300 - mirror_rew.get_width() / 2, (475 - mirror_rew.get_height() / 2)),
            )
            self._screen.blit(
                non_mirror_rew,
                (
                    x_offset + 300 - non_mirror_rew.get_width() / 2,
                    (475 - non_mirror_rew.get_height() / 2),
                ),
            )
            # Update the camera
            pygame.display.update()


if __name__ == "__main__":

    sim = SelfRecSim(2, 32, device="cuda:0", render=True)

    while True:
        rand_acts_mirror = torch.randn((sim.num_parallel), device=sim.device)
        rand_acts_non_mirror = torch.randn((sim.num_parallel), device=sim.device)

        sim.step(rand_acts_mirror, rand_acts_non_mirror)

        if sim.terminal:
            sim.reset()
