"""Sub-module containing command generators for generating a scalar value by sampling uniformly between a given range."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import torch
from collections.abc import Sequence

from omni.isaac.lab.managers import CommandTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import ScalarValueCommandCfg

class ScalarValueCommand(CommandTerm):
    """Command generator for generating a scalar value.

    The command generator generates a scalar value. The value is a scalar tensor that is generated
    by sampling from a uniform distribution within a specified range.
    """

    cfg: ScalarValueCommandCfg
    """Configuration for the command generator."""

    value_range: Tuple[float, float]
    """Range for the scalar value to be uniformly sampled"""

    def __init__(self, cfg: ScalarValueCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.value_range = cfg.value_range
        self.value = torch.zeros(self.num_envs, 1).to(self.device)


    def __str__(self):
        msg = "ScalarValueCommand:\n"
        msg += f"\tValue range: {self.cfg.value_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Scalar value command.

        Returns:
            torch.Tensor: A scalar tensor. Shape is (num_envs).
        """
        return self.value

    def _resample_command(self, env_ids: Sequence[int]):
        self.value[env_ids] = torch.FloatTensor(len(env_ids), 1).uniform_(*self.value_range).to(self.device)

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

