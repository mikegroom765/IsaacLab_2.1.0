from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class GiveUpAction(ActionTerm):
    r"""Give up action term.

    This action term is used to model the action of giving up on a task. This action term is useful in scenarios
    where the agent is unable to complete the task and decides to give up. The action term is a binary action
    which is either 1 (give up) or 0 (continue).

    The action term can be used to model the agent's decision making process, where the agent decides to give up
    on the task if it is unable to make progress. This can be useful in reinforcement learning scenarios where
    the agent is trained to make decisions based on the task's progress.
    """
    
    

    cfg: actions_cfg.GiveUpActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.GiveUpActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        # perform ManagerTermBase initialization first 
        super(ActionTerm, self).__init__(cfg, env)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)

    def process_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        # process the raw actions
        self._raw_actions.copy_(raw_actions)
        self._processed_actions.fill_(0)

        # check if the action is to give up
        give_up = raw_actions > 0.5
        self._processed_actions[give_up] = 1

        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # reset the processed actions
        self._raw_actions[env_ids] = 0.0
        
        
    @property
    def action_dim(self) -> int:
        return 1
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def apply_actions(self):
        # apply the processed actions - this is a no-op for the give up action term
        pass