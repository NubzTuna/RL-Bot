"""Minimal Vutrium-ready bot with fresh observation and policy code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlgym.rocket_league.action_parsers import LookupTableAction

from vutrium_checkpoint import resolve_checkpoint


OBS_SIZE = 17


@dataclass
class _CompactObservation:
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_rot: np.ndarray
    boost: float
    team: int

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.ball_pos, self.ball_vel, self.car_pos, self.car_vel, self.car_rot, np.array([self.boost, self.team])]
        ).astype(np.float32)


class _ObservationBuilder:
    """Transforms GameTickPacket data into a compact numeric vector."""

    @staticmethod
    def build(packet, index: int, team: int) -> np.ndarray:
        ball = packet.game_ball
        car = packet.game_cars[index]

        ball_state = _vec(ball.physics.location), _vec(ball.physics.velocity)
        car_state = _vec(car.physics.location), _vec(car.physics.velocity), _euler(car.physics.rotation)

        obs = _CompactObservation(
            ball_pos=ball_state[0],
            ball_vel=ball_state[1],
            car_pos=car_state[0],
            car_vel=car_state[1],
            car_rot=car_state[2],
            boost=float(car.boost) / 100.0,
            team=float(team),
        )
        return obs.to_vector()


def _vec(vec3) -> np.ndarray:
    return np.array([vec3.x, vec3.y, vec3.z], dtype=np.float32) / 2300.0


def _euler(rot) -> np.ndarray:
    return np.array([rot.pitch, rot.yaw, rot.roll], dtype=np.float32)


class _Policy(torch.nn.Module):
    def __init__(self, obs_size: int, action_size: int = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class VutriumBot(BaseAgent):
    """Brand-new bot implementation tailored for the Vutrium SDK."""

    def __init__(self, name: str, team: int, index: int, checkpoint_override: Optional[str] = None):
        super().__init__(name, team, index)
        self.checkpoint_override = checkpoint_override
        self.obs_builder = _ObservationBuilder()
        self.action_parser = LookupTableAction()
        self.policy: Optional[_Policy] = None
        self.device = torch.device("cpu")
        self._action_count = len(self.action_parser.get_action_space())

    def initialize_agent(self, field_info):  # noqa: D401 - RLBot signature
        """Load the newest checkpoint and prepare the policy."""
        checkpoint = resolve_checkpoint(Path(__file__), self.checkpoint_override)
        self.policy = _Policy(OBS_SIZE, self._action_count)
        self._load_weights(checkpoint)

    def _load_weights(self, checkpoint: Path) -> None:
        try:
            data = torch.load(checkpoint, map_location=self.device)
            state_dict = data.get("state_dict") or data.get("policy_state_dict") or data
            self.policy.load_state_dict(state_dict, strict=False)
        except Exception as exc:  # pragma: no cover - robustness at runtime
            print(f"[VutriumBot] Failed to load checkpoint {checkpoint}: {exc}. Using random weights.")

    def get_output(self, packet) -> SimpleControllerState:  # noqa: D401 - RLBot signature
        """Infer controls for the local player from the latest packet."""
        obs_vec = self.obs_builder.build(packet, self.index, self.team)
        obs_tensor = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)

        logits = self.policy(obs_tensor) if self.policy else torch.zeros((1, self._action_count))
        action_idx = int(torch.argmax(logits, dim=1).item())

        controls = self.action_parser.parse_actions(np.array([action_idx]))[0]
        return SimpleControllerState(
            throttle=float(controls[0]),
            steer=float(controls[1]),
            pitch=float(controls[2]),
            yaw=float(controls[3]),
            roll=float(controls[4]),
            jump=bool(controls[5]),
            boost=bool(controls[6]),
            handbrake=bool(controls[7]),
        )
