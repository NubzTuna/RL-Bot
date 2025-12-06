"""Aerial Bot that runs the trained RLGym-PPO policy with Vutrium."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from pathlib import Path
from rlbot.agents.base_agent import SimpleControllerState
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs

# Optional imports: these may not exist in the lightweight Vutrium runtime, so we
# only use them inside try/except blocks.
try:  # pragma: no cover - best effort when rlgym is installed
    from rlgym.api.gamestate import GameState
    from rlgym.rocket_league.gamestates import PlayerData
except Exception:  # pragma: no cover - silently fall back to manual obs
    GameState = None
    PlayerData = None


@dataclass
class _CarView:
    """Minimal snapshot of a car used for observation building."""

    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    rotation: np.ndarray
    boost: float
    team: int


class _ObservationAdapter:
    """Converts RLBot packets into the 92-dim vector used during training."""

    def __init__(self, agent_index: int, expected_obs_size: int):
        self.agent_index = agent_index
        self.expected_obs_size = expected_obs_size

        # Match the training setup exactly when the full rlgym stack is
        # available. If not, we fall back to a robust manual encoder that still
        # produces a correctly-sized tensor for the policy network.
        self.default_obs: Optional[DefaultObs] = None
        if GameState is not None and PlayerData is not None:
            self.default_obs = DefaultObs(
                zero_padding=None,
                pos_coef=np.asarray(
                    [
                        1 / common_values.SIDE_WALL_X,
                        1 / common_values.BACK_NET_Y,
                        1 / common_values.CEILING_Z,
                    ]
                ),
                ang_coef=1 / np.pi,
                lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            )

    def build(self, packet) -> np.ndarray:
        """Return a normalized observation vector sized for the policy."""

        if self.default_obs is not None:
            try:
                return self._build_with_default_obs(packet)
            except Exception as exc:  # pragma: no cover - runtime safety net
                print(f"Falling back to manual obs pipeline (DefaultObs failed: {exc})")

        manual_obs = self._build_manual_obs(packet)
        return self._align_size(manual_obs)

    # ---- DefaultObs path -------------------------------------------------
    def _build_with_default_obs(self, packet) -> np.ndarray:
        # Convert RLBot packet into the minimal GameState objects that DefaultObs
        # expects. We only populate the fields that DefaultObs reads: positions,
        # velocities, rotations, boost, and teams.
        cars = packet.game_cars
        ball = packet.game_ball

        players: List[PlayerData] = []
        for car in cars:
            players.append(
                PlayerData(
                    car_id=car.car_id,
                    team_num=car.team,
                    input_vector=None,
                    physics=car.physics,
                    match_goals=None,
                    match_saves=None,
                    match_shots=None,
                    match_demolishes=None,
                    boost_amount=car.boost,
                    has_jump=car.has_jump,
                    has_double_jump=car.has_double_jump,
                    has_flip=car.has_flip,
                    is_demoed=car.is_demoed,
                )
            )

        game_state = GameState(cars=players, ball=ball.physics)
        # DefaultObs expects previous action but does not use it, so pass None.
        obs = self.default_obs.build_obs(players[self.agent_index], game_state, None)
        return self._align_size(np.asarray(obs, dtype=np.float32))

    # ---- Manual fallback path -------------------------------------------
    def _build_manual_obs(self, packet) -> np.ndarray:
        ball = packet.game_ball
        cars = packet.game_cars

        car = cars[self.agent_index]
        ball_pos = np.array(
            [ball.physics.location.x, ball.physics.location.y, ball.physics.location.z],
            dtype=np.float32,
        )
        ball_vel = np.array(
            [ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z],
            dtype=np.float32,
        )
        ball_ang = np.array(
            [ball.physics.angular_velocity.x, ball.physics.angular_velocity.y, ball.physics.angular_velocity.z],
            dtype=np.float32,
        )

        def car_view(c) -> _CarView:
            return _CarView(
                position=np.array(
                    [c.physics.location.x, c.physics.location.y, c.physics.location.z],
                    dtype=np.float32,
                ),
                linear_velocity=np.array(
                    [c.physics.velocity.x, c.physics.velocity.y, c.physics.velocity.z],
                    dtype=np.float32,
                ),
                angular_velocity=np.array(
                    [
                        c.physics.angular_velocity.x,
                        c.physics.angular_velocity.y,
                        c.physics.angular_velocity.z,
                    ],
                    dtype=np.float32,
                ),
                rotation=np.array(
                    [c.physics.rotation.pitch, c.physics.rotation.yaw, c.physics.rotation.roll],
                    dtype=np.float32,
                ),
                boost=float(c.boost) / 100.0,
                team=c.team,
            )

        car_views: List[_CarView] = [car_view(c) for c in cars]
        me = car_views[self.agent_index]

        features: List[float] = []

        # Ball (absolute, normalized)
        features.extend(ball_pos / np.array(
            [common_values.SIDE_WALL_X, common_values.BACK_NET_Y, common_values.CEILING_Z]
        ))
        features.extend(ball_vel / common_values.BALL_MAX_SPEED)
        features.extend(ball_ang / common_values.CAR_MAX_ANG_VEL)

        # Self (absolute, normalized)
        features.extend(me.position / np.array(
            [common_values.SIDE_WALL_X, common_values.BACK_NET_Y, common_values.CEILING_Z]
        ))
        features.extend(me.linear_velocity / common_values.CAR_MAX_SPEED)
        features.extend(me.angular_velocity / common_values.CAR_MAX_ANG_VEL)
        features.extend(me.rotation / np.pi)
        features.append(me.boost)

        # Relative car info (teammates then opponents) to mirror DefaultObs style
        teammates = [c for idx, c in enumerate(car_views) if idx != self.agent_index and c.team == car.team]
        opponents = [c for c in car_views if c.team != car.team]

        for other in teammates + opponents:
            rel_pos = (other.position - me.position) / np.array(
                [common_values.SIDE_WALL_X, common_values.BACK_NET_Y, common_values.CEILING_Z]
            )
            rel_vel = (other.linear_velocity - me.linear_velocity) / common_values.CAR_MAX_SPEED
            features.extend(rel_pos)
            features.extend(rel_vel)
            features.extend(other.angular_velocity / common_values.CAR_MAX_ANG_VEL)
            features.extend(other.rotation / np.pi)
            features.append(other.boost)

        return np.asarray(features, dtype=np.float32)

    # ---- Utility ---------------------------------------------------------
    def _align_size(self, obs: np.ndarray) -> np.ndarray:
        """Pad or trim observations to the size expected by the network."""

        if obs.size == self.expected_obs_size:
            return obs
        if obs.size > self.expected_obs_size:
            return obs[: self.expected_obs_size]

        padded = np.zeros(self.expected_obs_size, dtype=np.float32)
        padded[: obs.size] = obs
        return padded


class AerialBot:
    def __init__(self, name, team, index, checkpoint_path):
        self.name = name
        self.team = team
        self.index = index

        self.policy = None
        self.action_parser = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.obs_adapter: Optional[_ObservationAdapter] = None

    def _find_latest_checkpoint(self, models_dir):
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        checkpoints = list(models_path.rglob("*.pt")) + list(models_path.rglob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {models_dir}")

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def initialize_agent(self, field_info=None):
        checkpoint_path = Path(self.checkpoint_path).expanduser().resolve(strict=False)

        if checkpoint_path.is_dir():
            checkpoint_path = Path(self._find_latest_checkpoint(checkpoint_path))
        elif not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found. Provide AERIAL_BOT_CHECKPOINT as a file or "
                "directory containing .pt/.pth checkpoints."
            )

        print(f"Loading model from: {checkpoint_path}")
        self.checkpoint_path = str(checkpoint_path)

        import torch.nn as nn

        # Build a network that can adapt to different saved checkpoint shapes by
        # inspecting the state dict.
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        first_layer = next(v for k, v in state_dict.items() if "0.weight" in k)
        obs_size = first_layer.shape[1]
        last_weight = next(v for k, v in state_dict.items() if "-1.weight" in k or k.endswith("4.weight") or k.endswith("6.weight"))
        action_size = last_weight.shape[0]

        self.policy = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()
        print(f"✅ Model ready on {self.device} (obs {obs_size} → actions {action_size})")

        self.action_parser = LookupTableAction()
        self.obs_adapter = _ObservationAdapter(agent_index=self.index, expected_obs_size=obs_size)

    def get_output(self, packet):
        if self.policy is None or self.obs_adapter is None:
            return SimpleControllerState()

        try:
            obs = self.obs_adapter.build(packet)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_logits = self.policy(obs_tensor)
                action_idx = torch.argmax(action_logits, dim=1).item()

            return self._action_to_controls(action_idx)
        except Exception as e:  # pragma: no cover - runtime safety net
            print(f"Error in get_output: {e}")
            return SimpleControllerState()

    def _action_to_controls(self, action_idx):
        action_array = self.action_parser._lookup_table[action_idx]

        controls = SimpleControllerState()
        controls.throttle = float(action_array[0])
        controls.steer = float(action_array[1])
        controls.pitch = float(action_array[2])
        controls.yaw = float(action_array[3])
        controls.roll = float(action_array[4])
        controls.jump = bool(action_array[5] > 0.5)
        controls.boost = bool(action_array[6] > 0.5)
        controls.handbrake = bool(action_array[7] > 0.5)

        return controls
