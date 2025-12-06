"""
Aerial Bot - Loads your trained RLGym-PPO model and plays in Rocket League via Vutrium SDK
"""
import torch
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction
from pathlib import Path


class AerialBot:
    def __init__(self, name, team, index, checkpoint_path):
        self.name = name
        self.team = team
        self.index = index
        
        # Model stuff
        self.policy = None
        self.obs_builder = None
        self.action_parser = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store checkpoint path
        self.checkpoint_path = checkpoint_path
        
    def _find_latest_checkpoint(self, models_dir):
        """Find the most recent checkpoint"""
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Look for .pt or .pth files
        checkpoints = list(models_path.glob("*.pt")) + list(models_path.glob("*.pth"))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {models_dir}")
        
        # Get the one with highest step count in filename
        latest = max(checkpoints, key=lambda p: int(''.join(filter(str.isdigit, p.stem))) if any(c.isdigit() for c in p.stem) else 0)
        return str(latest)
    
    def initialize_agent(self, field_info=None):
        """Load the trained model"""
        print(f"Loading model from: {self.checkpoint_path}")
        
        # Load the rlgym-ppo policy checkpoint
        # It's a state_dict, so we need to create the model architecture first
        import torch.nn as nn
        
        # Create policy network (same as training: [256, 256, 256])
        # Input size is 92 (from DefaultObs), output is 90 (action space)
        self.policy = nn.Sequential(
            nn.Linear(92, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 90)
        )
        
        # Load the trained weights
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        
        self.policy.to(self.device)
        self.policy.eval()
        print("✅ Model loaded and ready!")
        
        # Create obs builder (same as training)
        self.obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                 1 / common_values.BACK_NET_Y, 
                                 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
        )
        
        # Create action parser (same as training)
        self.action_parser = LookupTableAction()
        
        print("✅ Model loaded successfully!")
    
    def get_output(self, packet):
        """Main function - called every game tick"""
        if self.policy is None:
            return SimpleControllerState()
        
        try:
            # Convert RLBot packet → RLGym observation
            obs = self._packet_to_obs(packet)
            
            # Run model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_logits = self.policy(obs_tensor)
                action_idx = torch.argmax(action_logits, dim=1).item()
            
            # Convert action index → controller inputs
            controls = self._action_to_controls(action_idx)
            return controls
            
        except Exception as e:
            print(f"Error in get_output: {e}")
            return SimpleControllerState()
    
    def _packet_to_obs(self, packet):
        """Convert RLBot GameTickPacket to RLGym observation array"""
        # This is a simplified version - you may need to adjust based on your obs_builder
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        
        # Car position
        car_pos = np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z])
        
        # Car velocity  
        car_vel = np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z])
        
        # Car rotation (quaternion or euler)
        car_rot = np.array([car.physics.rotation.pitch, car.physics.rotation.yaw, car.physics.rotation.roll])
        
        # Ball position
        ball_pos = np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z])
        
        # Ball velocity
        ball_vel = np.array([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z])
        
        # Normalize positions
        car_pos_norm = car_pos / [common_values.SIDE_WALL_X, common_values.BACK_NET_Y, common_values.CEILING_Z]
        ball_pos_norm = ball_pos / [common_values.SIDE_WALL_X, common_values.BACK_NET_Y, common_values.CEILING_Z]
        
        # Normalize velocities
        car_vel_norm = car_vel / common_values.CAR_MAX_SPEED
        ball_vel_norm = ball_vel / common_values.BALL_MAX_SPEED
        
        # Normalize rotations
        car_rot_norm = car_rot / np.pi
        
        # Concatenate into observation (this should match DefaultObs output)
        obs = np.concatenate([
            car_pos_norm,
            car_vel_norm,
            car_rot_norm,
            np.array([car.boost / 100.0]),  # Boost amount
            ball_pos_norm,
            ball_vel_norm
        ])
        
        return obs
    
    def _action_to_controls(self, action_idx):
        """Convert action index to controller inputs"""
        # Get the action array from the lookup table
        action_array = self.action_parser._lookup_table[action_idx]
        
        # action_array format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
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
