"""Reward functions for aerial-focused Rocket League training."""
import numpy as np
from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BALL_RADIUS, CAR_MAX_SPEED, CEILING_Z, BALL_MAX_SPEED,
    SIDE_WALL_X, BACK_NET_Y, GOAL_HEIGHT, BLUE_GOAL_BACK, ORANGE_GOAL_BACK
)


class AerialFocusedReward(RewardFunction):
    """Simple, stable reward tuned for aerial training and SSL aspirations.

    The signal leans heavily on mechanics that transfer to real play:
    - Stay airborne (scaled by height).
    - Get above the ball when it is high to encourage vertical play.
    - Move toward the ball while airborne.
    - Touch the ball in the air (forwards/backward compatibility with different
      game state implementations by using ``ball_touched`` when available).
    - Lightly penalize sitting on the ground when the ball is elevated.
    """

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        # No per-agent state required; included for API completeness.
        return None

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}

        ball_pos = state.ball.position
        ball_height = float(ball_pos[2])
        ball_vel = state.ball.linear_velocity

        for agent in agents:
            car = state.cars[agent]

            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity
            car_height = float(car_pos[2])

            reward = 0.0

            # Encourage sustained aerial time with diminishing returns.
            if not car.on_ground:
                height_ratio = min(car_height / CEILING_Z, 1.0)
                reward += 0.6 * height_ratio

                # Reward chasing the ball while airborne.
                to_ball = ball_pos - car_pos
                dist_to_ball = float(np.linalg.norm(to_ball)) + 1e-6
                car_speed = float(np.linalg.norm(car_vel)) + 1e-6

                alignment = np.dot(to_ball / dist_to_ball, car_vel / car_speed)
                reward += max(alignment, 0.0) * 0.15

                # Extra reward for being above the ball when it is high.
                if ball_height > 500:
                    vertical_gap = max(car_height - ball_height, 0.0)
                    reward += 0.25 * min(vertical_gap / CEILING_Z, 1.0)

            else:
                # Discourage chilling on the ground when the ball is air dribbable.
                if ball_height > 800:
                    reward -= 0.1

            # Reward aerial ball touches across differing gym implementations.
            touched = False
            if hasattr(car, "ball_touched"):
                touched = bool(car.ball_touched)
            elif hasattr(state.ball, "latest_touch"):
                latest_touch = getattr(state.ball, "latest_touch")
                touched = getattr(latest_touch, "player_index", None) == agent

            if touched and not car.on_ground:
                reward += 2.0

            # Gentle incentive to keep the ball fast and high when airborne.
            if not car.on_ground and ball_height > 600:
                ball_speed = float(np.linalg.norm(ball_vel))
                reward += min(ball_speed / BALL_MAX_SPEED, 1.0) * 0.05

            rewards[agent] = reward

        return rewards


class UltimateAerialReward(RewardFunction):
    """
    MASSIVE reward combining:
    - Flip resets
    - Air dribbles  
    - Ascending/descending tracking
    - Air roll rewards
    - Offensive alignment
    - Ball height scaling
    """
    
    def __init__(self):
        super().__init__()
        # Flip reset tracking
        self.prev_has_flip = {}
        self.got_flip_reset = {}
        self.flip_reset_timestamp = {}
        
        # Air dribble tracking
        self.last_player_z = {}
        self.last_ball_z = {}
        self.last_ball_pos = {}
        
        # Stats
        self.total_flip_resets = 0
        self.total_air_dribble_touches = 0
        self.highest_aerial = 0
        self.steps_since_stats = 0
        
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_has_flip.clear()
        self.got_flip_reset.clear()
        self.flip_reset_timestamp.clear()
        self.last_player_z.clear()
        self.last_ball_z.clear()
        self.last_ball_pos.clear()
        
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        
        for agent in agents:
            car = state.cars[agent]
            
            # Initialize tracking
            if agent not in self.prev_has_flip:
                self.prev_has_flip[agent] = car.has_flip
                self.got_flip_reset[agent] = False
                self.flip_reset_timestamp[agent] = 0
                self.last_player_z[agent] = car.physics.position[2]
                self.last_ball_z[agent] = state.ball.position[2]
                self.last_ball_pos[agent] = state.ball.position.copy()
                rewards[agent] = 0.0
                continue
            
            reward = 0.0
            
            # Get physics
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity
            car_forward = car.physics.forward
            car_up = car.physics.up
            car_angvel = car.physics.angular_velocity
            ball_pos = state.ball.position
            ball_vel = state.ball.linear_velocity
            
            # Calculate key values
            to_ball = ball_pos - car_pos
            dist_to_ball = np.linalg.norm(to_ball)
            dir_to_ball = to_ball / (dist_to_ball + 1e-6)
            
            car_height = car_pos[2]
            ball_height = ball_pos[2]
            car_speed = np.linalg.norm(car_vel)
            ball_speed = np.linalg.norm(ball_vel)
            
            # Facing ball (cosine similarity)
            facing_ball = np.dot(car_forward, dir_to_ball)
            
            # Track highest aerial
            if car_height > self.highest_aerial and not car.on_ground:
                self.highest_aerial = car_height
                if car_height > 600:
                    print(f"🚀 New altitude record: {car_height:.0f}u!")
            
            # ==========================================
            # DETECT BALL TOUCH (distance-based)
            # ==========================================
            just_touched_ball = False
            prev_dist = np.linalg.norm(self.last_ball_pos[agent] - car_pos)
            if prev_dist < 150 and dist_to_ball > prev_dist + 50:
                # Ball moved away from car = likely touch
                just_touched_ball = True
            
            # ==========================================
            # AIR ROLL REWARD (bias term from C++ code)
            # ==========================================
            air_roll_reward = (car_angvel[1] / 5.5) * 5.0
            
            # ==========================================
            # OFFENSIVE ALIGNMENT (ball toward opponent goal)
            # ==========================================
            if car.is_orange:
                attack_goal = BLUE_GOAL_BACK
                defend_goal = ORANGE_GOAL_BACK
            else:
                attack_goal = ORANGE_GOAL_BACK  
                defend_goal = BLUE_GOAL_BACK
            
            to_goal = attack_goal - car_pos
            ball_to_goal = attack_goal - ball_pos
            
            # Cosine similarity: how aligned is ball movement toward goal
            if np.linalg.norm(to_goal) > 0 and np.linalg.norm(to_ball) > 0:
                offensive_alignment = np.dot(to_ball, to_goal) / (np.linalg.norm(to_ball) * np.linalg.norm(to_goal) + 1e-6)
            else:
                offensive_alignment = 0.0
            
            # ==========================================
            # AIR DRIBBLE DETECTION (from C++ code)
            # ==========================================
            # Ball position geometry check
            ball_y = max(ball_pos[1], 0.0)
            new_y = 6000.0 - ball_y
            ball_x = abs(ball_pos[0]) + BALL_RADIUS
            largest_x = new_y * 0.683
            
            in_air_dribble_zone = (
                not car.on_ground and
                ball_height > 250.0 and
                car_height < ball_height and
                ball_y > 1000.0 and
                ball_x < largest_x
            )
            
            if in_air_dribble_zone:
                # Check if ascending
                ascending = (car_height > self.last_player_z[agent] and 
                           ball_height > self.last_ball_z[agent])
                
                air_dribble_reward = 0.0
                
                # CASE 1: Ascending air dribble
                if ascending:
                    if dist_to_ball < 400 and car.boost_amount > 0.30 and facing_ball > 0.74:
                        # Perfect conditions
                        if car_pos[1] < ball_pos[1]:  # Behind ball
                            if just_touched_ball:
                                air_dribble_reward += 40.0
                                self.total_air_dribble_touches += 1
                                print(f"💎 Air dribble touch #{self.total_air_dribble_touches}!")
                            air_dribble_reward += 5.0
                            air_dribble_reward += air_roll_reward * 2.5
                        air_dribble_reward += facing_ball
                        air_dribble_reward += (1.0 - (dist_to_ball / 400.0)) * 7.5
                    
                    elif dist_to_ball < 400 and facing_ball > 0.74:
                        # Good conditions (lower boost)
                        if car_pos[1] < ball_pos[1]:
                            if just_touched_ball:
                                air_dribble_reward += 20.0
                            air_dribble_reward += 2.5
                            if car.boost_amount > 0.0:
                                air_dribble_reward += air_roll_reward * 1.25
                        air_dribble_reward += facing_ball / 2.0
                        air_dribble_reward += (1.0 - (dist_to_ball / 400.0)) * 3.75
                
                # CASE 2: Descending/level air dribble  
                else:
                    if dist_to_ball < 400 and car.boost_amount > 0.15 and facing_ball > 0.74:
                        if car_pos[1] < ball_pos[1]:
                            if just_touched_ball:
                                air_dribble_reward += 40.0
                            air_dribble_reward += 5.0
                            air_dribble_reward += air_roll_reward * 2.5
                        air_dribble_reward += facing_ball
                        air_dribble_reward += (1.0 - (dist_to_ball / 400.0)) * 7.5
                    
                    elif dist_to_ball < 400 and facing_ball > 0.74:
                        if car_pos[1] < ball_pos[1]:
                            if just_touched_ball:
                                air_dribble_reward += 20.0
                            air_dribble_reward += 2.5
                            if car.boost_amount > 0.0:
                                air_dribble_reward += air_roll_reward * 1.25
                        air_dribble_reward += facing_ball / 2.0
                        air_dribble_reward += (1.0 - (dist_to_ball / 400.0)) * 3.75
                    
                    # Descending is less valuable
                    air_dribble_reward /= 2.0
                
                # Apply offensive alignment multiplier
                air_dribble_reward *= max(offensive_alignment, 0.1)
                reward += air_dribble_reward
            
            # ==========================================
            # FLIP RESET DETECTION
            # ==========================================
            regained_flip = car.has_flip and not self.prev_has_flip[agent]
            
            if regained_flip and not car.on_ground:
                # Check if it's a flip reset (aerial flip regain)
                if car_height > 150:
                    reward += 15.0
                    self.got_flip_reset[agent] = True
                    self.flip_reset_timestamp[agent] = shared_info.get('step', 0)
                    self.total_flip_resets += 1
                    
                    print(f"")
                    print(f"{'='*60}")
                    print(f"🔥 FLIP RESET #{self.total_flip_resets}! 🔥")
                    print(f"   Height: {car_height:.0f}u | In air dribble zone: {in_air_dribble_zone}")
                    print(f"{'='*60}")
            
            # Using flip after reset
            current_step = shared_info.get('step', 0)
            time_since_reset = current_step - self.flip_reset_timestamp[agent]
            used_flip = not car.has_flip and self.prev_has_flip[agent]
            
            if used_flip and self.got_flip_reset[agent] and time_since_reset < 120:
                reward += 8.0
                print(f"💥 FLIP USED AFTER RESET! 💥")
            
            # Reset flip reset flag after timeout
            if time_since_reset > 300:
                self.got_flip_reset[agent] = False
            
            # ==========================================
            # GENERAL AERIAL REWARDS
            # ==========================================
            if not car.on_ground and car_height > 150:
                # Basic aerial reward
                height_factor = min(car_height / CEILING_Z, 1.0)
                reward += 0.3 * height_factor
                
                # Moving toward ball
                if dist_to_ball > 100:
                    vel_norm = car_vel / (car_speed + 1e-6)
                    alignment = np.dot(vel_norm, dir_to_ball)
                    if alignment > 0:
                        speed_factor = min(car_speed / CAR_MAX_SPEED, 1.0)
                        reward += 0.5 * alignment * speed_factor
                
                # Inverted orientation near ball (good for flip resets)
                if dist_to_ball < 400:
                    up_z = car_up[2]
                    if up_z < -0.4:
                        reward += 0.6 * abs(up_z)
                        if up_z < -0.85:
                            print(f"🔄 Fully inverted! (up_z={up_z:.2f})")
            
            # ==========================================
            # BALL HEIGHT SCALING (from C++ code)
            # ==========================================
            scaled_ball_z = ball_height
            if scaled_ball_z > 2000:
                scaled_ball_z = 2000
            elif scaled_ball_z < (GOAL_HEIGHT - (BALL_RADIUS * 3.5)):
                scaled_ball_z = 0
            
            # Add height bonus for aerial play near ball
            if not car.on_ground and dist_to_ball < 400 and facing_ball > 0.7:
                reward += (scaled_ball_z / 50.0)
            
            # ==========================================
            # NORMALIZE AND STORE
            # ==========================================
            reward = reward / 100.0  # Match C++ normalization
            
            # Update tracking
            self.prev_has_flip[agent] = car.has_flip
            self.last_player_z[agent] = car_height
            self.last_ball_z[agent] = ball_height
            self.last_ball_pos[agent] = ball_pos.copy()
            
            rewards[agent] = reward
        
        # Stats output
        self.steps_since_stats += 1
        if self.steps_since_stats >= 10000:
            self.steps_since_stats = 0
            print(f"")
            print(f"📊 TRAINING STATS")
            print(f"   Flip Resets: {self.total_flip_resets}")
            print(f"   Air Dribble Touches: {self.total_air_dribble_touches}")
            print(f"   Highest Altitude: {self.highest_aerial:.0f}u")
            print(f"")
        
        return rewards


class SimpleFlipResetReward(RewardFunction):
    """Lighter version for basic flip reset learning"""
    
    def __init__(self):
        super().__init__()
        self.prev_has_flip = {}
        self.flip_reset_count = 0
        
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_has_flip.clear()
        
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                   is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        
        for agent in agents:
            car = state.cars[agent]
            reward = 0.0
            
            if agent not in self.prev_has_flip:
                self.prev_has_flip[agent] = car.has_flip
                rewards[agent] = 0.0
                continue
            
            # Basic aerial
            if not car.on_ground and car.physics.position[2] > 200:
                reward += 0.1
            
            # Flip reset
            if car.has_flip and not self.prev_has_flip[agent]:
                if not car.on_ground and car.physics.position[2] > 150:
                    reward += 5.0
                    self.flip_reset_count += 1
                    print(f"🔥 FLIP RESET #{self.flip_reset_count}! 🔥")
            
            self.prev_has_flip[agent] = car.has_flip
            rewards[agent] = reward
        
        return rewards
