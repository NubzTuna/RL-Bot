import numpy as np

class AerialObsBuilder:
    def reset(self, initial_state):
        pass
    
    def build_obs(self, player, state, previous_action):
        obs = []
        
        # Ball relative to player
        ball_rel = state.ball.position - player.car_data.position
        obs.extend(ball_rel)
        obs.extend(state.ball.linear_velocity)
        obs.extend(state.ball.angular_velocity)
        
        # Player car
        obs.extend(player.car_data.position)
        obs.extend(player.car_data.linear_velocity)
        obs.extend(player.car_data.quaternion)
        obs.extend(player.car_data.angular_velocity)
        
        # Player state
        obs.append(player.boost_amount / 100)
        obs.append(float(player.on_ground))
        obs.append(float(player.has_flip))
        
        return np.array(obs, dtype=np.float32)