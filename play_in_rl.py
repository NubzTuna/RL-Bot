from rlgym_tools.rocket_league_interface import RocketLeagueInterface
from rlgym_ppo import Learner
from train import build_rocketsim_env
import numpy as np

if __name__ == "__main__":
    # Load your trained model
    learner = Learner(
        env_create_function=build_rocketsim_env,
        device="cuda",
        n_proc=1
    )
    
    checkpoint_path = "data/checkpoints/rlgym-ppo-run-1764984075826261900/7850000"
    print("Loading checkpoint...")
    learner.load(checkpoint_path, load_wandb=False)
    print("Checkpoint loaded!")
    
    # Connect to Rocket League
    print("Connecting to Rocket League...")
    rl = RocketLeagueInterface(team_size=1)
    print("Connected! Start a match in Rocket League now...")
    
    while True:
        obs = rl.receive_observation()
        
        # Get action from your trained model
        # Convert obs to the right format if needed
        obs_array = np.array(obs).reshape(1, -1)
        action = learner.policy.get_action(obs_array, deterministic=True)
        
        rl.send_actions(action[0])