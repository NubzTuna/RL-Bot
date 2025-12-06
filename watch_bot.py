from rlgym_ppo import Learner
from train import build_rocketsim_env

if __name__ == "__main__":
    # Create learner with rendering ON
    learner = Learner(
        env_create_function=build_rocketsim_env,
        device="cuda",
        render=True,  # This turns on the 3D visualization!
        n_proc=1,  # Just 1 environment to watch
        render_delay=0.05  # Slow it down a bit so you can see
    )

    # Load your latest checkpoint
    checkpoint_path = "data/checkpoints/rlgym-ppo-run-1764984075826261900/7850000"
    learner.load(checkpoint_path, load_wandb=False)  # Added load_wandb=False

    print("Loaded checkpoint! Watching bot play...")
    print("Press Ctrl+C to stop")

    # Run the bot (with rendering)
    learner.learn()