from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league import common_values
import numpy as np

from rewards import UltimateAerialReward

def build_rocketsim_env():
    """
    ULTIMATE environment for flip resets + air dribbles
    """
    spawn_opponents = True
    team_size = 1
    tick_skip = 8
    timeout_seconds = 20  # Longer for complex aerial plays
    
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_cond = GoalCondition()
    truncation_cond = NoTouchTimeoutCondition(timeout_seconds)
    
    # ULTIMATE REWARD COMBO
    ultimate_aerial = (UltimateAerialReward(), 1.0)  # Main reward - flip resets + air dribbles
    goal_reward = (GoalReward(), 25.0)  # MASSIVE goal reward
    touch_reward = (TouchReward(), 0.05)  # Tiny touch reward (ultimate reward handles it)
    
    rewards_and_weights = (ultimate_aerial, goal_reward, touch_reward)
    reward_fn = CombinedReward(*rewards_and_weights)
    
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                             1 / common_values.BACK_NET_Y, 
                             1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )
    
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size if spawn_opponents else 0),
        KickoffMutator()
    )
    
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_cond,
        truncation_cond=truncation_cond,
        transition_engine=RocketSimEngine()
    )
    
    return RLGymV2GymWrapper(env)

if __name__ == "__main__":
    print("="*70)
    print("ULTIMATE AERIAL TRAINING - Flip Resets + Air Dribbles")
    print("="*70)
    print("Reward features:")
    print("  🔥 FLIP RESETS:")
    print("     - Aerial flip regain: +15.0")
    print("     - Using flip after reset: +8.0")
    print()
    print("  💎 AIR DRIBBLES:")
    print("     - Ascending + perfect positioning: +40.0 per touch")
    print("     - Air roll bonus: up to +12.5")
    print("     - Facing ball + proximity: +7.5")
    print("     - Offensive alignment multiplier")
    print()
    print("  ✈️  GENERAL AERIAL:")
    print("     - Height rewards (scaled)")
    print("     - Inverted positioning: +0.6")
    print("     - Speed toward ball: +0.5")
    print("     - Ball height scaling")
    print()
    print("  ⚽ GOALS: +25.0")
    print("="*70)
    
    learner = Learner(
        env_create_function=build_rocketsim_env,
        n_proc=2,  # Parallel environments
        min_inference_size=7,
        ppo_batch_size=100_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,
        ppo_epochs=2,
        policy_layer_sizes=[256, 256, 256],
        critic_layer_sizes=[256, 256, 256],
        policy_lr=5e-5,
        critic_lr=5e-5,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=100_000,
        timestep_limit=50_000_000,
        log_to_wandb=False,
        device="cuda",
        render=False
    )
    
    print("\n🚀 Starting ULTIMATE training...")
    print("Watch for: 🔥 (flip resets), 💎 (air dribbles), 🔄 (inverted), 🚀 (altitude records)")
    print("Press Ctrl+C to stop\n")
    
    learner.learn()
