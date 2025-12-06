# AERIAL BOT PROJECT - NOTES FOR CLAUDE

## PROJECT STATUS: READY TO TEST (13.85M steps trained)

---

## QUICK FACTS:
- **Student:** Aiden at Embry-Riddle (Daytona Beach)
- **Goal:** Train RL bot to do aerials using reinforcement learning
- **Current Progress:** 13.85M steps (good aerial attempts, needs 25M+ for strong play)
- **Method:** RLGym-PPO training in RocketSim, deploy via VutriumSDK DLL injection

---

## PYTHON ENVIRONMENTS:

### Old env (Python 3.10.11):
- Location: `rl_bot_env/`
- Used for training (still running)
- Has PyTorch nightly with RTX 5060 support (sm_120)
- **DO NOT TOUCH - training is happening here**

### New env (Python 3.11):
- Location: `rl_bot_env_311/`
- Created for VutriumSDK compatibility (.pyd compiled for cp311)
- Installing PyTorch stable + rlgym-ppo
- **USE THIS for running the bot in Rocket League**

---

## KEY FILES:

### Training:
- `train.py` - Main training script (CURRENTLY RUNNING, 13.8M+ steps)
- `rewards.py` - Aerial-focused reward function
- `data/checkpoints/rlgym-ppo-run-1764991602082838500/` - Checkpoint directory
  - Latest: `13850000/PPO_POLICY.pt`

### Deployment:
- `run_aerial_bot.py` - Runs bot in real Rocket League (uses Python 3.11 env)
- `aerial_bot.py` - Bot class that loads trained model
- `VutriumSDK.pyd` - DLL injection library (cp311 version, renamed correctly)

### Utilities:
- `check_progress.py` - Check training progress
- `HOW_TO_USE.txt` - User guide
- `example_client (1).py` - Original Nexto example from Vutrium

---

## HARDWARE:
- GPU: NVIDIA RTX 5060 (sm_120, Blackwell architecture)
- Issue: RTX 5060 too new for stable PyTorch
- Solution: Using PyTorch nightly for training (Python 3.10 env)

---

## TRAINING STATS (Last seen):
```
Cumulative Timesteps: 13,800,000
Policy Reward: 13.17521
Collected Steps/sec: 1,867
Overall Steps/sec: 1,801
```

---

## HOW TO RUN BOT IN ROCKET LEAGUE:

1. Make sure Rocket League is running
2. Activate Python 3.11 env:
   ```
   cd C:\Users\Aiden\Documents\aerial_bot
   rl_bot_env_311\Scripts\activate
   ```
3. Run:
   ```
   python run_aerial_bot.py
   ```
4. Join any match - bot takes control
5. Set the checkpoint location via the `AERIAL_BOT_CHECKPOINT` environment
   variable (can point to a specific `.pt`/`.pth` file or a directory with
   checkpoints). `~` and relative paths are supported and normalized.
   If unset, the bot will look for checkpoints in a `checkpoints/` folder next
   to `rlbot_bot.py`.

## PUSHING CHANGES TO GITHUB FROM THIS ENVIRONMENT:

- This environment cannot log into your GitHub account directly (no interactive
  web sign-in). To push changes yourself:
  1. Run `git status` to review modifications.
  2. Commit locally: `git commit -am "<message>"`.
  3. If needed, add the remote with a temporary fine-grained token:
     `git remote add origin https://<username>:<token>@github.com/<owner>/<repo>.git`.
  4. Push your branch: `git push origin <branch>`.
- Use the smallest scopes and a short expiration for the token, then revoke it
  after pushing.

---

## NEXT STEPS:
- [ ] Finish installing packages in Python 3.11 env
- [ ] Test bot in Rocket League
- [ ] Continue training to 25M+ steps for better aerials
- [ ] Maybe train to 50M for advanced aerial play

---

## IMPORTANT NOTES:

### VutriumSDK:
- Not publicly available (proprietary)
- User has .pyd file from intgamer0815 (Discord)
- Works by DLL injection into RL process
- Requires Rocket League to be running BEFORE starting script

### Training Checkpoints:
- Saved every 100k steps
- Format: `data/checkpoints/[run-id]/[steps]/PPO_POLICY.pt`
- Policy network: [256, 256, 256] fully connected
- Input: 92 features (DefaultObs)
- Output: 90 actions (LookupTableAction)

### Known Issues:
- Training uses Python 3.10 (for PyTorch nightly)
- Deployment uses Python 3.11 (for VutriumSDK.pyd)
- Must keep them separate!

### Running the aerial bot via Vutrium
1) Launch Rocket League and keep it running.
2) Ensure ``Vutrium.dll``/``VutriumSDK.pyd`` live next to the scripts (provided by Vutrium).
3) Provide a trained checkpoint in ``./checkpoints`` or set ``AERIAL_BOT_CHECKPOINT=/path/to/model.pt``. You can also pass ``--checkpoint /path`` to override both.
4) Run:

```bash
python run_aerial_bot.py [--checkpoint /path/to/model_or_directory]
```

---

## REWARD FUNCTION:
Bot is rewarded for:
- Being airborne (height-based)
- Being airborne when ball is high
- Touching ball (goals weighted 10x)

Bot gets penalized for:
- Staying grounded when ball is high

---

**Last Updated:** 2025-12-06
**Training Started:** ~3-4 hours ago (based on 13.8M steps at ~1800 steps/sec)
