# CLAUDE MEMORY - AERIAL BOT PROJECT - FULL TECHNICAL REFERENCE

## CRITICAL CONTEXT FOR FUTURE CLAUDE INSTANCES

**User:** Aiden, student at Embry-Riddle Aeronautical University, Daytona Beach, FL
**Expertise:** Strong programming skills, MATLAB, Python, automation games (Factorio), direct communication style
**Communication Preference:** Casual, practical, skip theory, "just do it", doesn't want hand-holding
**Project Goal:** Train aerial-focused Rocket League bot using reinforcement learning, deploy via DLL injection

---

## CONVERSATION HISTORY SUMMARY

### Session Start:
- User wanted to modify existing bot "Vitrium" to do aerials instead of dribbling
- Initial approach: modify reward function of existing bot
- Pivoted: Decided to train custom bot from scratch using RLGym-PPO

### Major Technical Challenges Solved:

#### Challenge 1: Python Version Hell
- Started with Python 3.10.11
- rlgym 1.2.2 requires Python ≤3.9 (incompatible)
- rlgym 2.0.1 exists but API completely different
- **Solution:** Use rlgym 2.0 + rlgym-ppo (GitHub install)
- Multiple failed installation attempts before finding working combo

#### Challenge 2: GPU Compatibility (CRITICAL)
- User has NVIDIA RTX 5060 (brand new, Blackwell architecture)
- Compute capability: sm_120
- Stable PyTorch only supports up to sm_90
- **Solution:** PyTorch NIGHTLY build with CUDA 12.8
- Installation command: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`
- This was ESSENTIAL - stable PyTorch would NOT use GPU

#### Challenge 3: RLGym 2.0 API Differences
- Reward functions in 2.0 return DICT not single value
- Method is `get_rewards()` (plural) not `get_reward()`
- Takes 3 args in reset(), not 2
- CombinedReward uses positional args, not kwargs
- NoTouchTimeoutCondition takes `timeout_seconds` not `timeout=`

#### Challenge 4: Training Script Errors
- First error: `NoTouchTimeoutCondition.__init__() got unexpected keyword argument 'timeout'`
  - Fix: Change to positional arg `NoTouchTimeoutCondition(timeout_seconds)`
- Second error: `AttributeError: 'Car' object has no attribute 'ball_touch'`
  - Fix: Simplified reward function to remove ball touch detection
  - Used height-based rewards instead

#### Challenge 5: VutriumSDK Deployment
- User has proprietary .pyd file: `VutriumSDK.cp311-win_amd64 (1).pyd`
- cp311 = compiled for Python 3.11 ONLY
- No cp310 version exists
- **Solution:** Create SECOND virtual environment with Python 3.11
- Training stays on Python 3.10 (for PyTorch nightly)
- Deployment uses Python 3.11 (for VutriumSDK)

---

## TECHNICAL ARCHITECTURE

### Training Environment (Python 3.10.11)
**Virtual Env:** `rl_bot_env/`
**Packages:**
- torch 2.8.0.dev (nightly CUDA 12.8) - CRITICAL for RTX 5060
- rlgym-ppo 1.3.13 (from GitHub)
- rlgym-rocket-league 2.0.1 (RocketSim backend)
- numpy, etc.

**Key Script:** `train.py`
```python
# Environment Config:
- 1v1 (spawn_opponents=True)
- 8 tick skip (120 FPS / 8 = 15 Hz agent updates)
- 10s no-touch timeout
- LookupTableAction (90 discrete actions)
- DefaultObs (92 features)

# Network Architecture:
- Policy: [256, 256, 256] fully connected
- Critic: [256, 256, 256] fully connected
- Input: 92 (obs space)
- Output: 90 (action space)

# Hyperparameters:
- n_proc=2 (changed from 8 - user's request)
- lr=5e-5 (both policy and critic)
- batch_size=100k
- minibatch_size=50k
- ts_per_iteration=50k
- exp_buffer_size=150k
- ppo_epochs=2
- device="cuda"
```

**Reward Function:** `rewards.py` - `AerialFocusedReward`
```python
# Simplified design (no ball_touch attribute)
# Rewards:
- 0.5 * (height/2044) if car_height > 200
- +0.3 if ball_height > 500 (while airborne)
- +0.1 if car_height > 50 (slightly airborne)
# Combined with:
- GoalReward (weight 10.0)
- TouchReward (weight 0.1)
# Output: Dict[AgentID, float] in range [-1, 1]
```

**Training Performance:**
- Steps/sec: ~1,800-1,900 (on RTX 5060)
- Checkpoint every 100k steps
- Currently at: 15.35M+ steps (actively training)

### Deployment Environment (Python 3.11.0)
**Virtual Env:** `rl_bot_env_311/`
**Packages:** (currently installing)
- torch (stable CUDA 12.1) - nightly failed, switched to stable
- rlgym-ppo, rlgym[rl-sim]
- rlbot, colorama, numpy
- VutriumSDK.pyd (proprietary binary)

**Key Script:** `run_aerial_bot.py`
```python
# VutriumSDK Flow:
1. download_latest_and_inject() - DLL injection into RL process
2. SDK() - Create SDK instance
3. Subscribe to events:
   - PlayerTickHook - game state every frame
   - OnGameEventStart - match start
   - OnGameEventDestroyed - match end
4. on_tick():
   - Receive JSON game state
   - Convert to RLBot packet format
   - Pass to bot.get_output()
   - Send SimpleControllerState back as JSON
```

**Bot Class:** `aerial_bot.py`
```python
# Model Loading:
- Checkpoint path: data/checkpoints/.../15350000/PPO_POLICY.pt
- State dict only (not full model)
- Must recreate architecture:
  - nn.Sequential with [256,256,256] layers
  - Input: 92, Output: 90
  - ReLU activations
- Load weights: policy.load_state_dict(torch.load(...))

# Observation Conversion:
- RLBot GameTickPacket → numpy array
- Normalize positions by field dimensions
- Normalize velocities by max speeds
- Output: 92-element array matching training obs

# Action Conversion:
- Model output: action logits (90 values)
- argmax → action index
- LookupTableAction.lookup_table[idx] → 8-element array
- Array → SimpleControllerState
  - [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
```

---

## FILE STRUCTURE

```
aerial_bot/
├── train.py                    # Training script (ACTIVE, Python 3.10)
├── rewards.py                  # AerialFocusedReward class
├── aerial_bot.py               # Bot class for deployment
├── run_aerial_bot.py           # VutriumSDK deployment script (Python 3.11)
├── check_progress.py           # Training progress checker
├── VutriumSDK.pyd             # DLL injection library (cp311)
├── PROJECT_NOTES.md           # User-facing notes
├── HOW_TO_USE.txt             # User guide
├── rl_bot_env/                # Python 3.10 venv (training)
├── rl_bot_env_311/            # Python 3.11 venv (deployment)
├── data/
│   └── checkpoints/
│       └── rlgym-ppo-run-1764991602082838500/
│           ├── 14950000/
│           ├── 15050000/
│           ├── 15150000/
│           ├── 15250000/
│           └── 15350000/      # LATEST (15.35M steps)
│               ├── PPO_POLICY.pt
│               ├── PPO_POLICY_OPTIMIZER.pt
│               ├── PPO_VALUE_NET.pt
│               ├── PPO_VALUE_NET_OPTIMIZER.pt
│               └── BOOK_KEEPING_VARS.json
├── logs/                      # TensorBoard logs (if enabled)
└── models/                    # Empty (checkpoints in data/)
```

---

## CHECKPOINT STRUCTURE (RLGym-PPO format)

Each checkpoint directory contains:
- `PPO_POLICY.pt` - Policy network state_dict (what we load)
- `PPO_POLICY_OPTIMIZER.pt` - Optimizer state (for resuming training)
- `PPO_VALUE_NET.pt` - Value network state_dict
- `PPO_VALUE_NET_OPTIMIZER.pt` - Value optimizer state
- `BOOK_KEEPING_VARS.json` - Metadata (step count, etc.)

**CRITICAL:** PPO_POLICY.pt is a state_dict, NOT a full model
- Must recreate nn.Sequential architecture
- Then load_state_dict()
- Architecture MUST match training config

---

## KNOWN ISSUES & GOTCHAS

### Issue 1: Two Python Versions Required
- Training: Python 3.10 (for PyTorch nightly)
- Deployment: Python 3.11 (for VutriumSDK.pyd)
- DO NOT mix them up
- Keep separate virtual envs

### Issue 2: VutriumSDK Requirements
- Rocket League MUST be running FIRST
- Then run script
- If RL not running: injection fails
- Might need admin privileges
- Anti-cheat might interfere (Vutrium designed to avoid this)

### Issue 3: Observation Space Mismatch Risk
- Training uses DefaultObs (92 features)
- Deployment must match EXACTLY
- aerial_bot.py implements simplified conversion
- May need tuning if bot behaves weirdly

### Issue 4: Action Space
- LookupTableAction has 90 discrete actions
- Covers all combinations of:
  - Throttle: [-1, 0, 1]
  - Steer: [-1, 0, 1]
  - Pitch: [-1, 0, 1]
  - Yaw: [-1, 0, 1]
  - Roll: [-1, 0, 1]
  - Jump: [0, 1]
  - Boost: [0, 1]
  - Handbrake: [0, 1]
- Actually 3^5 * 2^3 = 243 * 8 = way more combinations
- LookupTableAction pre-filters to 90 "good" combos

### Issue 5: GPU Warnings Are Normal
- RTX 5060 shows "sm_120 not compatible" warning
- This is just a warning, GPU still works
- PyTorch nightly supports sm_120 in practice

---

## TRAINING MILESTONES & EXPECTATIONS

Based on similar RL bot projects:

**0-1M steps (~30 min):**
- Random movement
- Learning basic controls
- Reward: ~0-5

**1-5M steps (~2 hrs):**
- Drives toward ball
- First aerial attempts (random)
- Reward: ~5-10

**5-10M steps (~5 hrs):**
- Consistent aerial jumps
- Occasional aerial hits
- Reward: ~10-15

**10-25M steps (~12 hrs):**
- Good aerial control
- Intentional aerial hits
- Fast aerials starting
- Reward: ~15-25

**25M+ steps (20+ hrs):**
- Strong aerial play
- Advanced mechanics
- Fast aerials, air dribbles
- Reward: ~25-40+

**Current status: 15.35M steps**
- Should have decent aerial attempts
- Won't be perfect yet
- Needs ~25M for "good" performance

---

## DEBUGGING TIPS FOR FUTURE CLAUDE

### If training crashes:
1. Check GPU usage in Task Manager
2. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check if PyTorch nightly installed: `pip list | grep torch`
4. Look for out-of-memory errors (reduce n_proc or batch sizes)

### If deployment fails:
1. Verify Python 3.11: `python --version`
2. Check VutriumSDK.pyd exists and is renamed correctly
3. Verify Rocket League is running
4. Try running terminal as Administrator
5. Check checkpoint path is correct

### If bot behaves weirdly:
1. Check observation conversion (print obs array)
2. Verify model loaded correctly (check state_dict keys)
3. Test with different checkpoints
4. Check if action parser matches training

### If no checkpoints found:
- Check: `data/checkpoints/rlgym-ppo-run-*/`
- NOT in `models/` directory
- RLGym-PPO saves to data/ by default

---

## USER INTERACTION PATTERNS

**Communication style:**
- Direct, casual
- Hates over-explanation
- Wants solutions, not theory
- "just do it" mentality
- Appreciates when Claude is proactive
- Gets frustrated with verbose responses
- Values efficiency over politeness

**When stuck:**
- User will say "remember you can look at files"
- User expects Claude to use filesystem tools proactively
- User wants Claude to search/verify, not guess
- Phrase "maybe look it up?" = use web_search

**Red flags:**
- If user says "hello?" = Claude is taking too long or being unhelpful
- If user shares Discord screenshot = showing evidence/context
- If user says "bruh" = mild frustration, get to the point

---

## EXTERNAL REFERENCES

### VutriumSDK
- Source: intgamer0815 (Discord user)
- Not publicly available
- Proprietary DLL injection system for Rocket League
- Works by hooking game process
- Safer than BakkesMod for online play (allegedly)

### RLGym-PPO
- GitHub: https://github.com/AechPro/rlgym-ppo
- Creator: AechPro
- PPO implementation optimized for RLGym
- Uses RocketSim (headless RL physics simulator)

### RocketSim
- Headless Rocket League physics simulator
- Runs ~100x faster than real game
- No actual RL installation needed for training
- Used by: Nexto, Necto, most modern RL bots

---

## NEXT ACTIONS (Priority Order)

1. **IMMEDIATE:** Finish installing packages in Python 3.11 env
   - Currently stuck on PyTorch install (slow)
   - Switched from nightly to stable

2. **TEST:** Run bot in Rocket League
   - User wants to see if it works
   - 15.35M steps should show aerial attempts
   - Watch for injection issues

3. **CONTINUE TRAINING:** Let training run to 25M+
   - Currently at 15.35M
   - Target: 25M for strong aerials
   - Stretch goal: 50M for advanced play

4. **OPTIONAL:** Fine-tune reward function
   - If bot not aerial enough, increase aerial rewards
   - If bot too reckless, add positioning rewards
   - User comfortable with Python, can modify himself

---

## TECHNICAL DECISIONS MADE

**Why not use existing Vitrium bot?**
- User wanted full control
- Existing bot optimized for dribbling
- Easier to train from scratch than reverse-engineer

**Why RLGym-PPO instead of stable-baselines3?**
- RLGym-PPO optimized for RL specifically
- Better performance (faster training)
- Active community

**Why discrete actions instead of continuous?**
- Simpler to implement
- Faster training
- RL doesn't benefit much from continuous control
- LookupTableAction covers necessary actions

**Why two Python environments?**
- No choice - VutriumSDK.pyd only for Python 3.11
- PyTorch stable didn't support RTX 5060
- Could have downgraded GPU usage, but that defeats the purpose

---

## COST ESTIMATES

**Training time to 50M steps:**
- At 1800 steps/sec: ~7.7 hours
- GPU cost: ~$0.50-1.00 in electricity
- Already at 15.35M = ~2.4 hours elapsed

**Inference cost:**
- Negligible (single forward pass per frame)
- GPU can handle 1000+ FPS inference
- RL runs at 120 FPS, bot at 15 Hz

---

## FILES SAFE TO DELETE (if needed)

- `logs/` - TensorBoard logs (regenerated)
- `__pycache__/` - Python cache
- `observations.py` - Not used (DefaultObs instead)
- `play_in_rl.py` - Old attempt with rlgym_tools
- `watch_bot.py` - For visualizing, not essential
- `example_client (1).py` - Reference only
- Old checkpoints if disk space needed (keep latest few)

---

## FILES NEVER DELETE

- `train.py` - Main training script
- `rewards.py` - Custom reward function
- `aerial_bot.py` - Bot implementation
- `run_aerial_bot.py` - Deployment script
- `VutriumSDK.pyd` - Cannot be replaced
- `data/checkpoints/` - ALL TRAINING PROGRESS

---

## SYSTEM SPECS

**Hardware:**
- CPU: Unknown (works fine, not bottleneck)
- GPU: NVIDIA GeForce RTX 5060
  - Compute capability: sm_120 (Blackwell)
  - VRAM: Unknown (likely 8-16GB)
  - Performance: ~1800 steps/sec training
- RAM: Unknown (sufficient for training)
- OS: Windows 11

**Software:**
- Python 3.10.11 (training env)
- Python 3.11.0 (deployment env)
- CUDA: 12.8 (via PyTorch nightly)
- Rocket League: Unknown version (must be recent)

---

## DATETIME CONTEXT

**Session date:** 2025-12-06
**Training start:** ~4-5 hours before session (estimated from step count)
**Total training time:** 15.35M / 1800 steps/sec ≈ 2.4 hours actual compute
  - Implies started training, stopped, restarted multiple times
  - Or training speed varied
  - Or checkpoint count is from resumed training

**Training runs detected:**
- `rlgym-ppo-run-1764984075826261900` - 7.5M - 7.85M steps
- `rlgym-ppo-run-1764988177101760700` - 7.95M steps
- `rlgym-ppo-run-1764991602082838500` - 13.45M - 15.35M+ (CURRENT)

This suggests user:
1. Started training
2. Stopped (maybe crashed or tested)
3. Restarted (new run ID)
4. Currently on 3rd training run
5. Still actively training (checkpoints increasing)

---

## IMPORTANT: PROACTIVE BEHAVIOR EXPECTED

User expects Claude to:
- Check filesystem without being asked
- Search documentation when unsure
- Verify assumptions with tools
- Create helpful artifacts (like this file)
- Remember context across sessions
- Be direct and efficient

User does NOT want:
- Long explanations
- Asking permission before using tools
- Repetitive information
- Apologizing excessively
- Disclaimers about being an AI

---

**END OF TECHNICAL REFERENCE**
**Last updated:** 2025-12-06 by Claude (Sonnet 4.5)
**Update this file whenever significant changes occur**
