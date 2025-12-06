# FLIP RESET TRAINING GUIDE

## What's New?

I created a **comprehensive flip reset reward system** that teaches your bot to:
1. Stay airborne when the ball is high
2. Fly toward the ball quickly
3. Approach with inverted/sideways orientation
4. Touch the ball while aerial
5. **GET FLIP RESETS** (massive +10.0 reward!)
6. Use the flip after getting a reset (+5.0 reward)
7. Score goals (+20.0 reward)

## Reward Breakdown

### FlipResetMasterReward
**This is the main reward - it's CRAZY comprehensive:**

**Aerial Behavior:**
- Being airborne: +0.2 to +0.5 (scales with height)
- Moving toward ball while aerial: +0.5 (speed × alignment)
- Getting close to ball: +0.3 (proximity bonus)

**Orientation Rewards:**
- Inverted (upside down) near ball: +0.4
- Sideways near ball: +0.3
- *Why? Flip resets require wheels touching ball - easier when inverted!*

**Touch Rewards:**
- Aerial ball touch: +2.0
- Fast ball velocity after touch: +1.0

**THE BIG ONES:**
- **FLIP RESET DETECTED: +10.0** 🔥
  - Triggers when: car regains flip while aerial after touching ball
  - Prints "🔥 FLIP RESET DETECTED! 🔥" to console
- **Using flip after reset: +5.0** 💥
  - Must use flip within 2 seconds of getting reset
  - Prints "💥 USED FLIP AFTER RESET! 💥"

**Punishments:**
- Staying on ground when ball is high: -0.1

### SimpleFlipResetReward
**Lighter version if the main one is too complex:**
- Basic aerial: +0.1
- Flip reset: +5.0
- Ball touch: +0.5
- Aerial ball touch: +1.5

## How to Train

### Option 1: Start Fresh (Recommended)
```bash
# Switch to Python 3.10 training environment
rl_bot_env\Scripts\activate

# Start new training run with flip reset rewards
python train_flip_reset.py
```

This starts a **brand new training run** focused on flip resets.

### Option 2: Continue Your Current Training
If you want to keep your current 15.35M steps and just add flip reset rewards:

1. Stop your current training (Ctrl+C)
2. Backup your current train.py:
   ```bash
   copy train.py train_old.py
   ```
3. Replace train.py with train_flip_reset.py:
   ```bash
   copy train_flip_reset.py train.py
   ```
4. Restart training:
   ```bash
   python train.py
   ```

Your bot will continue from 15.35M steps but now optimize for flip resets!

## What to Expect

**Early training (0-5M steps):**
- Bot learns to jump
- Basic aerial movement
- Occasional ball touches

**Mid training (5-15M steps):**
- Consistent aerial hits
- Flying toward ball
- First flip reset attempts (probably accidental)

**Advanced training (15-30M steps):**
- Intentional aerial approaches
- Inverted/sideways positioning
- **FIRST SUCCESSFUL FLIP RESETS!** 🎉
- Using flips after resets

**Expert level (30M+ steps):**
- Consistent flip resets
- Chaining mechanics
- Advanced aerial plays
- Flip reset → goal sequences

## Monitoring Training

Watch the console output for:
```
🔥 FLIP RESET DETECTED! 🔥
💥 USED FLIP AFTER RESET! 💥
```

These indicate your bot is learning the mechanic!

## Reward Weights

In `train_flip_reset.py`, rewards are weighted:
```python
flip_reset_reward = (FlipResetMasterReward(), 1.0)
goal_reward = (GoalReward(), 20.0)
touch_reward = (TouchReward(), 0.05)
```

**You can tune these!** Examples:
- More flip reset focus: `(FlipResetMasterReward(), 2.0)`
- Less goal focus: `(GoalReward(), 10.0)`
- More ball touches: `(TouchReward(), 0.2)`

## Files Created

- `rewards.py` - Contains 3 reward functions:
  - **FlipResetMasterReward** (comprehensive, recommended)
  - **SimpleFlipResetReward** (lighter version)
  - **AerialFocusedReward** (your original, kept for reference)

- `train_flip_reset.py` - New training script using flip reset rewards

## Tips

1. **Patience!** Flip resets are HARD. Even at 50M steps, consistency might be low.

2. **Watch for console spam** - the print statements will fire A LOT once the bot starts getting resets. You can comment them out in `rewards.py` if annoying.

3. **Curriculum learning** - You could start with SimpleFlipResetReward, then switch to FlipResetMasterReward once basic aerials are learned.

4. **Tune the timeout** - I set it to 15 seconds in `train_flip_reset.py`. Longer = more time for complex plays, but slower training.

## Technical Details

**Flip reset detection:**
```python
# Checks if car regained flip ability while aerial after ball touch
regained_flip = player.has_flip and not self.prev_has_flip[player_id]
if regained_flip and is_aerial and just_touched_ball:
    reward += 10.0  # FLIP RESET!
```

**Orientation check:**
```python
# Rewards inverted/sideways orientation near ball
car_up[2] < -0.3  # Inverted (upside down)
abs(car_up[2]) < 0.4  # Sideways
```

**Velocity alignment:**
```python
# Rewards flying toward ball
alignment = np.dot(velocity_direction, direction_to_ball)
reward += 0.5 * alignment * speed_factor
```

---

**LET'S GOOOO! Your bot is about to learn FLIP RESETS! 🚀**
