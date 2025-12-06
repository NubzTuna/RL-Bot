# ULTIMATE AERIAL REWARD - Training Guide

## What's New?

**Combined your buddy's air dribble reward system with flip reset mechanics!**

### Reward System Features

#### 🔥 Flip Reset Detection
- Regaining flip while aerial: **+15.0**
- Using flip after reset: **+8.0**
- Inverted positioning near ball: **+0.6**

#### 💎 Air Dribble System (from C++ code)
**Geometric zone detection:**
- Ball must be >250u high
- Car below ball
- Ball in valid field geometry (calculated zone)
- Player behind ball (Y position check)

**Ascending air dribbles (both car & ball rising):**
- Perfect conditions (close, boost >30%, facing ball):
  - Touch: **+40.0**
  - Proximity: **+7.5**
  - Air roll bonus: **+12.5**
  - Facing ball: **+1.0**
- Good conditions (close, facing ball, low boost):
  - Touch: **+20.0**
  - Proximity: **+3.75**
  - Air roll bonus: **+6.25**

**Descending/level air dribbles (50% reward of ascending):**
- Same structure but halved rewards

**Multipliers:**
- Offensive alignment: scales reward by how well ball is directed toward goal
- Ball height scaling: rewards keeping ball high (maxes at 2000u)

#### ✈️ General Aerial Play
- Height-based rewards (scaled to ceiling)
- Velocity alignment toward ball
- Proximity bonuses

#### ⚽ Goals
- **+25.0** (massive!)

---

## How It Works

### Air Dribble Zone Detection
```python
# Ball position geometry (from C++ code)
ball_y = max(ball_pos[1], 0.0)
new_y = 6000.0 - ball_y
ball_x = abs(ball_pos[0]) + 92.75
largest_x = new_y * 0.683

# Valid if ball is in proper field zone
in_zone = (ball_x < largest_x)
```

### Ascending Detection
```python
ascending = (car_height > last_car_height AND ball_height > last_ball_height)
```
Both car and ball must be rising for ascending bonuses.

### Air Roll Bonus
```python
air_roll_reward = (car_angular_velocity.y / 5.5) * 5.0
```
Rewards rotation around the car's pitch axis (air roll left/right).

### Offensive Alignment
```python
# Cosine similarity between "to ball" and "to goal"
alignment = dot(to_ball, to_goal) / (|to_ball| * |to_goal|)
```
Multiplies reward based on how well positioned the ball is for scoring.

---

## Training

**Start training:**
```bash
cd C:\Users\Aiden\Documents\aerial_bot
rl_bot_env\Scripts\activate
python train_flip_reset.py
```

**What to expect:**

**Early (0-10M steps):**
- Random aerial attempts
- Learning to jump
- Occasional touches

**Mid (10-25M steps):**
- Consistent aerial movement
- First air dribble sequences
- Accidental flip resets
- Air roll usage developing

**Advanced (25-40M steps):**
- **INTENTIONAL AIR DRIBBLES** 💎
- Ascending touch chains
- Flip resets in air dribbles
- Goal-oriented play

**Expert (40M+ steps):**
- Advanced mechanics
- Flip reset → air dribble → goal
- Consistent offensive play

---

## Console Output

**You'll see:**
```
🔥 FLIP RESET #12!
   Height: 520u | In air dribble zone: True
==========================================================

💎 Air dribble touch #45!

🔄 Fully inverted! (up_z=-0.92)

💥 FLIP USED AFTER RESET! 💥

🚀 New altitude record: 892u!

📊 TRAINING STATS
   Flip Resets: 47
   Air Dribble Touches: 203
   Highest Altitude: 892u
```

---

## Key Differences from Basic Flip Reset Reward

1. **Air dribble detection** - geometric zone + ascending tracking
2. **Air roll rewards** - bonus for rotating while airborne
3. **Offensive alignment** - rewards positioning ball toward goal
4. **Touch discrimination** - ascending touches worth 2x descending
5. **Ball height scaling** - encourages keeping ball high
6. **Position checks** - rewards being behind ball (proper air dribble positioning)

---

## Fine-Tuning

**In `train_flip_reset.py`, you can adjust:**

```python
# Reward weights
ultimate_aerial = (UltimateAerialReward(), 1.0)  # Increase for more focus on mechanics
goal_reward = (GoalReward(), 25.0)  # Decrease if bot ball-chases too much
```

**In `rewards.py`, you can tweak:**
- Touch rewards (lines 139, 155): Currently 40.0 and 20.0
- Air roll multipliers (lines 143, 158): Currently 2.5 and 1.25
- Proximity scaling (lines 145, 160): Currently 7.5 and 3.75
- Flip reset reward (line 243): Currently 15.0
- Flip use after reset (line 256): Currently 8.0

---

## Why This Reward Is SICK

Your buddy's C++ reward is battle-tested - it's been used to train bots that can actually air dribble. By combining it with flip reset detection, you get a bot that learns:

1. **Flip resets** (core mechanic)
2. **Air dribbles** (advanced control)
3. **Air roll** (finesse)
4. **Positioning** (game sense)
5. **Goal-oriented play** (offensive alignment)

This is a **COMPLETE aerial mechanics reward system**! 🚀

---

**Good luck! This is gonna be INSANE once it trains!**
