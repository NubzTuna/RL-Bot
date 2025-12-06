"""
Check your bot's training progress
"""
import os
from pathlib import Path
from datetime import datetime


def check_training_progress():
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ No models directory found. Training hasn't started yet.")
        return
    
    # Find all checkpoints
    checkpoints = sorted(models_dir.glob("*.pt")) + sorted(models_dir.glob("*.pth"))
    
    if not checkpoints:
        print("❌ No checkpoints found yet. Keep training!")
        return
    
    print("=" * 60)
    print("🤖 AERIAL BOT TRAINING PROGRESS")
    print("=" * 60)
    
    for cp in checkpoints:
        # Get file info
        size_mb = cp.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(cp.stat().st_mtime)
        
        # Extract step count from filename
        steps = ''.join(filter(str.isdigit, cp.stem))
        steps_int = int(steps) if steps else 0
        steps_millions = steps_int / 1_000_000
        
        print(f"\n📁 {cp.name}")
        print(f"   Steps: {steps_int:,} ({steps_millions:.1f}M)")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Last Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    latest = checkpoints[-1]
    latest_steps = int(''.join(filter(str.isdigit, latest.stem)))
    
    print("\n" + "=" * 60)
    print(f"✅ Latest checkpoint: {latest.name}")
    print(f"   {latest_steps:,} steps ({latest_steps/1_000_000:.1f}M)")
    print("=" * 60)
    
    # Training milestones
    print("\n📊 TRAINING MILESTONES:")
    milestones = [
        (100_000, "Basic movement"),
        (500_000, "Ball awareness"),  
        (1_000_000, "Starting to aerial"),
        (5_000_000, "Consistent aerial attempts"),
        (10_000_000, "Decent aerial control"),
        (25_000_000, "Strong aerial play"),
        (50_000_000, "TARGET - Advanced aerials")
    ]
    
    for steps, desc in milestones:
        status = "✅" if latest_steps >= steps else "⏳"
        print(f"{status} {steps/1_000_000:>4.0f}M - {desc}")
    
    print("\n" + "=" * 60)
    print(f"Progress: {min(100, (latest_steps / 50_000_000) * 100):.1f}% to target")
    print("=" * 60)


if __name__ == "__main__":
    check_training_progress()
