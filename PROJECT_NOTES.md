# Vutrium Bot Notes

This repository now focuses solely on a Vutrium-compatible Rocket League bot.
Everything below is scoped to running inference through the Vutrium SDK DLL
injector—training notes and legacy aerial content were removed to start fresh.

## Key files
- `run_vutrium_bot.py` — CLI runner that injects the SDK, spins up a bot per
  local player, and streams controls back to Rocket League.
- `vutrium_bot.py` — Fresh bot implementation with a compact observation
  pipeline and lightweight policy network sized to the loaded checkpoint.
- `vutrium_checkpoint.py` — Utility for resolving the newest `.pt`/`.pth`
  checkpoint from an override path, environment variable, or local
  `checkpoints`/`models`/`weights` directories.
- `requirements.txt` — Python dependencies; the Vutrium SDK binaries must be
  placed beside the runner (`Vutrium.dll` + `VutriumSDK.pyd`).

## Running the bot (Windows example)
1. Ensure Rocket League is running before injection.
2. Open a terminal in the repo directory and activate the environment that has
   PyTorch, rlbot, and rlgym installed.
3. Provide a checkpoint via one of:
   - `--checkpoint /path/to/your/model.pt`
   - `set VUTRIUM_CHECKPOINT=C:\\path\\to\\folder` (newest `.pt`/`.pth` is
     selected automatically)
   - placing a `.pt`/`.pth` under a nearby `checkpoints/`, `models/`, or
     `weights/` directory.
4. Run:
   ```
   python run_vutrium_bot.py --checkpoint path\to\policy.pt
   ```
5. Join a match; the runner will instantiate a bot for the first local player
   and keep it updated every tick.

## Troubleshooting
- **SDK import fails**: confirm `Vutrium.dll` and `VutriumSDK.pyd` sit next to
  `run_vutrium_bot.py` and that Rocket League is already running.
- **No checkpoint found**: pass `--checkpoint`, set `VUTRIUM_CHECKPOINT`, or
  drop a `.pt`/`.pth` file into `checkpoints/`, `models/`, or `weights/` near
  the scripts.
- **Controls feel random**: the runner will still launch even if checkpoint load
  fails; check the console for load errors and verify the policy weights match
  the network shape (17-dim observation, LookupTableAction output space).
