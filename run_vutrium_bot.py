"""Vutrium entrypoint that builds a fresh bot instance per local player."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from colorama import Fore, Style, just_fix_windows_console
from rlbot.agents.base_agent import SimpleControllerState

from vutrium_bot import VutriumBot
from vutrium_checkpoint import resolve_checkpoint


def _load_vutrium_sdk():
    try:
        from VutriumSDK import SDK, download_latest_and_inject, Util  # type: ignore

        return SDK, download_latest_and_inject, Util
    except Exception as exc:  # pragma: no cover - runtime clarity
        print(
            Fore.RED
            + "VutriumSDK failed to import. Ensure Vutrium.dll/VutriumSDK.pyd sits next to this script "
            "and Rocket League is running."
            + Style.RESET_ALL
        )
        print(f"Import error: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the brand-new Vutrium bot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path or directory for the policy checkpoint (overrides VUTRIUM_CHECKPOINT)",
    )
    args = parser.parse_args()

    just_fix_windows_console()
    print(Fore.CYAN + "🚀 Vutrium Bot Runner" + Style.RESET_ALL)
    print(Fore.YELLOW + "Make sure Rocket League is running before injecting." + Style.RESET_ALL)

    SDK, download_latest_and_inject, Util = _load_vutrium_sdk()

    if not download_latest_and_inject():
        print(Fore.RED + "❌ Download/Injection failed. Start Rocket League and try again." + Style.RESET_ALL)
        return

    print(Fore.GREEN + "✅ Vutrium SDK injected!" + Style.RESET_ALL)

    sdk = SDK()
    bots_by_name = {}
    field_info_dict = None

    def on_start(evt: dict):
        print(Fore.GREEN + "🎮 Game started!" + Style.RESET_ALL)

    def on_destroy(evt: dict):
        print(Fore.YELLOW + "Game ended, clearing bots..." + Style.RESET_ALL)
        bots_by_name.clear()

    def on_tick(evt: dict):
        nonlocal field_info_dict

        game = evt.get("gameTickPacket") or {}
        field_info_dict = field_info_dict or evt.get("fieldInfoPacket")

        if not game:
            return

        cars = game.get("game_cars", [])
        locals_i = game.get("localPlayerIndices", [])
        locals_n = game.get("localPlayerNames", [])

        if not cars or not locals_i:
            return

        name = locals_n[0] if locals_n else None
        idx = locals_i[0]

        if name is None or idx < 0 or idx >= len(cars):
            return

        if name not in bots_by_name:
            if not field_info_dict:
                return
            fi = Util.json_to_field_info_packet(field_info_dict)
            team = cars[idx].get("team", 0)

            checkpoint = resolve_checkpoint(Path(__file__), args.checkpoint)
            print(
                Fore.MAGENTA
                + f"Creating VutriumBot for player: {name} (Team {team}) using {checkpoint}"
                + Style.RESET_ALL
            )

            bot = VutriumBot(name=name, team=team, index=idx, checkpoint_override=str(checkpoint))
            bot.initialize_agent(fi)
            bots_by_name[name] = bot

        bot = bots_by_name[name]
        bot.index = idx
        bot.team = cars[idx].get("team", 0)

        pkt = Util.json_to_game_tick_packet(game)

        if pkt.game_info.is_round_active:
            cs = bot.get_output(pkt)
        else:
            cs = SimpleControllerState()

        sdk.send_json(
            {
                "num_inputs": 1,
                "inputs": [
                    {
                        "throttle": float(cs.throttle),
                        "steer": float(cs.steer),
                        "pitch": float(cs.pitch),
                        "yaw": float(cs.yaw),
                        "roll": float(cs.roll),
                        "jump": bool(cs.jump),
                        "boost": bool(cs.boost),
                        "handbrake": bool(cs.handbrake),
                        "use_item": bool(cs.use_item),
                    }
                ],
            }
        )

    sdk.subscribe("OnGameEventStart", on_start)
    sdk.subscribe("OnGameEventDestroyed", on_destroy)
    sdk.subscribe("PlayerTickHook", on_tick)
    sdk.start()

    print(Fore.GREEN + "\n✅ Bot is ready! Join a Rocket League match now." + Style.RESET_ALL)
    print(Fore.CYAN + "Press Ctrl+C to stop\n" + Style.RESET_ALL)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n👋 Shutting down..." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
