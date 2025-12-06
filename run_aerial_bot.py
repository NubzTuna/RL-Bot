import time
from colorama import just_fix_windows_console, Fore, Style

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket

from aerial_bot import AerialBot  # YOUR TRAINED BOT!
from VutriumSDK import SDK, download_latest_and_inject, Util

def main():
    just_fix_windows_console()
    print(Fore.CYAN + "🚀 Aerial Bot - Vutrium Client" + Style.RESET_ALL)
    print(Fore.YELLOW + "Make sure Rocket League is running!" + Style.RESET_ALL)

    # 1) Download and inject latest DLL
    if not download_latest_and_inject():
        print(Fore.RED + "❌ Download/Injection failed. Make sure Rocket League is running." + Style.RESET_ALL)
        return

    print(Fore.GREEN + "✅ Vutrium SDK injected!" + Style.RESET_ALL)

    # 2) Create SDK and subscribe to events
    sdk = SDK()

    # Bot instances by player name
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
        
        cars = game.get('game_cars', [])
        locals_i = game.get('localPlayerIndices', [])
        locals_n = game.get('localPlayerNames', [])
        
        if not cars or not locals_i:
            return
        
        # Get current player
        name = locals_n[0] if locals_n else None
        idx = locals_i[0]
        
        if name is None or idx < 0 or idx >= len(cars):
            return
        
        # Create bot instance if needed
        if name not in bots_by_name:
            if not field_info_dict:
                return
            
            fi = Util.json_to_field_info_packet(field_info_dict)
            team = cars[idx].get('team', 0)
            
            print(Fore.MAGENTA + f"Creating AerialBot for player: {name} (Team {team})" + Style.RESET_ALL)
            
            # CREATE YOUR AERIAL BOT
            # Using latest checkpoint at 15.35M steps!
            checkpoint = "data/checkpoints/rlgym-ppo-run-1764991602082838500/15350000/PPO_POLICY.pt"
            bot = AerialBot(name=name, team=team, index=idx, checkpoint_path=checkpoint)
            bot.initialize_agent(fi)
            bots_by_name[name] = bot
        
        # Get bot and update its state
        bot = bots_by_name[name]
        bot.index = idx
        bot.team = cars[idx].get('team', 0)
        
        # Convert to RLBot packet format
        pkt = Util.json_to_game_tick_packet(game)
        
        # Get bot's decision
        if pkt.game_info.is_round_active:
            cs = bot.get_output(pkt)
        else:
            cs = SimpleControllerState()
        
        # Send controls back to game
        sdk.send_json({
            "num_inputs": 1,
            "inputs": [{
                "throttle": float(cs.throttle),
                "steer": float(cs.steer),
                "pitch": float(cs.pitch),
                "yaw": float(cs.yaw),
                "roll": float(cs.roll),
                "jump": bool(cs.jump),
                "boost": bool(cs.boost),
                "handbrake": bool(cs.handbrake),
                "use_item": bool(cs.use_item)
            }]
        })

    sdk.subscribe("OnGameEventStart", on_start)
    sdk.subscribe("OnGameEventDestroyed", on_destroy)
    sdk.subscribe("PlayerTickHook", on_tick)
    sdk.start()

    print(Fore.GREEN + "\n✅ Bot is ready! Join a Rocket League match now." + Style.RESET_ALL)
    print(Fore.CYAN + "Press Ctrl+C to stop\n" + Style.RESET_ALL)

    # 3) Keep the process alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n👋 Shutting down..." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
