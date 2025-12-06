import time
from colorama import just_fix_windows_console, Fore, Style

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket

from nexto.bot import Nexto
#from yourbot.bot import RLGymPPOBot
from VutriumSDK import SDK, download_latest_and_inject, Util

def main():
    just_fix_windows_console()
    print(Fore.MAGENTA + "Vutrium Example Client (Nexto-only)" + Style.RESET_ALL)

    # 1) Download and inject latest DLL before doing anything
    if not download_latest_and_inject():
        print(Fore.RED + "Download/Injection failed. Make sure RocketLeague is running." + Style.RESET_ALL)
        return

    # 2) Create SDK and subscribe to events
    sdk = SDK()

    # Minimal state: single Nexto instance keyed by player name discovered
    nexto_by_name = {}
    field_info_dict = None
    last_tick = None

    def on_start(evt: dict):
        pass

    def on_destroy(evt: dict):
        nexto_by_name.clear()

    def on_tick(evt: dict):
        nonlocal field_info_dict, last_tick
        game = evt.get("gameTickPacket") or {}
        field_info_dict = field_info_dict or evt.get("fieldInfoPacket")
        if not game:
            return
        cars = game.get('game_cars', [])
        locals_i = game.get('localPlayerIndices', [])
        locals_n = game.get('localPlayerNames', [])
        if not cars or not locals_i:
            return
        # current ticking player
        name = locals_n[0] if locals_n else None
        idx = locals_i[0]
        if name is None or idx < 0 or idx >= len(cars):
            return
        if name not in nexto_by_name:
            if not field_info_dict:
                return
            fi = Util.json_to_field_info_packet(field_info_dict)
            team = cars[idx].get('team', 0)
            bot = Nexto(name=name, team=team, index=idx)
            # IMPORTANT if you want to use your own bot, you need to use this:
            #bot = RLGymPPOBot(name=name, team=team_idx, index=car_index)
            #bot._BaseAgent__field_info_func = lambda: field_info_packet
            #bote.initialize_agent()
            bot.initialize_agent(fi)
            nexto_by_name[name] = bot
        bot = nexto_by_name[name]
        bot.index = idx
        bot.team = cars[idx].get('team', 0)
        pkt = Util.json_to_game_tick_packet(game)
        if pkt.game_info.is_round_active:
            cs = bot.get_output(pkt)
        else:
            cs = SimpleControllerState()
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

    # 3) Keep the process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()


