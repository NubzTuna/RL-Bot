"""RLBot-compatible wrapper for your trained aerial bot."""
from pathlib import Path

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

from aerial_bot import AerialBot
from checkpoint_utils import resolve_checkpoint_path


class RLBotAerialBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        checkpoint_path = resolve_checkpoint_path(Path(__file__))
        self.bot = AerialBot(name, team, index, checkpoint_path)
        
    def initialize_agent(self):
        """Called once when the bot starts"""
        self.bot.initialize_agent()
        
    def get_output(self, packet):
        """Called every game tick"""
        return self.bot.get_output(packet)
