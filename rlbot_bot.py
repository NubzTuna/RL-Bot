"""
RLBot-compatible wrapper for your trained aerial bot
"""
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from aerial_bot import AerialBot


class RLBotAerialBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        
        # Path to your latest checkpoint
        checkpoint_path = r"C:\Users\Aiden\Documents\aerial_bot\data\checkpoints\rlgym-ppo-run-1764991602082838500\15350000\PPO_POLICY.pt"
        
        self.bot = AerialBot(name, team, index, checkpoint_path)
        
    def initialize_agent(self):
        """Called once when the bot starts"""
        self.bot.initialize_agent()
        
    def get_output(self, packet):
        """Called every game tick"""
        return self.bot.get_output(packet)
