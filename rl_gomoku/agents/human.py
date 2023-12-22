import numpy as np


class HumanAgent:
    def __init__(self, env, player_id=1):
        self.player_id = player_id
        self.env = env

    def take_action(self, state):
        act = input("Your turn: ").split()
        # act = np.array([act], dtype=int)
        return act
