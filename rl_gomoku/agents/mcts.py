import copy
import warnings

from ..envs import GomokuEnv
from typing import Optional, Tuple, Dict
import random
import numpy as np
from .heuristic import HeuristicAgent
import torch
import torch.nn as nn
import torch.nn.functional as F


class Node:
    def __init__(self, parent=None, position=None, player=None, depth=None):
        self.parent: Optional[Node] = parent
        self.position: Tuple[int] = position
        self.player: int = player  # 1 or 2
        self.depth: int = depth

        self.n_wins: int = 0
        self.n_visits: int = 0

        self.value: float = 0.
        self.children: Dict[Tuple[int], Node] = {}

    def __repr__(self):
        return f"""Node(player: {self.player}, position: {self.position}, value: {self.value}, visits: {self.n_visits}, depth: {self.depth})"""

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

    # def expand(self, action_priors):
    #     """Expand tree by creating new children.
    #     action_priors: a list of tuples of actions and their prior probability
    #         according to the policy function.
    #     """
    #     for action, prob in action_priors:
    #         if action not in self._children:
    #             self._children[action] = Node(self, prob)

    # def select(self, c_puct):
    #     """Select action among children that gives maximum action value Q
    #     plus bonus u(P).
    #     Return: A tuple of (action, next_node)
    #     """
    #     return max(self._children.items(),
    #                key=lambda act_node: act_node[1].get_value(c_puct))


class MCTSAgent:
    def __init__(self,
                 env: GomokuEnv,
                 max_search_num: int = 300,
                 c_param: float = 1.,
                 random_state: int = 42,
                 verbose: bool = False,
                 player_id: int = 1,
                 ):
        self.env = copy.deepcopy(env)  # a copy of the environment
        self.max_search_num = max_search_num
        self.c_param = c_param
        self.root = Node(parent=None, depth=0)
        self.current_node: Optional[Node] = self.root
        self.rng = random.Random(random_state)
        self.verbose = verbose

    def _step(self, node: Node):
        self.current_node = node
        self.current_node.n_visits += 1
        self.env.step(self.current_node.position)

    def update(self, position):
        if position in self.current_node.children:
            # if the position is adjacent from the current node, we set the root to this node
            self.root = self.current_node.children[position]
            self.root.parent = None
            self._step(self.root)
        else:
            # otherwise, we create a new node and search the best strategy based on the current state
            node = Node(
                parent=self.root,
                position=position,
                player=self.env.current_player,
                depth=self.current_node.depth + 1,
            )
            self.root = node
            self.root.parent = None
            self._step(self.root)

    @staticmethod
    def calc_ucb(value, n_visits, total_visits, c_param):
        part_1 = value / n_visits
        part_2 = c_param * np.sqrt(2 * np.log(total_visits) / n_visits)
        return part_1 + part_2

    def choose_best_child(self):
        zero_visits = []
        total_visits = 0
        for key, child in self.current_node.children.items():
            total_visits += child.n_visits
            if child.n_visits == 0:
                zero_visits.append(child)

        if zero_visits:
            return self.rng.choice(zero_visits)
        else:
            tmp_dict = self.current_node.children
            best_node = max(tmp_dict.values(),
                            key=lambda node: self.calc_ucb(node.value,
                                                           node.n_visits,
                                                           total_visits,
                                                           self.c_param))
        return best_node

    def expand(self):
        if self.current_node.children:
            warnings.warn('This node is already expanded.', Warning, 2)
        else:
            available_positions = self.env.adjacent_vacancies()
            for position in available_positions:
                child = Node(
                    parent=self.current_node,
                    player=3 - self.current_node.player,
                    depth=self.current_node.depth + 1,
                    position=position
                )
                self.current_node.children[position] = child

    def roll_out(self) -> int:
        count = 0
        while not self.env.is_ended():
            available_positions = self.env.adjacent_vacancies()
            if not available_positions:
                return 0

            position = self.rng.choice(list(available_positions))
            self.env.step(position)
            count += 1

        if self.env.winner == self.current_node.player:
            return 1
        elif self.env.winner == 3 - self.current_node.player:
            return -1
        else:
            return 0

    def back_propagate(self, reward):
        while self.current_node.parent is not None:
            self.current_node.value += reward
            self.current_node = self.current_node.parent
            reward = -reward

    def best_child(self):
        max_node = max(self.current_node.children.values(), key=lambda node: node.n_visits)
        return max_node

    def _take_action(self, position):
        # update the environment, set the root node to the opponent move
        self.update(position)

        if self.env.is_ended():
            return None

        for _ in range(self.max_search_num):
            # before planning, we need to save checkpoints
            self.env.slow_save_checkpoints()

            # if current node has children, we choose the best
            while self.current_node.children:
                child_node = self.choose_best_child()
                self._step(child_node)

            if self.env.is_ended() and self.verbose:
                print(f"simulated winner: {self.env.current_player}")
            else:
                # otherwise, we expand the node, and then choose the best node
                self.expand()
                # using ucb to explore
                child_node = self.choose_best_child()
                self._step(child_node)

            # when switching to the next node, we apply random strategy to simulate the following game until end.
            reward = self.roll_out()
            # back propagate the reward to the root
            self.back_propagate(reward)
            # recover the environment to the previous checkpoint
            self.env.slow_load_checkpoints()

        # based on plannings, we choose the best child
        best_move = self.best_child().position
        self.update(best_move)
        return best_move

    def take_action(self, last_move):
        x, y = last_move
        if x == -1:
            middle_point = int(self.env.size / 2)
            self.update((middle_point, middle_point))
            return middle_point, middle_point
        else:
            return self._take_action((x, y))
