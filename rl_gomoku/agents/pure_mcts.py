import copy

import numpy as np
from ..envs import GomokuEnv
from operator import itemgetter


def rollout_policy_fn(env: GomokuEnv):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    legal_positions = env.adjacent_vacancies()
    action_probs = np.random.rand(len(legal_positions))
    return zip(legal_positions, action_probs)


def random_policy_value_fn(env: GomokuEnv):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    legal_positions = env.adjacent_vacancies()
    action_probs = np.ones(len(legal_positions)) / len(env.availables)
    return zip(legal_positions, action_probs), 0


def ensure_idx(x, size=15):
    if isinstance(x, tuple):
        return x[0] * size + x[1]
    else:
        return x


def idx_to_position(idx, size=15):
    return idx // size, idx % size


class Node:
    def __init__(self, parent, prior_prob, position_idx=None):
        self.parent = parent
        self.position_idx = position_idx
        self.n_visits = 0
        self.Q = 0.
        self.children = {}
        self.prior_prob = prior_prob

    def expand(self, prior_probs: list[tuple]):
        for action, prob in prior_probs:
            if action not in self.children:
                # 感觉这里不合理, 这个先验概率, 本质上是条件概率, 不同状态的条件概率在这里进行了混用
                self.children[action] = Node(self, prob, ensure_idx(action))

    def select(self, c_param) -> "Node":
        """
        Perform UCB exploration algorithm
        Parameters
        ----------
        c_param: temperature parameter
        Returns
        -------
        """
        return max(self.children.values(), key=lambda node: node._get_value(c_param))

    def update(self, leaf_value: float):
        self.n_visits += 1
        self.Q += + 1.0 * (leaf_value - self.Q) / self.n_visits

    @staticmethod
    def back_propagate(node: "Node", leaf_value: float):
        while node.parent:
            node.update(leaf_value)
            node = node.parent
            leaf_value = -leaf_value

    def _get_value(self, c_param):
        explore = (c_param * self.prior_prob *
                   np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + explore

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

    def __repr__(self):
        return f"""Node(position: {idx_to_position(self.position_idx)}, value: {self.Q}, visits: {self.n_visits})"""


class PureMCTSAgent:
    def __init__(
            self,
            env,
            player_id=1,
            policy_value_func=random_policy_value_fn,
            c_param=5,
            n_playout=10000
    ):
        self.env = copy.deepcopy(env)
        self.player_id = player_id
        self.root = Node(None, 1.)
        self.policy = policy_value_func
        self.c_param = c_param
        self.n_playout = n_playout

    def _playout(self, env: GomokuEnv):
        node = self.root
        while True:
            if node.is_leaf():
                break
            node = node.select(self.c_param)
            env.step(idx_to_position(node.position_idx))

        action_probs, _ = self.policy(env)
        if not env.is_ended():
            node.expand(action_probs)
        leaf_value = self._roll_out(env)
        Node.back_propagate(node, -leaf_value)  # current player is the child

    def _roll_out(self, env: GomokuEnv):
        player_id = env.current_player
        terminated = env.is_ended()
        while not terminated:
            action_probs = rollout_policy_fn(env)
            action_probs = list(action_probs)
            max_action = max(action_probs, key=itemgetter(1))[0]
            env.step(idx_to_position(ensure_idx(max_action)))

            terminated = env.is_ended()
        winner = env.winner
        if winner is None:
            return 0
        elif winner == player_id:
            return 1
        else:
            return -1

    def _take_action(self, env: GomokuEnv):
        for _ in range(self.n_playout):
            env.slow_save_checkpoints()
            self._playout(env)
            env.slow_load_checkpoints()

        return max(self.root.children.values(),
                   key=lambda node: node.n_visits)

    def take_action(self, last_move):
        env = self.env
        if last_move[0] == -1:
            middle_point = env.size // 2
            position = (middle_point, middle_point)
            self.root = Node(None, 1., ensure_idx(position))
            env.step(position)
            return position

        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            env.step(last_move)
        else:
            self.root = Node(None, 1., ensure_idx(last_move))
            env.step(last_move)

        legal_positions = env.adjacent_vacancies()

        if len(legal_positions) > 0:
            node = self._take_action(env)
            position = idx_to_position(node.position_idx)
            env.step(position)
            self.root = node
            return position
        else:
            print("WARNING: the board is full")


