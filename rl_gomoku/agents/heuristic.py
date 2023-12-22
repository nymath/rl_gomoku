import numpy as np
import copy
from ..envs import GomokuEnv
import joblib
from joblib import Memory
from itertools import product
from numba import jit, njit
# memory = Memory(".cache")
from enum import Enum


class Pattern(Enum):
    Huo_4 = 0
    Chong_4 = 1
    Huo_3 = 2
    Mian_3 = 3
    Huo_2 = 4
    Mian_2 = 5

    NaT = 99


USE_SET = False


def position_to_idx(position, size=15):
    x, y = position
    return x * size + y


def idx_to_position(idx, size=15):
    return idx // size, idx % size


def is_in(list1: list, list2: set):
    for i in list1:
        if i not in list2:
            return False
    return True


def ensure_tuple(x, size=15):
    if isinstance(x, tuple):
        return x
    else:
        return x // size, x % size


def sublist(list1: list, list2: list):
    return set(list2).difference(list1).pop()


class HeuristicAgent:
    def __init__(self, env, player_id, random_state=42, cache_data=None):
        self.env: GomokuEnv = env
        self.random_state: int = random_state
        self.rng = np.random.default_rng(random_state)
        self.player_id = player_id
        self.cache_data = cache_data

        if hasattr(self.env, "size"):
            self.size = self.env.size
        elif hasattr(self.env, "height"):
            self.size = self.env.height

        self.quick_list_5 = cache_data["quick_list_5"]
        self.quick_list_4 = cache_data["quick_list_4"]
        self.quick_list_3 = cache_data["quick_list_3"]
        self.quick_list_2 = cache_data["quick_list_2"]
        self.quick_set_5 = ...
        self.quick_set_4 = ...
        self.quick_set_3 = ...
        self.quick_set_2 = ...
        # self.quick_set_5 = cache_data["quick_set_5"]
        # self.quick_set_4 = cache_data["quick_set_4"]
        # self.quick_set_3 = cache_data["quick_set_3"]
        # self.quick_set_2 = cache_data["quick_set_2"]
        self.current_black_list = []
        self.current_white_list = []

    def select_move(self):
        if self.env.episode_len == 1:
            m, n = self.env.last_move
            row_range = [m - 1, m, m + 1]
            col_range = [n - 1, n, n + 1]
        else:
            row_range = range(self.size)
            col_range = range(self.size)

        feasible_positions = list(product(row_range, col_range))

        eval_list = []

        where_black = np.where(self.env.board == 1)
        where_white = np.where(self.env.board == 2)
        self.current_black_list = (where_black[0] * self.size + where_black[1]).tolist()
        self.current_white_list = (where_white[0] * self.size + where_white[1]).tolist()

        # self.current_black_list = list(map(tuple, (np.argwhere(self.env.board == 1)).tolist()))
        # self.current_white_list = list(map(tuple, (np.argwhere(self.env.board == 2)).tolist()))
        # def p_eval(current_black_list, current_white_list, board, row, col):
        #     if self.env.is_valid_move(row, col):
        #         # 假设在这个位置下棋
        #         board[row, col] = self.player_id
        #         if self.player_id == 1:  # black
        #             current_black_list.append((row, col))
        #         else:
        #             current_white_list.append((row, col))
        #         score = self.evaluate_position(current_black_list, current_black_list, (row, col))
        #         eval_list.append([(row, col), score])
        #         return [(row, col), score]
        #
        # works_to_do = [joblib.delayed(p_eval(
        #     current_black_list=self.current_black_list.copy(),
        #     current_white_list=self.current_white_list.copy(),
        #     board=self.env.board.copy(),
        #     row=row,
        #     col=col,)) for (row, col) in feasible_positions]
        # eval_list = joblib.Parallel(n_jobs=6)(works_to_do)

        for row, col in feasible_positions:
            if self.env.is_valid_move(row, col):
                # 假设在这个位置下棋
                position_idx = position_to_idx((row, col), self.size)
                self.env.board[row, col] = self.player_id

                if self.player_id == 1:  # black
                    self.current_black_list.append(position_idx)
                else:
                    self.current_white_list.append(position_idx)

                score = self.evaluate_position(self.env.board, self.current_black_list, self.current_white_list, position_idx)
                eval_list.append([position_idx, score])

                if self.player_id == 1:
                    self.current_black_list.pop()
                else:
                    self.current_white_list.pop()
                self.env.board[row, col] = 0

        eval_list = sorted(eval_list, key=lambda x: x[1], reverse=True)
        max_score = eval_list[0][1]
        best_positions = []
        for position_idx, score in eval_list:
            if score < max_score:
                break
            best_positions.append(position_idx)
        self.best_positions = best_positions
        
        return idx_to_position(self.rng.choice(best_positions), self.size)

    def evaluate_position(self, board, current_black_list, current_white_list, position_idx):
        # 此函数判断在某一个点落子的评分
        score = 0
        if self.is_5(position_idx, self.player_id):
            score += 10000000

        # 假设对手下这一手棋, 是否5, 自己下这一手棋等价于阻碍对手的行为
        if self.is_5(position_idx, 3 - self.player_id):
            score += 1000000

        if self.player_id == 1:
            my_patterns = self.evaluation_position_player(board, current_black_list, position_idx)
            op_patterns = self.evaluation_position_player(board, current_white_list, position_idx)
        else:
            my_patterns = self.evaluation_position_player(board, current_white_list, position_idx)
            op_patterns = self.evaluation_position_player(board, current_black_list, position_idx)

        if my_patterns[Pattern.Huo_4] >= 1 or my_patterns[Pattern.Chong_4] >= 2 or (
                my_patterns[Pattern.Chong_4] == 1 and my_patterns[Pattern.Huo_3] >= 1):
            score += 100000
        if op_patterns[Pattern.Huo_4] >= 1 or op_patterns[Pattern.Chong_4] >= 2 or (
                op_patterns[Pattern.Chong_4] == 1 and op_patterns[Pattern.Huo_3] >= 1):
            score += 10000
        if my_patterns[Pattern.Huo_3] >= 2:
            score += 1000
        if op_patterns[Pattern.Huo_3] >= 2:
            score += 100
        score += ((my_patterns[Pattern.Chong_4] + my_patterns[Pattern.Huo_3]) * 4 +
                  (my_patterns[Pattern.Mian_3] + my_patterns[Pattern.Huo_2]) * 2 + my_patterns[Pattern.Mian_2] +
                  (op_patterns[Pattern.Chong_4] + op_patterns[Pattern.Huo_3]) * 4 +
                  (op_patterns[Pattern.Mian_3] + op_patterns[Pattern.Huo_2]) * 2 + op_patterns[Pattern.Mian_2])
        return score
        # 对一个位置的评分: 若落子后自己可以形成一个冲四或活三, 或者阻碍对手在此处落子形成一个冲四或活三, 加4分; 眠三或活二, 加2分; 眠二加1分.

    def evaluation_position_player(self, board, positions_idx, position_idx):
        # 此函数对在一个点落子后, 所形成的所有棋型进行统计(成五除外), 返回的是一个字典, 关键字为棋型名, 关联值为形成的数量.
        next_state = {*positions_idx, position_idx}

        eval_dict = {Pattern.Huo_4: 0,
                     Pattern.Chong_4: 0,
                     Pattern.Huo_3: 0,
                     Pattern.Mian_3: 0,
                     Pattern.Huo_2: 0,
                     Pattern.Mian_2: 0}

        for direction in range(1, 5):
            stats = []
            for pattern in self.get_positions_4(position_idx, direction):
                if is_in(pattern, next_state):  # check whether the pattern is allowed
                    result = self.type_4(board, pattern)
                    if result != Pattern.NaT:
                        stats.append(result)

            for pattern in self.get_positions_3(position_idx, direction):
                if is_in(pattern, next_state):  # check whether the pattern is allowed
                    result = self.type_3(board, pattern)
                    if result != Pattern.NaT:
                        stats.append(result)

            for pattern in self.get_positions_2(position_idx, direction):
                if is_in(pattern, next_state):  # check whether the pattern is allowed
                    result = self.type_2(board, pattern)
                    if result != Pattern.NaT:
                        stats.append(result)

            if not stats:
                pass
            else:
                if stats[0] == Pattern.Chong_4 and stats != [Pattern.Chong_4] and stats[1] == Pattern.Chong_4:
                    eval_dict[Pattern.Chong_4] += 2
                else:
                    eval_dict[stats[0]] += 1
        return eval_dict

    def type_4(self, board, pattern):
        num = 0
        for j in self.get_positions_5(pattern[0]):
            cond1 = is_in(pattern, j)
            if cond1:  # whether the pattern is matched
                x, y = idx_to_position(sublist(pattern, j), self.size)
                cond2 = board[x, y] == 0
                if cond2:  # whether the diff is empty
                    num += 1
        if num == 1:
            return Pattern.Chong_4  # opponent on the side
        elif num == 2:
            return Pattern.Huo_4  # no opponent
        else:
            return Pattern.NaT

    def type_3(self, board, pattern):
        is_mian3 = False
        for j in self.get_positions_4(pattern[0]):
            cond1 = is_in(pattern, j)  # whether the pattern is matched
            if cond1:
                x, y = idx_to_position(sublist(pattern, j), self.size)
                cond2 = board[x, y] == 0  # whether the diff is empty
                if cond2:
                    result = self.type_4(board, j)
                    if result == Pattern.Huo_4:  # Huo4 is induced by Huo3
                        return Pattern.Huo_3
                    elif result == Pattern.Chong_4:
                        is_mian3 = True
        if is_mian3:
            return Pattern.Mian_3

        return Pattern.NaT

    def type_2(self, board, pattern):
        is_mian2 = False
        for j in self.get_positions_3(pattern[0]):
            cond1 = is_in(pattern, j)
            if cond1:
                x, y = idx_to_position(sublist(pattern, j), self.size)
                cond2 = board[x, y] == 0
                if cond2:
                    result = self.type_3(board, j)
                    if result == Pattern.Huo_3:
                        return Pattern.Huo_2
                    elif result == Pattern.Mian_3:
                        is_mian2 = True
        if is_mian2:
            return Pattern.Mian_2

        return Pattern.NaT

    # 以上定义的函数判断的都是连续几个棋子的性质, 下面考虑一个点的性质, 以便AI落子时可以根据该点的性质来判断是否要把棋子落在该点
    # 分为两步: 首先判断在该点落子有无杀招.
    #   一步杀招: 该点落子后直接成五.
    #   两步杀招: 该点落子后至少形成一个活四, 或两个冲四, 或一个冲四加一个活三.
    #   三步杀招: 该点落子后形成两个活三.
    # 若没有杀招, 则根据在落子后形成的以上各种棋型的数量和阻碍对方形成以上各种棋型的数量进行打分.
    # 首先我们要定义以下函数, 计算出某一方在某一点落子后, 形成的各种棋型的数量
    # 在定义该函数前, 作出一点说明: 利用之前判断棋型的函数可能会造成失误.
    # 例如, 黑方现在有棋子(1,1), (1,4), (1,5). 在(1,2), (1,3), (1,6)处没有棋子.
    # 用之前的函数判断, 如果落子在(1,3), 会形成一个冲四[(1,1),(1,3),(1,4),(1,5)]和一个活三[(1,3),(1,4),(1,5)].
    # 但显然, 这样的"冲四加活三"与"两步杀招"中的不同.
    # 因此这种情况我们只计算冲四, 不计算活三, 究其原因是因为这两种棋型在同一直线上, 这也是我在list_5函数中区分方向的原因.
    # 但又有一种情况: 如现有(1,1),(1,3),(1,5),(1,7)四个棋子, 若落子到(1,4), 则可以在一条直线上形成两个冲四, 且也为两步杀招.
    # 所以我们下面计算时在一条直线上除了出现双冲四之外, 其他情况均只计算一种棋型.
    # 当然, 即使这样也是不准确的, 我们在此不作过于深入的讨论.

    def take_action(self, state):
        # 选择下一步走法
        x, y = ensure_tuple(state, size=self.size)
        # x, y = state
        if x == -1:
            return self.size // 2, self.size // 2
        else:
            return self.select_move()

    def get_action(self, env, *args, **kwargs):
        if env.last_move == -1:
            result = (self.size // 2, self.size // 2)
            probs = np.ones(self.size * self.size) / self.size * self.size
            return position_to_idx(result, size=self.size), probs
        else:
            result = self.select_move()
            probs = np.zeros(self.size * self.size)
            probs[self.best_positions] = 1 / len(self.best_positions)
        return position_to_idx(result, size=self.size), probs

    def get_positions_5(self, position_idx, direction=0):
        # 此函数的第一个参数为棋盘上的一个位置, 返回的是包含该位置的所有五子连珠的表. 下面list_4到list_2类似, 返回含有该位置的所有"四""三""二"的表.
        # x在此表示方向, 1为横向, 2为纵向, 3为左上到右下斜向, 4为左下到右上斜向. 0为默认值, 表示所有方向. 下同.
        if USE_SET:
            return self.quick_set_5[position_idx][direction]
        else:
            return self.quick_list_5[position_idx][direction]

    def get_positions_4(self, position_idx, direction=0):
        if USE_SET:
            return self.quick_set_4[position_idx][direction]
        else:
            return self.quick_list_4[position_idx][direction]

    def get_positions_3(self, position_idx, direction=0):
        if USE_SET:
            return self.quick_set_3[position_idx][direction]
        else:
            return self.quick_list_3[position_idx][direction]

    def get_positions_2(self, position_idx, direction=0):
        if USE_SET:
            return self.quick_set_2[position_idx][direction]
        else:
            return self.quick_list_2[position_idx][direction]

    def is_5(self, position_idx, player_id):
        if player_id == 1:
            lst = self.current_black_list.copy()
        else:
            lst = self.current_white_list.copy()
        _ = lambda x,y: position_to_idx((x, y), size=self.size)
        m, n = idx_to_position(position_idx, self.size)
        lst.append(position_idx)
        # 这一步在玩家对战判定胜负时多余, 但在人机对战的电脑算法中有用
        for i in range(m - 4, m + 1):
            if i < 0 or i + 4 >= self.size:
                continue
                # 处理棋盘边界特殊情况, 下同
            if _(i, n) in lst and _(i + 1, n) in lst and _(i + 2, n) in lst and _(i + 3, n) in lst and _(i + 4, n) in lst:
                return True
            # 纵向五子连珠
        for i in range(n - 4, n + 1):
            if i < 0 or i + 4 >= self.size:
                continue
            if _(m, i) in lst and _(m, i + 1) in lst and _(m, i + 2) in lst and _(m, i + 3) in lst and _(m, i + 4) in lst:
                return True
            # 横向五子连珠
        for i in range(5):
            if m - i < 0 or m - i + 4 >= self.size or n - i < 0 or n - i + 4 >= self.size:
                continue
            if _(m - i, n - i) in lst and _(m - i + 1, n - i + 1) in lst and _(m - i + 2, n - i + 2) in lst and _(
                    m - i + 3, n - i + 3) in lst and _(m - i + 4, n - i + 4) in lst:
                return True
        for i in range(5):
            if m - i < 0 or m - i + 4 >= self.size or n + i - 4 < 0 or n + i >= self.size:
                continue
            if _(m - i, n + i) in lst and _(m - i + 1, n + i - 1) in lst and _(m - i + 2, n + i - 2) in lst and _(
                    m - i + 3, n + i - 3) in lst and _(m - i + 4, n + i - 4) in lst:
                return True
            # 斜向五子连珠
        return False

