class MinimaxAgent:
    def __init__(self, player_id, size=15, depth=3):
        self.player_id = player_id
        self.size = size
        self.depth = depth

    def select_move(self, board):
        best_score = float('-inf')
        best_move = None

        for row in range(self.size):
            for col in range(self.size):
                if board[row][col] == 0:
                    board[row][col] = self.player_id
                    score = self.minimax(board, self.depth, False, float('-inf'), float('inf'))
                    board[row][col] = 0  # Undo move
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)

        return best_move

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board)

        if maximizing_player:
            max_eval = float('-inf')
            for row in range(self.size):
                for col in range(self.size):
                    if board[row][col] == 0:
                        board[row][col] = self.player_id
                        eval = self.minimax(board, depth - 1, False, alpha, beta)
                        board[row][col] = 0  # Undo move
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(self.size):
                for col in range(self.size):
                    if board[row][col] == 0:
                        board[row][col] = 3 - self.player_id  # Opponent's move
                        eval = self.minimax(board, depth - 1, True, alpha, beta)
                        board[row][col] = 0  # Undo move
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board):
        # Implement a heuristic to evaluate board positions
        return 0

    def is_game_over(self, board):
        # Check if the game is over (win or draw)
        return False
