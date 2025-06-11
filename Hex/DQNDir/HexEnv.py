from DQNgame import *
import random
import numpy as np
import matplotlib.pyplot as plt

class HexEnv:
    SIZE = 5
    WIN_REWARD = 5
    LOSS_PENALTY = -5
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # Shape of the conv input
    ACTION_SPACE_SIZE = SIZE*SIZE + 1
    SWAP_ACTION = SIZE*SIZE

    def __init__(self):
        self.hex = Hex_Game(self.SIZE)
        self.player_num = random.randint(1, 2)
        self.turn_penalty = -0.1
        self.fig = None
        self.ax = None
        # Pre-calculate the adjacency matrix for the GNN
        self.adj_matrix = self._create_adjacency_matrix()

    def _create_adjacency_matrix(self):
        """Creates a static adjacency matrix for the hex board."""
        n_nodes = self.SIZE * self.SIZE
        adj = np.zeros((n_nodes, n_nodes))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                idx1 = r * self.SIZE + c
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                        idx2 = nr * self.SIZE + nc
                        adj[idx1, idx2] = 1
        return adj

    def reset(self, render=False):
        if render:
            self.close()
        self.hex.reset()
        self.player_num = 1
        return self.getObservation(1)

    def step(self, action, player):
        is_swap = (action == self.SWAP_ACTION)
        is_swap_available = (player == 2 and self.hex.swap and not self.hex.first_turn)

        if is_swap:
            if is_swap_available:
                self.hex.placeMove(pos=None, player=player, is_swap_action=True)
            else:
                raise Exception("Illegal move: Swap not available")
        else:
            row, col = divmod(action, self.SIZE)
            move = (col, row) if player == 2 else (row, col)
            if move in self.hex.actionspace:
                self.hex.placeMove(pos=move, player=player)
            else:
                raise Exception(f"Illegal move: {move} not available")

        if self.hex.checkWin(player):
            reward = self.WIN_REWARD
            return self.getObservation(player), reward, True
        
        return self.getObservation(player), self.turn_penalty, False

    def getObservation(self, player):
        """
        Constructs the four-part observation required by the hybrid GNN-ConvNet model.
        Returns: [convolutional_input, node_features, adjacency_matrix, swap_flag]
        """
        # --- Part 1: Convolutional Input ---
        swap_channel = np.zeros((self.SIZE, self.SIZE))
        is_swap_available = player == 2 and self.hex.swap and not self.hex.first_turn
        
        if is_swap_available and self.hex.first_move_pos:
            i, j = self.hex.first_move_pos
            swap_channel[i, j] = 1.0

        if player == 1:
            conv_input = np.stack([self.hex.p1Board, self.hex.p2Board, swap_channel], axis=-1)
        else: # player == 2
            conv_input = np.stack([self.hex.p2Board.T, self.hex.p1Board.T, swap_channel.T], axis=-1)

        # --- Part 2: Node Feature Input ---
        node_features = conv_input.reshape(-1, conv_input.shape[-1])

        # --- Part 3: Adjacency Matrix ---
        adj_matrix = self.adj_matrix
        
        # --- Part 4: Swap Flag Input ---
        swap_flag = np.array([1.0 if is_swap_available else 0.0])

        return (conv_input.astype(np.float32), node_features.astype(np.float32), adj_matrix.astype(np.float32), swap_flag.astype(np.float32))
    def resetRand(self, render=False):
        if render:
            self.close()
        self.hex.reset()
        self.player_num = random.randint(1,2)
        if self.player_num == 1:
            return self.getObservation(self.player_num)
        else:
            self.hex.placeMove(self.calc_op_move(), 1)
            return self.getObservation(self.player_num)

    def calc_op_move(self):
        return random.choice(tuple(self.hex.actionspace))
    
    def render(self):
        self.fig, self.ax = self.hex.draw_board(self.fig, self.ax)
    
    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None