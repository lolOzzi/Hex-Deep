from DQNgame import *
import random
from HexEnv import *
from DQN import *
import random

class HexEnv:
    SIZE = 5
    WIN_REWARD = 5
    LOSS_PENALTY = -5
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 2)
    ACTION_SPACE_SIZE = SIZE*SIZE + 1
    SWAP_ACTION = SIZE*SIZE # The index for the swap action
    
    def __init__(self):
        self.hex = Hex_Game(self.SIZE)
        self.player_num = random.randint(1,2)
        self.turn_penalty = -0.1

        self.fig = None
        self.ax = None
    
    def reset(self, render=False):
        if render:
            self.close()
        self.hex.reset()
        self.player_num = 1
        return self.getObservation(1)

    def step(self, action, player):
        """
        Takes an integer action from the agent and applies it to the game.
        Action can be a board move (0-24) or a swap action (25).
        """
        is_swap = (action == self.SWAP_ACTION)
        is_swap_available = (player == 2 and self.hex.swap and not self.hex.first_turn)

        # Case 1: Agent chose to swap
        if is_swap:
            if is_swap_available:
                self.hex.placeMove(pos=None, player=player, is_swap_action=True)
            else:
                 raise Exception("Illegal move: Swap not available")
        
        # Case 2: Agent chose to place a piece
        else:
            row, col = divmod(action, self.SIZE)
            # Player 2's board is transposed, so we must un-transpose the move.
            move = (col, row) if player == 2 else (row, col)

            if move in self.hex.actionspace:
                self.hex.placeMove(pos=move, player=player)
            else:
                 raise Exception("Illegal move:", move, "not available" )

        # Check for a win after the move
        if self.hex.checkWin(player):
            reward = self.WIN_REWARD if player == self.player_num else self.LOSS_PENALTY
            return self.getObservation(player), reward, True
        
        # If no win, return the standard turn penalty
        return self.getObservation(player), self.turn_penalty, False

    def getObservation(self, player):
        swap_available = 1.0 if player == 2 and self.hex.swap and not self.hex.first_turn else 0.0
        swap_flag = np.array([swap_available])

        swap_channel = np.zeros((self.SIZE, self.SIZE))

        # The swap is only available for player 2, on their first move
        # self.hex.swap is True and self.hex.first_turn is False
        if swap_available:
            if self.hex.first_move_pos is not None:
                i, j = self.hex.first_move_pos
                swap_channel[i, j] = 1
        
        # Return the observation, transposed if player 2 for consistency.
        if player == 1:
            board_state = np.stack([self.hex.p1Board, self.hex.p2Board, swap_channel], axis=-1)
        else:
            board_state = np.stack([self.hex.p2Board.T, self.hex.p1Board.T, swap_channel.T], axis=-1)
        return (board_state, swap_flag)

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
    
    def stepRandWithResponse(self, action):
        self.hex.placeMove(action, self.player_num)
        if (self.hex.checkWin(self.player_num)):
            return (self.getObservation(self.player_num), self.WIN_REWARD, True)
        self.hex.placeMove(self.calc_op_move(), 1 if self.player_num==2 else 2 )
        if (self.hex.checkWin(1 if self.player_num==2 else 2)):
            return (self.getObservation(self.player_num), self.LOSS_PENALTY, True)
        return (self.getObservation(self.player_num), self.turn_penalty, False)

    def calc_op_move(self):
        return random.sample(list(self.hex.actionspace), 1)[0]
    
    def render(self):
        self.fig, self.ax = self.hex.draw_board(self.fig, self.ax)
    
    def close(self):
        """Closes the matplotlib window."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None