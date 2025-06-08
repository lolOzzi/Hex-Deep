from DQNgame import *
import random
from HexEnv import *
from DQN import *
import random

class HexEnv:
    SIZE = 5
    WIN_REWARD = 5
    LOSS_PENALTY = -5
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = SIZE*SIZE
    
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
        self.hex.placeMove(action, player)
        if self.hex.checkWin(player):
            return self.getObservation(player), self.WIN_REWARD if player == self.player_num else self.LOSS_PENALTY, True
        return self.getObservation(player), self.turn_penalty, False

    def getObservation(self, player):
        swap_channel = np.zeros((self.SIZE, self.SIZE))

        # The swap is only available for player 2, on their first move
        # self.hex.swap is True and self.hex.first_turn is False
        if player == 2 and self.hex.swap and not self.hex.first_turn:
            if self.hex.first_move_pos is not None:
                i, j = self.hex.first_move_pos
                swap_channel[i, j] = 1
        
        # Return the observation, transposed if player 2 for consistency.
        if player == 1:
            return np.stack([self.hex.p1Board, self.hex.p2Board, swap_channel], axis=-1)
        else:
            return np.stack([self.hex.p2Board.T, self.hex.p1Board.T, swap_channel.T], axis=-1)
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
        return random.choice(tuple(self.hex.actionspace))
    
    def render(self):
        self.fig, self.ax = self.hex.draw_board(self.fig, self.ax)
    
    def close(self):
        """Closes the matplotlib window."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None