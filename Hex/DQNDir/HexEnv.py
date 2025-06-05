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
    ACTION_SPACE_SIZE = SIZE*SIZE
    
    def __init__(self):
        self.hex = Hex_Game(self.SIZE)
        self.player_num = random.randint(1,2)
        
    def reset(self):
        self.hex.reset()
        self.player_num = random.randint(1,2)
        if self.player_num == 1:
            return self.getObservation()
        else:
            self.hex.placeMove(self.calc_op_move(), 1)
            return self.getObservation()
    
    def step(self, action):
        self.hex.placeMove(action, self.player_num)
        if (self.hex.checkWin(self.player_num)):
            return (self.getObservation(), self.WIN_REWARD, True)
        self.hex.placeMove(self.calc_op_move(), 1 if self.player_num==2 else 2 )
        if (self.hex.checkWin(1 if self.player_num==2 else 2)):
            return (self.getObservation(), self.LOSS_PENALTY, True)
        return (self.getObservation(), -1, False)

    def calc_op_move(self):
        return random.choice(self.hex.actionspace)
    
    def getObservation(self):
        if self.player_num == 1:
            return np.stack([self.hex.p1Board, self.hex.p2Board], axis=-1)
        else:
            return np.stack([self.hex.p2Board, self.hex.p1Board], axis=-1)
    
    def render(self):
        self.hex.hex_grid(size=1, dims=self.hex.BOARD_SIZE)
    