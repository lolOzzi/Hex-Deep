import numpy as np;
import matplotlib.pyplot as plt



class Hex_Game:
    def __init__(self, n):
        self.BOARD_SIZE = n
        self.board = np.zeros((n, n), dtype=int)
        self.p1Board = np.zeros((n, n), dtype=int)
        self.p2Board = np.zeros((n, n), dtype=int)
        self.current_player = 1
        self.directions = [(1,0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        self.actionspace =[(i, j) for i in range(n) for j in range(n)]
        self.posMoves = n*n
        self.swap = True
        self.first_turn = True


    def reset(self):
        self.__init__(self.BOARD_SIZE)

    def placeMove(self, pos, player):
        self.board[pos] = player
        if player == 1:
             self.p1Board[pos] = 1
        else:
            if (self.p1Board[pos] == 1):
                self.p1Board[pos] = 0
            self.p2Board[pos] = 1
        if (player == 1 and self.swap and self.first_turn):
            self.first_turn = False
        else:
            self.actionspace.remove(pos)
            self.posMoves -= 1

    def hex_corner(self, center_x, center_y, size, i):
        angle_deg = 60 * i - 30
        angle_rad = np.radians(angle_deg)
        return (center_x + size * np.cos(angle_rad),
                center_y + size * np.sin(angle_rad))

    def draw_hex(self, ax, x, y, size, **kwargs):
        corners = [self.hex_corner(x, y, size, i) for i in range(6)]
        hexagon = plt.Polygon(corners, closed=True, **kwargs)
        ax.add_patch(hexagon)


    def hex_grid(self, size=1, dims=11):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        
        width = np.sqrt(3) * size
        height = 2 * size
        vert_spacing = 0.75 * height
        horiz_spacing = width 
        
        for q in range(dims):
            for r in range(dims):
                x = (q + r/2) * horiz_spacing
                y = -r * vert_spacing
                hexColor = 'blue' if self.board[q][r] == 1 else ('red' if self.board[q][r]==2 else 'white')
                self.draw_hex(ax, x, y, size, edgecolor='black', facecolor=hexColor)
        
        ax.autoscale_view()
        ax.axis('off')
        plt.tight_layout()
        plt.show()


    def checkWin(self, player):
        """Return True if `player` has a connecting path."""
        n = self.BOARD_SIZE
        visited = set()
        stack = []

        # initialize stack with all starting-edge cells for this player
        if player == 1:
            # Player 1: top edge (row=0), target bottom edge (row=n-1)
            for col in range(n):
                if self.board[0, col] == player:
                    stack.append((0, col))
                    visited.add((0, col))
            target_row = n - 1
            is_target = lambda x, y: x == target_row

        else:
            # Player 2: left edge (col=0), target right edge (col=n-1)
            for row in range(n):
                if self.board[row, 0] == player:
                    stack.append((row, 0))
                    visited.add((row, 0))
            target_col = n - 1
            is_target = lambda x, y: y == target_col

        # DFS
        while stack:
            x, y = stack.pop()
            if is_target(x, y):
                return True
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n \
                   and (nx, ny) not in visited \
                   and self.board[nx, ny] == player:
                    visited.add((nx, ny))
                    stack.append((nx, ny))

        return False

    # (Your drawing methods would go here unchanged…

hg = Hex_Game(5)
hg.placeMove((1,0), 2)
print(hg.checkWin(1))
hg.placeMove((1,1), 2)
hg.placeMove((1,2), 2)
hg.placeMove((1,3), 2)
hg.placeMove((1,4), 2)
print(hg.checkWin(2))
hg.placeMove((3,1), 1)

hg.hex_grid(size=1, dims=hg.BOARD_SIZE)









