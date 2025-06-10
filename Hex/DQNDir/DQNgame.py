import numpy as np
import matplotlib.pyplot as plt


class Hex_Game:
    def __init__(self, n):
        self.BOARD_SIZE = n
        self.board = np.zeros((n, n), dtype=int)

        self.p1Board = np.zeros((n, n), dtype=int)
        self.p2Board = np.zeros((n, n), dtype=int)

        self.current_player = 1
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        self.actionspace = set((i, j) for i in range(n) for j in range(n))
        self.posMoves = n * n

        # Swap‐rule flags
        self.swap = True
        self.first_turn = True
        self.first_move_pos = None  # will store Player 1's first move

        # —– Union‐Find setup (two virtual nodes per player) —–
        size_uf = n * n + 2
        self.p1_parent = list(range(size_uf))
        self.p1_rank = [0] * size_uf
        self.p2_parent = list(range(size_uf))
        self.p2_rank = [0] * size_uf

        # Virtual nodes:
        self.P1_TOP = n * n
        self.P1_BOTTOM = n * n + 1
        self.P2_LEFT = n * n
        self.P2_RIGHT = n * n + 1

    def reset(self):
        """Reset the game to a brand‐new empty board of the same size."""
        self.__init__(self.BOARD_SIZE)

    # —– Union‐Find helpers —–
    def _find(self, parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, parent, rank, x, y):
        rx = self._find(parent, x)
        ry = self._find(parent, y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    def _undo_p1_first_move(self):
        """Resets the board state and union-find data for Player 1's first move."""
        orig_i, orig_j = self.first_move_pos

        # Reset board state
        self.board[orig_i, orig_j] = 0
        self.p1Board[orig_i, orig_j] = 0

        # Reset union-find data for the specific cell and connected virtual nodes.
        # This simple reset is sufficient because it was the first move with no same-color neighbors.
        orig_idx = orig_i * self.BOARD_SIZE + orig_j
        self.p1_parent[orig_idx] = orig_idx
        self.p1_rank[orig_idx] = 0
        if orig_i == 0:
            self.p1_parent[self.P1_TOP] = self.P1_TOP
            self.p1_rank[self.P1_TOP] = 0
        if orig_i == self.BOARD_SIZE - 1:
            self.p1_parent[self.P1_BOTTOM] = self.P1_BOTTOM
            self.p1_rank[self.P1_BOTTOM] = 0

    def _perform_union(self, pos, player):
        """Helper method to perform union-find operations for a given move."""
        i, j = pos
        idx = i * self.BOARD_SIZE + j

        parent, rank, neighbor_val = (self.p1_parent, self.p1_rank, 1) if player == 1 else (self.p2_parent, self.p2_rank, 2)

        # Connect to virtual nodes (edges of the board)
        if player == 1:
            if i == 0: self._union(parent, rank, idx, self.P1_TOP)
            if i == self.BOARD_SIZE - 1: self._union(parent, rank, idx, self.P1_BOTTOM)
        else: # player == 2
            if j == 0: self._union(parent, rank, idx, self.P2_LEFT)
            if j == self.BOARD_SIZE - 1: self._union(parent, rank, idx, self.P2_RIGHT)

        # Connect to adjacent neighbors of the same color
        for dx, dy in self.directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.BOARD_SIZE and 0 <= nj < self.BOARD_SIZE and self.board[ni, nj] == neighbor_val:
                self._union(parent, rank, idx, ni * self.BOARD_SIZE + nj)

    def placeMove(self, pos, player, is_swap_action=False):
        """
        Places a stone for a player, handling the special cases for the first two moves
        (including the custom swap rule) and all subsequent moves.
        """
        
        # --- Handle Player 1's First Move ---
        if player == 1 and self.swap and self.first_turn:
            i,j = pos
            self.board[i, j] = 1
            self.p1Board[i, j] = 1
            self._perform_union(pos, 1)
            self.first_move_pos = pos
            self.first_turn = False
            self.actionspace.remove(pos)
            return

        # --- Handle Player 2's First Move (Swap Decision) ---
        if player == 2 and self.swap and not self.first_turn:
            self.swap = False  # Swap opportunity is now used/declined

            # Case A: P2 swaps using the dedicated swap action
            if is_swap_action:
                swapped_pos = (self.first_move_pos[1], self.first_move_pos[0])
                self._undo_p1_first_move() # Call the new helper method
                self.actionspace.add(self.first_move_pos)
                # Place P2's piece at the new swapped position
                self.board[swapped_pos[0], swapped_pos[1]] = 2
                self.p2Board[swapped_pos[0], swapped_pos[1]] = 1
                self.actionspace.remove(swapped_pos)    
                self.posMoves -= 1
                self._perform_union(swapped_pos, 2)
                return

            # Case B: P2 declines to swap by placing a piece elsewhere

        
        # --- Generic move placement for all other turns ---
        if pos not in self.actionspace:
            return # Ignore illegal move
            
        i, j = pos
        self.board[i, j] = player
        if player == 1: self.p1Board[i, j] = 1
        else: self.p2Board[i, j] = 1
        
        self.actionspace.remove(pos)
        self.posMoves -= 1
        self._perform_union(pos, player)

    def checkWin(self, player):
        """
        Return True if `player` has connected their two opposite edges:
          • Player 1: top edge ↔ bottom edge
          • Player 2: left edge ↔ right edge
        This is now an O(α(n)) union-find check.
        """
        if player == 1:
            return (
                self._find(self.p1_parent, self.P1_TOP)
                == self._find(self.p1_parent, self.P1_BOTTOM)
            )
        else:
            return (
                self._find(self.p2_parent, self.P2_LEFT)
                == self._find(self.p2_parent, self.P2_RIGHT)
            )

    # ——— Drawing routines (unchanged) ———
    def hex_corner(self, center_x, center_y, size, i):
        angle_deg = 60 * i - 30
        angle_rad = np.radians(angle_deg)
        return (
            center_x + size * np.cos(angle_rad),
            center_y + size * np.sin(angle_rad),
        )

    def draw_hex(self, ax, x, y, size, **kwargs):
        corners = [self.hex_corner(x, y, size, i) for i in range(6)]
        hexagon = plt.Polygon(corners, closed=True, **kwargs)
        ax.add_patch(hexagon)

    def draw_board(self, fig=None, ax=None, size=1):
        """Draws the board on a given figure/axis, or creates a new one."""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.ion()  # Turn on interactive mode
            fig.show()

        ax.clear()  # Clear the axis for the new drawing
        ax.set_aspect("equal")
        ax.set_title("Hex Game")

        width, height = np.sqrt(3) * size, 2 * size
        vert_spacing, horiz_spacing = 0.75 * height, width

        for q in range(self.BOARD_SIZE):
            for r in range(self.BOARD_SIZE):
                x, y = (q + r / 2) * horiz_spacing, -r * vert_spacing
                hexColor = {0: "white", 1: "blue", 2: "red"}.get(self.board[r, q])
                self.draw_hex(ax, x, y, size, edgecolor="black", facecolor=hexColor)

        ax.autoscale_view()
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        return fig, ax


if __name__ == "__main__":
    hg = Hex_Game(4)

    # 1) P1 plays at (1,1). first_turn → False. This position stays in actionspace.
    hg.placeMove((1, 1), 1)
    print("After P1(1,1):", 
          "board[1,1]=", hg.board[1,1], 
          "first_turn=", hg.first_turn, 
          "swap=", hg.swap,
          "(1,1) in actionspace?", ( (1,1) in hg.actionspace ), 
          "posMoves=", hg.posMoves)

    # 2a) If P2 now swaps on (1,1):
    hg2 = Hex_Game(4)
    hg2.placeMove((1, 1), 1)
    hg2.placeMove((1, 1), 2)  # swap
    print("\nAfter P2 swaps on (1,1):",
          "board[1,1]=", hg2.board[1,1],
          "p1Board[1,1]=", hg2.p1Board[1,1],
          "p2Board[1,1]=", hg2.p2Board[1,1],
          "swap=", hg2.swap,
          "(1,1) in actionspace?", ((1,1) in hg2.actionspace),
          "posMoves=", hg2.posMoves)

    # 2b) If instead P2 plays somewhere else, e.g. (0,0):
    hg3 = Hex_Game(4)
    hg3.placeMove((1, 1), 1)
    hg3.placeMove((0, 0), 2)  # P2 declines swap
    print("\nAfter P2 plays (0,0) (no swap):",
          "board[1,1] still=", hg3.board[1,1],
          "(1,1) in actionspace?", ((1,1) in hg3.actionspace),
          "swap=", hg3.swap,
          "posMoves=", hg3.posMoves)

    # 3) Continuing randomly until the board fills; no unexpected IndexError from actionspace.
    import random
    hg_full = Hex_Game(3)
    hg_full.placeMove((0, 0), 1)   # first move
    player = 2
    while hg_full.actionspace:
        mv = random.choice(hg_full.actionspace)
        hg_full.placeMove(mv, player)
        player = 1 if player == 2 else 2
    print("\nFinal posMoves (should be 0):", hg_full.posMoves,
          "actionspace length (should be 0):", len(hg_full.actionspace))



    hg = Hex_Game(5)

    hg.placeMove((1, 0), 1)
    hg.placeMove((1, 0), 2)
    print("miav", hg.actionspace)
    print(hg.checkWin(1))  # False
    hg.placeMove((1, 1), 2)
    hg.placeMove((1, 2), 2)
    hg.placeMove((1, 3), 2)
    hg.placeMove((1, 4), 2)

    hg.placeMove((0, 0), 1)
    hg.placeMove((2, 0), 1)
    hg.placeMove((3, 0), 1)
    hg.placeMove((4, 0), 1)
    print(hg.checkWin(2))  # False
    print(hg.checkWin(1))  # True


    hg.draw_board(size=1, dims=hg.BOARD_SIZE)