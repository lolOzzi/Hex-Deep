import time
import numpy as np
import tensorflow as tf
from HexEnv import HexEnv
from DQN import DQNAgent

# --- Configuration ---
# 1. Provide the paths to your two trained models.
MODEL_1_PATH = "models/5x5-jupiter_____5.00max____2.70avg____0.00min__1749753604.keras"  # Player 1 Model
MODEL_2_PATH = "models/5x5-jupiter_____5.00max____2.70avg____0.00min__1749753604.keras" # Player 2 Model

# 2. Set to True to see the graphical representation of the board after each move.
SHOW_BOARD_VISUALIZATION = True

# --- Helper Function for Human Input ---

def get_human_move(env: HexEnv) -> tuple[int, int]:
    """
    Prompts the human player for a move and validates it.

    Args:
        env (HexEnv): The game environment instance.

    Returns:
        tuple[int, int]: The validated (row, col) move.
    """
    while True:
        try:
            move_str = input(f"Enter your move as 'row,col' (e.g., '2,3'): ")
            parts = move_str.replace(' ', '').split(',')
            if len(parts) != 2:
                raise ValueError("Invalid format. Please use 'row,col'.")
            
            row, col = int(parts[0]), int(parts[1])
            move = (row, col)

            # Check if the move is within the board's boundaries
            if not (0 <= row < env.SIZE and 0 <= col < env.SIZE):
                print("Error: Move is outside the board boundaries.")
                continue

            # Check if the chosen cell is already occupied
            if move not in env.hex.actionspace: #
                print("Error: That cell is already occupied.")
                continue
            
            return move

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# --- Game Mode Functions ---

def run_ai_vs_ai_match(model_path_1, model_path_2):
    """
    Loads two models and has them play a game of Hex against each other.
    """
    print("--- Starting AI vs. AI Match ---")
    try:
        env = HexEnv()
        print("Loading models...")
        agent1 = DQNAgent(env, train=False) #
        agent1.model.load_weights(model_path_1)
        print(f"Player 1 model loaded from: {model_path_1}")

        agent2 = DQNAgent(env, train=False) #
        agent2.model.load_weights(model_path_2)
        print(f"Player 2 model loaded from: {model_path_2}")
        
        agents = {1: agent1, 2: agent2}
    except (IOError, ValueError) as e:
        print(f"Error loading models: {e}")
        return

    current_state = env.reset() #
    done = False
    if SHOW_BOARD_VISUALIZATION: env.render()
        
    while not done:
        player_num = env.player_num #
        current_agent = agents[player_num]

        all_q_values = current_agent.get_qs(current_state) #
        if (player_num==1):
            valid_flat_actions = [i * env.SIZE + j for (i, j) in env.hex.actionspace]
        else:
            valid_flat_actions = [i * env.SIZE + j for (j, i) in env.hex.actionspace] #
        valid_q_values = {action: all_q_values[action] for action in valid_flat_actions}
        action = max(valid_q_values, key=valid_q_values.get)
        
        row, col = divmod(action, env.SIZE)
        print("debug: ", row, col)
        move = (col, row) if player_num == 2 else (row, col) #

        print(f"AI Player {player_num} chooses move: {move}")
        _, _, done = env.step(action, player_num) #

        if SHOW_BOARD_VISUALIZATION: env.render()
        time.sleep(1)

        if done:
            print(f"\n--- Game Over! Player {player_num} wins! ---")
            break

        next_player_num = 3 - player_num
        current_state = env.getObservation(next_player_num) #
        env.player_num = next_player_num #


def run_human_vs_ai_match(ai_model_path, human_player_num):
    """
    Loads one AI model and lets a human player challenge it.
    """
    print("--- Starting Human vs. AI Match ---")
    try:
        env = HexEnv()
        print("Loading AI model...")
        ai_agent = DQNAgent(env, train=False) #
        ai_agent.model.load_weights(ai_model_path)
        print(f"AI model loaded from: {ai_model_path}")
    except (IOError, ValueError) as e:
        print(f"Error loading model: {e}")
        return

    ai_player_num = 3 - human_player_num
    current_state = env.reset() #
    done = False
    if SHOW_BOARD_VISUALIZATION: env.render()

    while not done:
               # ... inside run_human_vs_ai_match ...
        player_num = env.player_num 
        action = None # Use a single 'action' variable

        if player_num == human_player_num:
            print(f"\nYour turn (Player {player_num}).")
            move_tuple = get_human_move(env)
            # Convert the human's (row, col) tuple to a flat integer action
            action = move_tuple[0] * env.SIZE + move_tuple[1]
        else:
            print(f"\nAI's turn (Player {ai_player_num})...")
            all_q_values = ai_agent.get_qs(current_state) 
            
            # Get valid actions based on the current player
            if player_num == 1:
                valid_flat_actions = {i * env.SIZE + j for (i, j) in env.hex.actionspace}
            else: # player == 2
                valid_flat_actions = {j * env.SIZE + i for (i, j) in env.hex.actionspace}

            valid_q_values = {a: all_q_values[a] for a in valid_flat_actions}
            action = max(valid_q_values, key=valid_q_values.get)
            
            # For display purposes, convert AI action back to a tuple
            row, col = divmod(action, env.SIZE)
            move_display = (col, row) if player_num == 2 else (row, col)
            print(f"AI (Player {ai_player_num}) chooses move: {move_display}")
            time.sleep(1)

        # Always call env.step with the integer 'action'
        _, _, done = env.step(action, player_num)

        if SHOW_BOARD_VISUALIZATION: env.render()

        if done:
            winner = "You" if player_num == human_player_num else "The AI"
            print(f"\n--- Game Over! {winner} (Player {player_num}) won! ---")
            break

        next_player_num = 3 - player_num
        current_state = env.getObservation(next_player_num) #
        env.player_num = next_player_num #

# --- Main Menu ---

def main():
    """
    Displays the main menu to select the game mode.
    """
    # Check if model paths are valid before starting
    if "path/to" in MODEL_1_PATH or "path/to" in MODEL_2_PATH:
        print("ERROR: Please update the MODEL_1_PATH and MODEL_2_PATH variables in the script before running.")
        return

    while True:
        print("\n--- Hex Game Menu ---")
        print("1. AI Model vs. AI Model")
        print("2. Human vs. AI Model")
        print("3. Exit")
        choice = input("Select an option (1, 2, or 3): ")

        if choice == '1':
            run_ai_vs_ai_match(MODEL_1_PATH, MODEL_2_PATH)
        elif choice == '2':
            while True:
                print("\nWhich AI model do you want to play against?")
                print(f"1. Model 1 ({MODEL_1_PATH.split('/')[-1]})")
                print(f"2. Model 2 ({MODEL_2_PATH.split('/')[-1]})")
                model_choice = input("Select a model (1 or 2): ")
                if model_choice in ['1', '2']:
                    ai_model_path = MODEL_1_PATH if model_choice == '1' else MODEL_2_PATH
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            
            while True:
                print("\nDo you want to be Player 1 (Blue, Top-to-Bottom) or Player 2 (Red, Left-to-Right)?")
                player_choice = input("Enter 1 or 2: ")
                if player_choice in ['1', '2']:
                    human_player_num = int(player_choice)
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            
            run_human_vs_ai_match(ai_model_path, human_player_num)
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()