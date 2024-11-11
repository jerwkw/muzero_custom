import numpy as np
import torch
import random

from envs.tictactoe import TicTacToeEnv
from models.mcts import MCTSNode, mcts_search
from models.muzero import MuZeroNetwork

def play_against_ai(network, num_simulations=50):
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    human_player = 1  # You can let the human decide to be 1 (X) or -1 (O)

    print("You are playing Tic-Tac-Toe against MuZero!")
    print_board(state)

    while not done:
        if env.current_player == human_player:
            # Human's turn
            print("Your move! Enter your move as 'row col' (0-indexed): ")
            row, col = map(int, input().split())
            action = (row, col)
            if action not in env.get_legal_actions():
                print("Invalid move. Try again.")
                continue
        else:
            # AI's turn: use MCTS to choose the best action
            action = ai_move(state, network, num_simulations, env)

        # Make the move and get the game state
        reward, done = env.make_move(action)
        state = env.get_state()
        print_board(state)

        if done:
            if reward == human_player:
                print("Congratulations! You won!")
            elif reward == -human_player:
                print("MuZero won! Better luck next time.")
            else:
                print("It's a draw!")
            break

def ai_move(state, network, num_simulations, env):
    # Initialize the root node for the current state
    root = MCTSNode()
    root.state = torch.tensor(state, dtype=torch.float32).view(1, 3, 3)

    # Run MCTS to find the best move
    mcts_search(root, network, num_simulations)

    # Choose the action with the highest visit count
    if root.children:
        action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return divmod(action, 3)  # Convert action index back to (row, col) format
    else:
        # If no valid children, choose a random legal action
        legal_actions = env.get_legal_actions()
        return random.choice(legal_actions)

def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    for i, row in enumerate(board):
        # Convert each cell in the row to the appropriate symbol
        row_symbols = [symbols[cell] for cell in row]
        
        # Print the row with separators between columns
        print(" | ".join(row_symbols))
        
        # Print row separator after each row, except the last
        if i < len(board) - 1:
            print("---------")

if __name__ == "__main__":
    network = MuZeroNetwork()
    network.load_model("muzero_tictactoe_model.pth")
    play_against_ai(network)
