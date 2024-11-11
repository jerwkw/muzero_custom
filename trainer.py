import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from envs.tictactoe import TicTacToeEnv
from models.mcts import MCTSNode, mcts_search
from models.muzero import MuZeroNetwork
import numpy as np

class MuZeroTrainer:
    def __init__(self, network, env, num_simulations=50):
        self.network = network
        self.env = env
        self.num_simulations = num_simulations
        self.optimizer = optim.Adam(
            list(network.representation.parameters()) + 
            list(network.prediction.parameters()) + 
            list(network.dynamics.parameters()),
            lr=0.001
        )
        self.replay_buffer = deque(maxlen=1000)  # Replay buffer to store games

    def play_game(self):
        """Plays one game of Tic-Tac-Toe and stores (state, policy, value) in replay buffer."""
        env = TicTacToeEnv()
        state = env.reset()  # Initial game state
        done = False
        states, policies, values = [], [], []

        while not done:
            # Initialize the root node with the current game state
            root = MCTSNode(state=torch.tensor(state, dtype=torch.float32).view(1, 3, 3))
            
            # Run MCTS from the root node to get the best action
            mcts_search(root, self.network, self.num_simulations)

            # Get the visit counts of actions and normalize them to get the policy
            if root.children:
                visit_counts = np.array([child.visit_count for child in root.children.values()])
                policy = visit_counts / visit_counts.sum()  # Normalized visit counts as policy
                action = random.choices(list(root.children.keys()), weights=policy)[0]
            else:
                # If no valid children, choose a random legal action
                legal_actions = env.get_legal_actions()
                action = random.choice(legal_actions)
                policy = np.zeros(9, dtype=float)  # Define the policy as an empty numpy array

            states.append(state)
            policies.append(policy)

            # Apply the selected action to the environment
            reward, done = env.make_move(action)
            state = env.get_state()
            values.append(reward)

        # Store the episode in the replay buffer
        self.replay_buffer.append((states, policies, values))

    def sample_batch(self, batch_size=32):
        """Samples a batch of (state, policy, value) from the replay buffer."""
        batch = random.sample(self.replay_buffer, batch_size)
        states, policies, values = [], [], []
        for game in batch:
            states.extend(game[0])
            policies.extend(game[1])
            values.extend(game[2])
        return torch.tensor(np.array(states), dtype=torch.float32), torch.tensor(np.array(policies), dtype=torch.float32), torch.tensor(np.array(values), dtype=torch.float32)

    def train_step(self, batch_size=32):
        """Trains the network on a single batch from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return

        states, policies, values = self.sample_batch(batch_size)
        
        # Forward pass through the networks
        hidden_states = self.network.representation(states)
        pred_policies, pred_values = self.network.prediction(hidden_states)

        # Calculate the loss
        value_loss = F.mse_loss(pred_values.squeeze(-1), values)
        policy_loss = F.cross_entropy(pred_policies, policies.argmax(dim=1))
        loss = value_loss + policy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

if __name__ == "__main__":
    # Initialize environment and MuZero network
    env = TicTacToeEnv()
    network = MuZeroNetwork()
    trainer = MuZeroTrainer(network, env, num_simulations=50)

    # Training parameters
    num_games = 1000
    batch_size = 32

    # Run the training loop
    for game in range(num_games):
        # Step 1: Play a game and store it in the replay buffer
        trainer.play_game()
        
        # Step 2: Train the model from the replay buffer
        trainer.train_step(batch_size=batch_size)

        # Optional: Log the progress
        if game % 50 == 0:
            print(f"Game {game}: Training in progress...")

    print("Training completed!")
    network.save_model("muzero_tictactoe_model.pth")
