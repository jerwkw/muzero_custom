import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(9, 64)
    
    def forward(self, x):
        x = F.relu(self.fc(x.view(-1, 9)))
        return x

class PredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_policy = nn.Linear(64, 9)  # 9 possible moves
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))  # Scaled value for win/loss
        return policy, value

class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = nn.Linear(64 + 9, 64)
        self.fc_reward = nn.Linear(64, 1)

    def forward(self, x, action):
        x = torch.cat([x, F.one_hot(action, 9).float()], dim=-1)
        new_state = F.relu(self.fc_state(x))
        reward = torch.tanh(self.fc_reward(new_state))
        return new_state, reward

class MuZeroNetwork:
    def __init__(self):
        self.representation = RepresentationNetwork()
        self.prediction = PredictionNetwork()
        self.dynamics = DynamicsNetwork()
    
    def initial_inference(self, state):
        hidden_state = self.representation(state.view(-1, 9))
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

    def recurrent_inference(self, hidden_state, action):
        new_hidden_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(new_hidden_state)
        return new_hidden_state, policy, value, reward
    
    def save_model(self, path="muzero_tictactoe_model.pth"):
        """Save the MuZero network model to a file."""
        torch.save({
            'representation': self.representation.state_dict(),
            'prediction': self.prediction.state_dict(),
            'dynamics': self.dynamics.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path="muzero_tictactoe_model.pth"):
        """Load the MuZero network model from a file."""
        checkpoint = torch.load(path)
        self.representation.load_state_dict(checkpoint['representation'])
        self.prediction.load_state_dict(checkpoint['prediction'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        print(f"Model loaded from {path}")
