import math
import torch.nn.functional as F

class MCTSNode:
    def __init__(self, prior=0.0, state=None):
        self.state = state.clone() if state is not None else None
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.prior = prior
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
def mcts_search(root, network, num_simulations):
    for _ in range(num_simulations):
        node = MCTSNode(state=root.state.clone())
        search_path = [node]

        # Traverse tree
        while node.children:
            action, node = max(node.children.items(), key=lambda item: ucb_score(node, item[1]))
            search_path.append(node)
        
        # Expand
        state, policy_logits, _ = network.initial_inference(node.state)
        policy = F.softmax(policy_logits, dim=0).detach().numpy()
        node.children = {a: MCTSNode(prior=p) for a, p in enumerate(policy)}

        # Backup
        value = node.value()
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1

def ucb_score(parent, child):
    c = 1.25  # Exploration parameter
    return child.value() + c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
