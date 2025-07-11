import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# These need to be kept in sync with the environment
ACTION_SPACE = ["move", "mine", "talk", "craft", "offer", "accept_offer", "wait"]
# Input size = 5x5x5 (view) + 12 (inventory) + 384 (memory) = 125 + 12 + 384 = 521
# The inventory size is len(RESOURCE_TYPES) + len(CRAFTING_RECIPES)
# Let's define it dynamically in a real scenario, but for now, we hardcode it.
# num_resources = 9, num_recipes = 5 -> inventory_size = 14
# Input size = 125 (view) + 14 (inventory) + 384 (memory) = 523
# Let's make inventory size flexible.
# It seems inventory size is len(RESOURCE_TYPES) + len(CRAFTING_RECIPES) = 9 + 5 = 14
# view_size=125, inv_size=14, memory_size=384 -> total=523

class StudentPolicy(nn.Module):
    def __init__(self, input_size=523, num_actions=len(ACTION_SPACE)):
        super(StudentPolicy, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def get_action_from_model(model: StudentPolicy, state_vector: np.ndarray) -> str:
    """
    Get an action from the trained policy model.
    """
    if not isinstance(state_vector, torch.Tensor):
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
    else:
        state_tensor = state_vector.float().unsqueeze(0)
        
    with torch.no_grad():
        logits = model(state_tensor)
        probabilities = F.softmax(logits, dim=1)
        action_index = torch.multinomial(probabilities, 1).item()
        
    return ACTION_SPACE[action_index] 