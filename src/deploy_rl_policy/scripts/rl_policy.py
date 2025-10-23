import torch
import torch.nn as nn

# =====================================================================================
# RLPolicy Class - Modified to call the JIT model correctly
# =====================================================================================

class RLPolicy:
    def __init__(self, policy_path):
        print("Loading TorchScript policy from:", policy_path)
        
        # Use torch.jit.load() to directly load the entire compiled model.
        self.actor_critic = torch.jit.load(policy_path, map_location=torch.device('cpu'))
        
        # Set the model to evaluation mode
        self.actor_critic.eval()
        
        print("TorchScript RL Policy loaded successfully!")

    def get_action(self, observations):
        # Convert observations to a tensor
        obs_batch = torch.from_numpy(observations).float().unsqueeze(0)
        
        # Get actions from the policy
        with torch.no_grad():
            # ---
            # THE FIX IS HERE:
            # Call the loaded JIT module directly as a function.
            # This will execute its compiled 'forward' method, which in the case of
            # ActorCritic, returns the action mean from the actor network.
            # ---
            actions_mean = self.actor_critic(obs_batch)
        
        return actions_mean.detach().numpy().flatten()