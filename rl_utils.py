"""
Utility functions for Reinforcement Learning with Human Feedback
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


class RLMetrics:
    """Metrics calculation for RL training."""
    
    @staticmethod
    def calculate_policy_gradient_loss(predictions: torch.Tensor, 
                                     rewards: torch.Tensor,
                                     actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate policy gradient loss."""
        # Convert predictions to log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        
        if actions is not None:
            # Discrete action case
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -torch.mean(selected_log_probs * rewards)
        else:
            # Continuous case - use entropy regularization
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1)
            loss = -torch.mean(log_probs.mean(dim=1) * rewards) - 0.01 * torch.mean(entropy)
        
        return loss
    
    @staticmethod
    def calculate_value_loss(predicted_values: torch.Tensor, 
                           target_values: torch.Tensor) -> torch.Tensor:
        """Calculate value function loss."""
        return F.mse_loss(predicted_values, target_values)
    
    @staticmethod
    def normalize_rewards(rewards: List[float]) -> List[float]:
        """Normalize rewards to have zero mean and unit variance."""
        rewards = np.array(rewards)
        if len(rewards) > 1:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            if std_reward > 0:
                rewards = (rewards - mean_reward) / std_reward
        return rewards.tolist()


class ExperienceBuffer:
    """Buffer for storing RL experiences."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: torch.Tensor, action: torch.Tensor, 
             reward: float, next_state: torch.Tensor, done: bool):
        """Store an experience tuple."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences."""
        return np.random.choice(self.buffer, min(batch_size, len(self.buffer)), replace=False).tolist()
    
    def __len__(self):
        return len(self.buffer)


class HumanFeedbackProcessor:
    """Process and analyze human feedback patterns."""
    
    def __init__(self):
        self.feedback_patterns = {}
        self.user_preferences = {}
    
    def analyze_feedback_consistency(self, feedback_history: List[Dict]) -> Dict[str, float]:
        """Analyze consistency in human feedback."""
        if len(feedback_history) < 2:
            return {'consistency_score': 1.0, 'variance': 0.0}
        
        scores = [epoch['avg_feedback'] for epoch in feedback_history]
        
        # Calculate running consistency
        consistency_scores = []
        for i in range(1, len(scores)):
            window = scores[max(0, i-5):i+1]  # 5-epoch window
            if len(window) > 1:
                variance = np.var(window)
                consistency = 1.0 / (1.0 + variance)  # Higher consistency = lower variance
                consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        overall_variance = np.var(scores)
        
        return {
            'consistency_score': avg_consistency,
            'variance': overall_variance,
            'trend': 'improving' if scores[-1] > scores[0] else 'declining'
        }
    
    def extract_preferences(self, feedback_history: List[Dict]) -> Dict[str, Any]:
        """Extract user preferences from feedback history."""
        preferences = {
            'preferred_feedback_range': None,
            'feedback_bias': None,
            'learning_rate_suggestion': None
        }
        
        if not feedback_history:
            return preferences
        
        all_scores = []
        for epoch in feedback_history:
            if 'feedback' in epoch:
                scores = [f['feedback_score'] for f in epoch['feedback']]
                all_scores.extend(scores)
        
        if all_scores:
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            preferences['preferred_feedback_range'] = [mean_score - std_score, mean_score + std_score]
            preferences['feedback_bias'] = 'positive' if mean_score > 0.1 else 'negative' if mean_score < -0.1 else 'neutral'
            
            # Suggest learning rate based on feedback variance
            if std_score < 0.2:
                preferences['learning_rate_suggestion'] = 'increase'  # Consistent feedback, can learn faster
            elif std_score > 0.6:
                preferences['learning_rate_suggestion'] = 'decrease'  # Inconsistent feedback, learn slower
            else:
                preferences['learning_rate_suggestion'] = 'maintain'
        
        return preferences


def create_rl_training_config(base_lr: float = 1e-4, 
                            feedback_weight: float = 1.0,
                            entropy_weight: float = 0.01) -> Dict[str, Any]:
    """Create default RL training configuration."""
    return {
        'learning_rate': base_lr,
        'feedback_weight': feedback_weight,
        'entropy_weight': entropy_weight,
        'gamma': 0.95,  # Discount factor
        'epsilon': 0.1,  # Exploration rate
        'batch_size': 32,
        'buffer_size': 10000,
        'update_frequency': 5,  # Update every N epochs
        'save_frequency': 10,   # Save model every N epochs
        'max_epochs': 100,
        'convergence_threshold': 0.01  # Stop if improvement < threshold
    }


def save_rl_checkpoint(model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer,
                      epoch: int,
                      training_history: List[Dict],
                      feedback_history: List[Dict],
                      save_path: str) -> None:
    """Save comprehensive RL checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'training_history': training_history,
        'feedback_history': feedback_history,
        'rl_config': create_rl_training_config(),
        'save_timestamp': str(np.datetime64('now'))
    }
    
    torch.save(checkpoint, save_path)
    print(f"RL checkpoint saved to {save_path}")


def load_rl_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Load RL checkpoint and restore training state."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'training_history': checkpoint.get('training_history', []),
        'feedback_history': checkpoint.get('feedback_history', []),
        'rl_config': checkpoint.get('rl_config', create_rl_training_config())
    } 