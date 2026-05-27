import numpy as np


class RLHFSimulator:
    """
    Reinforcement Learning with Human Feedback (RLHF)

    DEFINITION: Technique where a model is aligned with human preferences by
                using human feedback as reward signal.

    HOW IT WORKS:
    1. Train reward model from human preferences
    2. Use PPO (Proximal Policy Optimization) to optimize policy
    3. Human ranks multiple outputs → used to fine-tune model

    WHERE IT IS USED: ChatGPT, Claude, Llama, Grok (to make AI helpful and safe).
    """

    def __init__(self):
        self.reward_model = None
        self.policy = "Base Policy"

    def train_reward_model(self, prompts, responses, human_preferences):
        """Simulate training reward model from human rankings"""
        print(f"Trained reward model on {len(prompts)} human preference pairs")
        return self

    def optimize_with_ppo(self, epochs=5):
        """Simulate RLHF fine-tuning"""
        for i in range(epochs):
            print(f"PPO Epoch {i+1}: Policy updated based on human feedback")
        self.policy = "Aligned Policy (RLHF)"
        return self


# ===================== SAMPLE USAGE =====================
if __name__ == "__main__":
    rlhf = RLHFSimulator()

    # Sample data
    prompts = ["Explain AI", "Write a story"]
    responses = ["Response A", "Response B"]
    preferences = [1, 0]  # Human preferred response index

    rlhf.train_reward_model(prompts, responses, preferences)
    rlhf.optimize_with_ppo(epochs=3)

    print("\nFinal Policy:", rlhf.policy)
    print("RLHF makes AI more helpful, honest, and human-aligned.")
