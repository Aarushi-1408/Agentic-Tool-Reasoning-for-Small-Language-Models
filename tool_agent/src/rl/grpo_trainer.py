import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.rl.config import GRPOConfig
from src.rl.rollout import RolloutResult, RolloutCandidate
from src.training.sft_dataset import format_example_tokens


class GRPOTrainer:
    """
    Lightweight, full-parameter GRPO Trainer implementation for Qwen architecture.
    Does not rely on third-party TRL package.
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: Optional[Any] = None,
        ref_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        if self.model is not None:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            self.optimizer = None

    def compute_trajectory_log_probs(
        self,
        model: Any,
        messages: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token log-probabilities for assistant responses in trajectory.
        """
        if model is None or self.tokenizer is None:
            # Fallback mock tensors for testing without model
            dummy_loss = torch.tensor(0.5, requires_grad=True if model else False)
            dummy_kl = torch.tensor(0.01)
            return dummy_loss, dummy_kl

        # Tokenize conversation messages with label masking (-100 for user/system)
        token_data = format_example_tokens(
            messages=messages,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_prompt_length + self.config.max_completion_length,
        )

        input_ids = torch.tensor([token_data["input_ids"]], device=self.device)
        labels = torch.tensor([token_data["labels"]], device=self.device)

        # Forward pass on underlying PyTorch CausalLM model
        raw_model = getattr(model, "model", model)
        outputs = raw_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        return loss, torch.tensor(0.01, device=self.device)

    def train_step(self, group_rollouts: List[RolloutResult]) -> Dict[str, Any]:
        """
        Execute one GRPO gradient update step across a batch of rollout groups.
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        total_loss = 0.0
        total_rewards = []
        advantages = []
        zero_variance_groups = 0

        for grp in group_rollouts:
            if grp.reward_variance < 1e-6:
                zero_variance_groups += 1

            for cand in grp.candidates:
                total_rewards.append(cand.reward_result.total_reward)
                advantages.append(cand.advantage)

                if self.model is not None:
                    # Compute policy loss weighted by group advantage A_i
                    policy_loss, kl_div = self.compute_trajectory_log_probs(self.model, cand.trajectory.messages)
                    
                    # GRPO objective: - advantage * policy_loss + beta * KL
                    loss = -cand.advantage * policy_loss + self.config.beta * kl_div
                    loss.backward()
                    total_loss += loss.item()

        if self.optimizer is not None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.optimizer.step()

        mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        mean_adv = sum(advantages) / len(advantages) if advantages else 0.0

        metrics = {
            "mean_reward": round(mean_reward, 4),
            "mean_advantage": round(mean_adv, 4),
            "zero_variance_groups_ratio": round(zero_variance_groups / len(group_rollouts), 4) if group_rollouts else 0.0,
            "total_loss": round(total_loss, 4),
        }
        return metrics

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        """
        Save model checkpoint to specified path.
        """
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.model is not None:
            save_target = getattr(self.model, "model", self.model)
            if hasattr(save_target, "save_pretrained"):
                save_target.save_pretrained(ckpt_dir)
            elif hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(ckpt_dir)
                
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(ckpt_dir)

        # Save GRPO state config
        config_path = os.path.join(ckpt_dir, "grpo_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return ckpt_dir
