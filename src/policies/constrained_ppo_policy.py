from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.torch_ops import explained_variance
import torch
import torch.nn.functional as F
import numpy as np

def custom_constrained_ppo_loss(policy, model, dist_class, train_batch):
    """
    v4.0约束优化PPO损失
    包含：
    1. 标准PPO损失
    2. Cost Critic损失
    3. 拉格朗日约束
    """
    
    # 1. 标准PPO前向传播
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)
    
    # 2. 计算动作概率比
    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch["actions"]) - train_batch["action_logp"]
    )
    
    # 3. 标准PPO损失
    action_kl = curr_action_dist.kl(train_batch["action_dist_inputs"])
    mean_kl = torch.mean(action_kl)
    
    curr_epsilon = policy.config["clip_param"]
    surrogate_loss = torch.min(
        train_batch["advantages"] * logp_ratio,
        train_batch["advantages"] * torch.clamp(logp_ratio, 1 - curr_epsilon, 1 + curr_epsilon),
    )
    
    # 4. Value函数损失
    value_targets = train_batch["value_targets"]
    value_pred = model.value_function()
    vf_loss = torch.pow(value_pred - value_targets, 2.0)
    vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
    mean_vf_loss = torch.mean(vf_loss_clipped)
    
    # 5. 【v4.0核心】Cost Critic损失
    if "cost_targets" in train_batch:
        cost_targets = train_batch["cost_targets"]
        cost_pred = model.cost_value_function()
        cost_vf_loss = torch.pow(cost_pred - cost_targets, 2.0)
        mean_cost_vf_loss = torch.mean(cost_vf_loss)
    else:
        cost_vf_loss = torch.tensor(0.0, device=value_pred.device)
        mean_cost_vf_loss = torch.tensor(0.0, device=value_pred.device)
    
    # 6. 【v4.0核心】拉格朗日约束
    if "cost_targets" in train_batch:
        lagrange_multiplier = model.lagrange_multiplier
        cost_constraint = torch.mean(cost_pred) - policy.config.get("cost_limit", 10.0)
        constrained_loss = lagrange_multiplier * cost_constraint
    else:
        constrained_loss = torch.tensor(0.0, device=value_pred.device)
    
    # 7. 总损失
    total_loss = (
        -torch.mean(surrogate_loss) +  # Policy loss
        policy.config["vf_loss_coeff"] * mean_vf_loss +  # Value loss
        policy.config.get("vf_loss_coeff", 1.0) * mean_cost_vf_loss +  # Cost value loss
        policy.config["entropy_coeff"] * -torch.mean(curr_action_dist.entropy()) +  # Entropy
        policy.config.get("cost_coeff", 1.0) * constrained_loss  # Lagrange constraint
    )
    
    # 8. 调试信息
    policy.explained_vf_adv = explained_variance(
        train_batch["value_targets"], value_pred
    )
    if "cost_targets" in train_batch:
        policy.explained_vf_cost = explained_variance(
            train_batch["cost_targets"], cost_pred
        )
    
    # 9. 更新拉格朗日乘子
    if "cost_targets" in train_batch:
        mean_ep_cost = torch.mean(cost_targets).item()
        model.update_lagrange_multiplier(mean_ep_cost)
    
    return total_loss

# 自定义策略类
ConstrainedPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="ConstrainedPPOTorchPolicy",
    loss_fn=custom_constrained_ppo_loss,
    stats_fn=lambda policy: {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "policy_loss": -torch.mean(policy.surrogate_loss).item() if hasattr(policy, 'surrogate_loss') else 0,
        "vf_loss": policy.vf_loss.item() if hasattr(policy, 'vf_loss') else 0,
        "cost_vf_loss": policy.mean_cost_vf_loss.item() if hasattr(policy, 'mean_cost_vf_loss') else 0,
        "entropy": policy.entropy.item() if hasattr(policy, 'entropy') else 0,
        "lagrange_multiplier": policy.model.lagrange_multiplier.item() if hasattr(policy.model, 'lagrange_multiplier') else 0,
        "mean_kl": policy.mean_kl.item() if hasattr(policy, 'mean_kl') else 0,
        "explained_vf_adv": policy.explained_vf_adv.item() if hasattr(policy, 'explained_vf_adv') else 0,
        "explained_vf_cost": policy.explained_vf_cost.item() if hasattr(policy, 'explained_vf_cost') else 0,
    }
)