from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np

class TrafficTrainingCallbacks(DefaultCallbacks):
    """交通控制训练的自定义回调函数"""
    
    def on_train_result(self, trainer, result: dict, **kwargs):
        """每次训练后调用"""
        print(f"Episode {result['episodes_total']}: "
              f"Reward={result['episode_reward_mean']:.2f}, "
              f"Length={result['episode_len_mean']:.1f}, "
              f"KL={result.get('kl', 0):.4f}")
        
        # 记录安全相关指标
        if "custom_metrics" in result:
            custom_metrics = result["custom_metrics"]
            if "safety_interventions_mean" in custom_metrics:
                print(f"  Safety interventions: {custom_metrics['safety_interventions_mean']:.1f}")
            if "controlled_vehicles_mean" in custom_metrics:
                print(f"  Controlled vehicles: {custom_metrics['controlled_vehicles_mean']:.1f}")
            if "episode_cost_mean" in custom_metrics:
                print(f"  Episode cost: {custom_metrics['episode_cost_mean']:.2f}")
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """每次episode结束时调用"""
        # 从最后一个info中提取安全相关信息
        last_info = episode.last_info_for()
        if last_info:
            safety_interventions = last_info.get("safety_interventions", 0)
            num_controlled = last_info.get("num_controlled", 0)
            emergency_count = last_info.get("emergency_count", 0)
            episode_cost = last_info.get("episode_cost", 0.0)
            
            episode.custom_metrics["safety_interventions"] = safety_interventions
            episode.custom_metrics["controlled_vehicles"] = num_controlled
            episode.custom_metrics["emergency_count"] = emergency_count
            episode.custom_metrics["episode_cost"] = episode_cost

class ConstrainedTrainingCallbacks(TrafficTrainingCallbacks):
    """约束训练的自定义回调函数"""
    
    def on_train_result(self, trainer, result: dict, **kwargs):
        """每次训练后调用，更新拉格朗日乘子"""
        super().on_train_result(trainer, result, **kwargs)
        
        # 1. 获取平均episode成本
        if "custom_metrics" in result and "episode_cost_mean" in result["custom_metrics"]:
            mean_ep_cost = result["custom_metrics"]["episode_cost_mean"]
            
            # 2. 更新拉格朗日乘子（通过trainer的workers）
            try:
                trainer.workers.foreach_worker(
                    lambda w: w.foreach_policy(
                        lambda policy, pid: policy.update_lagrange_multiplier(mean_ep_cost) 
                        if hasattr(policy, 'update_lagrange_multiplier') 
                        else None
                    )
                )
            except:
                # 如果上面的方法不起作用，尝试其他方法
                pass
            
            print(f"  Cost={mean_ep_cost:.2f}")
        
        # 3. 课程学习：动态调整成本限制
        if trainer.config.get("curriculum_enabled", False):
            curriculum_config = trainer.config.get("curriculum_config", {})
            cost_schedule = curriculum_config.get("cost_limit_schedule", [])
            
            current_iter = result["training_iteration"]
            for stage in cost_schedule:
                if current_iter >= stage["episode"]:
                    new_limit = stage["limit"]
                    
                    # 更新环境的成本限制
                    try:
                        trainer.workers.foreach_worker(
                            lambda w: w.foreach_env(
                                lambda env: setattr(env, 'cost_limit', new_limit) 
                                if hasattr(env, 'cost_limit') else None
                            )
                        )
                    except:
                        pass
                    
                    print(f"  Curriculum: Cost limit updated to {new_limit} at iteration {current_iter}")
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """每次episode结束时调用，记录成本信息"""
        super().on_episode_end(worker, base_env, policies, episode, **kwargs)
        
        # 从环境获取成本信息
        last_info = episode.last_info_for()
        if last_info:
            episode_cost = last_info.get("episode_cost", 0.0)
            safety_cost = last_info.get("safety_cost", 0.0)
            control_cost = last_info.get("control_cost", 0.0)
            
            episode.custom_metrics["episode_cost"] = episode_cost
            episode.custom_metrics["safety_cost"] = safety_cost
            episode.custom_metrics["control_cost"] = control_cost