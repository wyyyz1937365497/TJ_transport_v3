import os
import json
import time
import argparse
import logging
from typing import Dict, Any

class TrainingPipeline:
    """
    三阶段训练管道
    管理阶段切换和模型传递
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载基础配置"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        log_dir = self.base_config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger("TrainingPipeline")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_phase1(self):
        """运行阶段1：世界模型预训练"""
        self.logger.info("=" * 50)
        self.logger.info("Starting Phase 1: World Model Pre-training")
        self.logger.info("=" * 50)
        
        # 准备阶段1配置
        phase1_config = self.base_config["phase1_config"]
        
        # 1. 数据收集
        if phase1_config.get("collect_data", True):
            self.logger.info("Collecting training data...")
            from src.dataset.data_collector import WorldModelDataCollector
            
            collector = WorldModelDataCollector(phase1_config["data_collection"])
            collector.collect_data(
                num_episodes=phase1_config["data_collection"]["num_episodes"],
                strategy=phase1_config["data_collection"]["strategy"]
            )
            collector.save_buffer(phase1_config["data_collection"]["buffer_file"])
        
        # 2. 训练世界模型
        self.logger.info("Training world model...")
        from scripts.train_world_model import train_world_model_phase1, train_world_model_phase2
        
        # Phase 1 training
        phase1_model = train_world_model_phase1(phase1_config["training"])
        
        # Phase 2 training
        self.logger.info("Training world model phase 2...")
        phase2_model = train_world_model_phase2(phase1_config["training"], phase1_model)
        
        self.logger.info("Phase 1 completed successfully!")
        return phase1_config["training"]["final_model_path"]
    
    def run_phase2(self, world_model_path: str):
        """运行阶段2：带安全屏障的RL训练"""
        self.logger.info("=" * 50)
        self.logger.info("Starting Phase 2: Safe RL Training")
        self.logger.info("=" * 50)
        
        # 准备阶段2配置
        phase2_config = self.base_config["phase2_config"]
        
        # 更新世界模型路径
        phase2_config["model"]["custom_model_config"]["world_model_path"] = world_model_path
        
        # 保存阶段2配置
        os.makedirs(os.path.dirname(phase2_config["config_file"]), exist_ok=True)
        with open(phase2_config["config_file"], 'w') as f:
            json.dump(phase2_config, f, indent=2)
        
        # 运行阶段2训练
        from scripts.train_safe_rl import train_safe_rl
        
        train_safe_rl(phase2_config["config_file"])
        
        # 获取最佳检查点
        checkpoint_dir = phase2_config["experiment"]["checkpoint_dir"]
        best_checkpoint = os.path.join(checkpoint_dir, "best_checkpoint")
        
        self.logger.info("Phase 2 completed successfully!")
        return best_checkpoint
    
    def run_phase3(self, safe_rl_checkpoint: str):
        """运行阶段3：约束优化训练"""
        self.logger.info("=" * 50)
        self.logger.info("Starting Phase 3: Constrained RL Training")
        self.logger.info("=" * 50)
        
        # 准备阶段3配置
        phase3_config = self.base_config["phase3_config"]
        
        # 设置恢复检查点
        phase3_config["resume_from"] = safe_rl_checkpoint
        
        # 保存阶段3配置
        os.makedirs(os.path.dirname(phase3_config["config_file"]), exist_ok=True)
        with open(phase3_config["config_file"], 'w') as f:
            json.dump(phase3_config, f, indent=2)
        
        # 运行阶段3训练
        from scripts.train_constrained_rl import train_constrained_rl
        
        train_constrained_rl(phase3_config["config_file"], safe_rl_checkpoint)
        
        self.logger.info("Phase 3 completed successfully!")
    
    def run_full_pipeline(self):
        """运行完整三阶段训练管道"""
        start_time = time.time()
        
        try:
            # 阶段1：世界模型预训练
            world_model_path = self.run_phase1()
            
            # 阶段2：安全RL训练
            safe_rl_checkpoint = self.run_phase2(world_model_path)
            
            # 阶段3：约束优化训练
            self.run_phase3(safe_rl_checkpoint)
            
            total_time = time.time() - start_time
            self.logger.info(f"Full training pipeline completed successfully in {total_time/3600:.2f} hours!")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full three-stage training pipeline")
    parser.add_argument('--config', type=str, default='configs/training_pipeline_config.json',
                        help="Path to pipeline configuration file")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                        help="Run specific phase only (1, 2, or 3)")
    parser.add_argument('--resume-phase2', type=str, default=None,
                        help="Resume phase 2 from specific checkpoint")
    parser.add_argument('--resume-phase3', type=str, default=None,
                        help="Resume phase 3 from specific checkpoint")
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(args.config)
    
    if args.phase == 1:
        pipeline.run_phase1()
    elif args.phase == 2:
        if args.resume_phase2:
            # 从指定检查点恢复阶段2
            pipeline.run_phase2(args.resume_phase2)
        else:
            # 需要先运行阶段1或指定世界模型路径
            world_model_path = pipeline.base_config["phase1_config"]["training"]["final_model_path"]
            pipeline.run_phase2(world_model_path)
    elif args.phase == 3:
        if args.resume_phase3:
            pipeline.run_phase3(args.resume_phase3)
        else:
            # 需要先运行阶段2或指定检查点
            safe_rl_checkpoint = pipeline.base_config["phase2_config"]["experiment"]["checkpoint_dir"]
            pipeline.run_phase3(safe_rl_checkpoint)
    else:
        # 运行完整管道
        pipeline.run_full_pipeline()