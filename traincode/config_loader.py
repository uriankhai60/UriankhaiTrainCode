import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Union
import os

@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    variant: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None

@dataclass
class TrainingConfig:
    output_dir: str = "flux-control"
    seed: Optional[int] = 42
    resolution: int = 1024
    train_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    max_grad_norm: float = 1.0

@dataclass
class OptimizerConfig:
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8

@dataclass
class DataConfig:
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    jsonl_for_train: Optional[str] = None
    image_column: str = "image"
    conditioning_image_column: str = "conditioning_image"
    caption_column: str = "text"
    max_train_samples: Optional[int] = None
    proportion_empty_prompts: float = 0
    dataloader_num_workers: int = 0
    log_dataset_samples: bool = False

@dataclass
class ValidationConfig:
    validation_prompt: Optional[List[str]] = None
    validation_image: Optional[List[str]] = None
    num_validation_images: int = 1
    validation_steps: int = 100
    guidance_scale: float = 30.0

@dataclass
class CheckpointConfig:
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

@dataclass
class LoggingConfig:
    logging_dir: str = "logs"
    allow_tf32: bool = True
    report_to: str = "tensorboard"
    mixed_precision: str = "bf16"
    tracker_project_name: str = "flux_train_control"

@dataclass
class HubConfig:
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

@dataclass
class AdvancedConfig:
    only_target_transformer_blocks: bool = False
    upcast_before_saving: bool = False
    weighting_scheme: str = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    offload: bool = False

@dataclass
class FluxControlConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

def load_config_from_yaml(yaml_path: str) -> FluxControlConfig:
    """YAML 파일에서 설정을 로드합니다."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 각 섹션별로 dataclass 인스턴스 생성
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    validation_config = ValidationConfig(**config_dict.get('validation', {}))
    checkpoint_config = CheckpointConfig(**config_dict.get('checkpoint', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    hub_config = HubConfig(**config_dict.get('hub', {}))
    advanced_config = AdvancedConfig(**config_dict.get('advanced', {}))
    
    config = FluxControlConfig(
        model=model_config,
        training=training_config,
        optimizer=optimizer_config,
        data=data_config,
        validation=validation_config,
        checkpoint=checkpoint_config,
        logging=logging_config,
        hub=hub_config,
        advanced=advanced_config
    )
    
    # 설정 검증
    validate_config(config)
    
    return config

def config_to_args(config: FluxControlConfig):
    """FluxControlConfig를 기존 args 스타일 객체로 변환합니다."""
    class Args:
        pass
    
    args = Args()
    
    # Model config
    args.pretrained_model_name_or_path = config.model.pretrained_model_name_or_path
    args.variant = config.model.variant
    args.revision = config.model.revision
    args.cache_dir = config.model.cache_dir
    
    # Training config
    args.output_dir = config.training.output_dir
    args.seed = config.training.seed
    args.resolution = config.training.resolution
    args.train_batch_size = config.training.train_batch_size
    args.num_train_epochs = config.training.num_train_epochs
    args.max_train_steps = config.training.max_train_steps
    args.gradient_accumulation_steps = config.training.gradient_accumulation_steps
    args.gradient_checkpointing = config.training.gradient_checkpointing
    args.learning_rate = config.training.learning_rate
    args.scale_lr = config.training.scale_lr
    args.lr_scheduler = config.training.lr_scheduler
    args.lr_warmup_steps = config.training.lr_warmup_steps
    args.lr_num_cycles = config.training.lr_num_cycles
    args.lr_power = config.training.lr_power
    args.max_grad_norm = config.training.max_grad_norm
    
    # Optimizer config
    args.use_8bit_adam = config.optimizer.use_8bit_adam
    args.adam_beta1 = config.optimizer.adam_beta1
    args.adam_beta2 = config.optimizer.adam_beta2
    args.adam_weight_decay = config.optimizer.adam_weight_decay
    args.adam_epsilon = config.optimizer.adam_epsilon
    
    # Data config
    args.dataset_name = config.data.dataset_name
    args.dataset_config_name = config.data.dataset_config_name
    args.jsonl_for_train = config.data.jsonl_for_train
    args.image_column = config.data.image_column
    args.conditioning_image_column = config.data.conditioning_image_column
    args.caption_column = config.data.caption_column
    args.max_train_samples = config.data.max_train_samples
    args.proportion_empty_prompts = config.data.proportion_empty_prompts
    args.dataloader_num_workers = config.data.dataloader_num_workers
    args.log_dataset_samples = config.data.log_dataset_samples
    
    # Validation config
    args.validation_prompt = config.validation.validation_prompt
    args.validation_image = config.validation.validation_image
    args.num_validation_images = config.validation.num_validation_images
    args.validation_steps = config.validation.validation_steps
    args.guidance_scale = config.validation.guidance_scale
    
    # Checkpoint config
    args.checkpointing_steps = config.checkpoint.checkpointing_steps
    args.checkpoints_total_limit = config.checkpoint.checkpoints_total_limit
    args.resume_from_checkpoint = config.checkpoint.resume_from_checkpoint
    
    # Logging config
    args.logging_dir = config.logging.logging_dir
    args.allow_tf32 = config.logging.allow_tf32
    args.report_to = config.logging.report_to
    args.mixed_precision = config.logging.mixed_precision
    args.tracker_project_name = config.logging.tracker_project_name
    
    # Hub config
    args.push_to_hub = config.hub.push_to_hub
    args.hub_token = config.hub.hub_token
    args.hub_model_id = config.hub.hub_model_id
    
    # Advanced config
    args.only_target_transformer_blocks = config.advanced.only_target_transformer_blocks
    args.upcast_before_saving = config.advanced.upcast_before_saving
    args.weighting_scheme = config.advanced.weighting_scheme
    args.logit_mean = config.advanced.logit_mean
    args.logit_std = config.advanced.logit_std
    args.mode_scale = config.advanced.mode_scale
    args.offload = config.advanced.offload
    
    return args

def load_args_from_yaml(yaml_path: str):
    """YAML 파일에서 직접 args 객체를 로드합니다."""
    config = load_config_from_yaml(yaml_path)
    return config_to_args(config)

def validate_config(config: FluxControlConfig):
    """설정 값들을 검증합니다."""
    if config.data.dataset_name is None and config.data.jsonl_for_train is None:
        raise ValueError("dataset_name 또는 jsonl_for_train 중 하나는 반드시 설정해야 합니다.")
    
    if config.data.dataset_name is not None and config.data.jsonl_for_train is not None:
        raise ValueError("dataset_name과 jsonl_for_train은 동시에 설정할 수 없습니다.")
    
    if config.data.proportion_empty_prompts < 0 or config.data.proportion_empty_prompts > 1:
        raise ValueError("proportion_empty_prompts는 0과 1 사이의 값이어야 합니다.")
    
    if config.validation.validation_prompt is not None and config.validation.validation_image is None:
        raise ValueError("validation_prompt가 설정된 경우 validation_image도 설정해야 합니다.")
    
    if config.validation.validation_prompt is None and config.validation.validation_image is not None:
        raise ValueError("validation_image가 설정된 경우 validation_prompt도 설정해야 합니다.")
    
    if config.training.resolution % 8 != 0:
        raise ValueError("resolution은 8의 배수여야 합니다.")

# 하위 호환성을 위한 별칭
load_config = load_config_from_yaml