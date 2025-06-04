#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse # YAML 로더를 사용하므로 직접적인 argparse 사용은 줄어들지만, 호환성을 위해 유지 가능
import copy
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

# YAML 설정 로더 import
from config_loader import load_args_from_yaml # YAML 파일에서 설정을 로드하는 함수

if is_wandb_available():
    import wandb

# Diffusers 최소 버전 확인
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

# 정규화 레이어 접두사 (학습 대상 파라미터 식별에 사용될 수 있음)
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    """
    VAE를 사용하여 이미지를 잠재 공간으로 인코딩합니다.

    Args:
        pixels (torch.Tensor): 인코딩할 이미지 텐서.
        vae (torch.nn.Module): 사전 학습된 VAE 모델.
        weight_dtype: 가중치 데이터 타입 (e.g., torch.float16).

    Returns:
        torch.Tensor: 인코딩된 잠재 벡터.
    """
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(flux_transformer, args, accelerator, weight_dtype, step, is_final_validation=False):
    """
    검증 프롬프트를 사용하여 이미지를 생성하고 로깅합니다.

    Args:
        flux_transformer (FluxTransformer2DModel): 학습된 (또는 학습 중인) Flux Transformer 모델.
        args (Namespace): 학습 설정 값.
        accelerator (Accelerator): Accelerate 라이브러리 인스턴스.
        weight_dtype: 가중치 데이터 타입.
        step (int): 현재 학습 스텝.
        is_final_validation (bool): 최종 검증인지 여부.

    Returns:
        list: 생성된 이미지와 관련 정보를 담은 로그 리스트.
    """
    logger.info("Running validation... ")

    # 검증 시 사용할 파이프라인 로드
    if not is_final_validation:
        # 학습 중인 모델 사용
        flux_transformer_model = accelerator.unwrap_model(flux_transformer)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=flux_transformer_model,
            torch_dtype=weight_dtype,
        )
    else:
        # 최종 저장된 모델 사용
        transformer_model = FluxTransformer2DModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer_model,
            torch_dtype=weight_dtype,
        )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True) # 검증 중에는 진행 표시줄 비활성화

    # 시드 설정 (재현성을 위해)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # 검증 이미지 및 프롬프트 준비
    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args` or YAML config"
        )

    image_logs = []
    # MPS (Apple Silicon) 환경에서는 autocast 컨텍스트를 다르게 처리
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

    # 각 검증 프롬프트 및 이미지에 대해 이미지 생성
    for validation_prompt, validation_image_path in zip(validation_prompts, validation_images):
        validation_image = load_image(validation_image_path)
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    control_image=validation_image,
                    num_inference_steps=50, # 검증 시 추론 스텝 수
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=512, # T5 인코더의 최대 시퀀스 길이
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            images.append(image)
        
        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    # 생성된 이미지를 트래커(TensorBoard, WandB 등)에 로깅
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images_to_log = [log["validation_image"]] + log["images"]
                formatted_images = [np.asarray(img) for img in images_to_log]
                formatted_images_np = np.stack(formatted_images)
                tracker.writer.add_images(f"{tracker_key}/{log['validation_prompt']}", formatted_images_np, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images_wandb = []
            for log in image_logs:
                formatted_images_wandb.append(wandb.Image(log["validation_image"], caption="Conditioning Image"))
                for i, img in enumerate(log["images"]):
                    img = wandb.Image(img, caption=f"{log['validation_prompt']} - Generated {i+1}")
                    formatted_images_wandb.append(img)
            tracker.log({tracker_key: formatted_images_wandb}, step=step)
        else:
            logger.warning(f"Image logging not implemented for tracker: {tracker.name}")

    del pipeline
    free_memory() # 메모리 확보
    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None, args=None):
    """
    Hugging Face Hub에 업로드할 모델 카드를 생성합니다.

    Args:
        repo_id (str): 리포지토리 ID.
        image_logs (list, optional): 검증 이미지 로그.
        base_model (str, optional): 기반 모델 이름.
        repo_folder (str, optional): 모델 파일이 저장된 로컬 폴더.
        args (Namespace): 학습 설정 값.
    """
    img_str = ""
    if image_logs is not None and repo_folder is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            
            # 제어 이미지 저장
            control_image_path = os.path.join(repo_folder, f"control_image_{i}.png")
            validation_image.save(control_image_path)
            img_str += f"**Control Image {i}:**\n![control_image_{i}](./control_image_{i}.png)\n"
            
            # 생성된 이미지 그리드 저장
            grid_images = [validation_image] + images
            image_grid = make_image_grid(grid_images, rows=1, cols=len(grid_images))
            grid_path = os.path.join(repo_folder, f"generated_images_grid_{i}.png")
            image_grid.save(grid_path)
            img_str += f"**Prompt:** {validation_prompt}\n"
            img_str += f"![generated_images_grid_{i}](./generated_images_grid_{i}.png)\n\n"

    model_description = f"""
# flux-control-{repo_id}

These are ControlNet-like weights trained on `{base_model}` for FLUX models, enabling conditional image generation using control images.
This model was trained with the `diffusers` library.

{img_str}

## Usage

```python
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image

# Load the base FLUX pipeline
base_model_id = "{args.pretrained_model_name_or_path if args else 'your-base-flux-model'}"
# Load the trained ControlNet weights
control_model_id = "{repo_id}" # Or your local path to the control weights (args.output_dir)

transformer = FluxTransformer2DModel.from_pretrained(control_model_id)
pipeline = FluxControlPipeline.from_pretrained(base_model_id, transformer=transformer, torch_dtype=torch.float16)
pipeline.to("cuda")

prompt = "A beautiful landscape painting"
control_image = load_image("path/to/your/control_image.png").resize((1024, 1024)) # Ensure image is 1024x1024

image = pipeline(
    prompt, 
    control_image=control_image, 
    guidance_scale={args.guidance_scale if args else 7.0}, 
    num_inference_steps=50
).images[0]
image.save("controlled_image.png")
```

## License

Please adhere to the licensing terms of the base model `{base_model}`.
The original FLUX model license can typically be found on its Hugging Face model card.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other", # 기반 모델의 라이선스를 따르도록 명시
        base_model=base_model,
        model_description=model_description,
        inference=True, # 추론 가능함을 명시
    )

    tags = [
        "flux",
        "controlnet", # 유사 기술이므로 태그 추가
        "flux-diffusers",
        "text-to-image",
        "image-to-image", # 제어 이미지를 사용하므로
        "diffusers",
        "control",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    if repo_folder is not None:
        model_card.save(os.path.join(repo_folder, "README.md"))
    else:
        logger.warning("repo_folder not provided, model card will not be saved locally.")


def get_train_dataset(args, accelerator):
    """
    학습 데이터셋을 로드하고 전처리합니다.

    Args:
        args (Namespace): 학습 설정 값.
        accelerator (Accelerator): Accelerate 라이브러리 인스턴스.

    Returns:
        datasets.Dataset: 전처리된 학습 데이터셋.
    """
    dataset = None
    # Hugging Face Hub 또는 로컬 경로에서 데이터셋 로드
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name, # 데이터셋의 특정 설정 (e.g., 'default')
            cache_dir=args.cache_dir, # 캐시 디렉토리
        )
    # JSONL 파일에서 데이터셋 로드
    elif args.jsonl_for_train is not None:
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices() # 중첩된 인덱스 평탄화
    else:
        raise ValueError("Either --dataset_name or --jsonl_for_train must be provided in the config.")

    # 데이터셋 컬럼 이름 확인
    column_names = dataset["train"].column_names

    # 이미지, 캡션, 조건 이미지 컬럼 설정
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(f"Image column '{image_column}' not found in dataset. Available columns: {column_names}")
    
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(f"Caption column '{caption_column}' not found in dataset. Available columns: {column_names}")

    conditioning_image_column = args.conditioning_image_column
    if conditioning_image_column not in column_names:
        raise ValueError(f"Conditioning image column '{conditioning_image_column}' not found in dataset. Available columns: {column_names}")

    # 메인 프로세스에서만 데이터셋 셔플 및 샘플 선택 수행
    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed) # 데이터셋 셔플
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples)) # 최대 학습 샘플 수 제한
    return train_dataset


def prepare_train_dataset(dataset, args, accelerator):
    """
    학습 데이터셋에 대한 이미지 변환 및 전처리를 정의합니다.

    Args:
        dataset (datasets.Dataset): 원본 학습 데이터셋.
        args (Namespace): 학습 설정 값.
        accelerator (Accelerator): Accelerate 라이브러리 인스턴스.

    Returns:
        datasets.Dataset: 변환이 적용된 학습 데이터셋.
    """
    # 이미지 변환 정의: 리사이즈, 텐서 변환, 정규화
    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 이미지를 [-1, 1] 범위로 정규화
        ]
    )

    def preprocess_train(examples):
        # 타겟 이미지 로드 및 변환
        images = [
            (Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB"))
            for image_path in examples[args.image_column]
        ]
        images = [image_transforms(image) for image in images]

        # 조건 이미지 로드 및 변환
        conditioning_images = [
            (Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB"))
            for image_path in examples[args.conditioning_image_column]
        ]
        conditioning_images = [image_transforms(image) for image in conditioning_images]
        
        examples["pixel_values"] = images # 변환된 타겟 이미지
        examples["conditioning_pixel_values"] = conditioning_images # 변환된 조건 이미지

        # 캡션 처리 (리스트인 경우 가장 긴 캡션 선택)
        is_caption_list = isinstance(examples[args.caption_column][0], list)
        if is_caption_list:
            examples["captions"] = [max(example_captions, key=len) if example_captions else "" for example_captions in examples[args.caption_column]]
        else:
            examples["captions"] = list(examples[args.caption_column])
        
        # 빈 캡션 처리 (None일 경우 빈 문자열로)
        examples["captions"] = [cap if cap is not None else "" for cap in examples["captions"]]


        return examples

    # 메인 프로세스에서만 데이터셋 변환 적용
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    """
    데이터로더를 위한 배치 콜레이트 함수.

    Args:
        examples (list): 샘플 리스트.

    Returns:
        dict: 배치된 텐서를 포함하는 딕셔너리.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    captions = [example["captions"] for example in examples]
    
    return {"pixel_values": pixel_values, "conditioning_pixel_values": conditioning_pixel_values, "captions": captions}


def main(config_path: str):
    """
    메인 학습 함수.

    Args:
        config_path (str): YAML 설정 파일 경로.
    """
    # YAML 파일에서 설정 로드
    args = load_args_from_yaml(config_path)

    # WandB 사용 시 Hub 토큰 동시 사용 방지 (보안 위험)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # 로깅 디렉토리 설정
    logging_out_dir = Path(args.output_dir, args.logging_dir)

    # MPS 환경에서 bf16 사용 불가 처리
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Accelerator 프로젝트 설정
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    # Accelerator 초기화
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to, # 로깅 대상 (tensorboard, wandb 등)
        project_config=accelerator_project_config,
    )

    # MPS 환경에서 AMP 비활성화 (AMP: Automatic Mixed Precision)
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP for stability.")
        accelerator.native_amp = False

    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False) # Accelerator 상태 로깅

    # 프로세스별 로깅 레벨 설정
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 시드 설정 (재현성을 위해)
    if args.seed is not None:
        set_seed(args.seed)

    # Hugging Face Hub 리포지토리 생성 (메인 프로세스에서만)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True) # 출력 디렉토리 생성

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, # 리포지토리 ID (없으면 출력 디렉토리 이름 사용)
                exist_ok=True, 
                token=args.hub_token
            ).repo_id
            logger.info(f"Created/updated Hub repository: {repo_id}")

    # 모델 로드: VAE, Flux Transformer
    # VAE는 이미지 인코딩/디코딩에 사용
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae", # VAE 가중치가 있는 하위 폴더
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) # VAE 스케일 팩터 계산

    # Flux Transformer 모델 로드
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer", # Transformer 가중치가 있는 하위 폴더
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models (VAE, Flux Transformer) loaded successfully.")

    # 노이즈 스케줄러 로드 (Flow Matching 용)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler", # 스케줄러 설정이 있는 하위 폴더
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler) # 시그마 계산 등에 사용할 복사본

    # 학습 대상 파라미터 설정
    if not args.only_target_transformer_blocks:
        flux_transformer.requires_grad_(True) # 전체 Transformer 학습
    else:
        # 특정 블록만 학습 (x_embedder 및 transformer_blocks)
        logger.info("Training only target transformer blocks and x_embedder.")
        flux_transformer.requires_grad_(False) # 기본적으로 모든 파라미터 동결
        flux_transformer.x_embedder.requires_grad_(True)
        for name, module in flux_transformer.named_modules():
            if "transformer_blocks" in name: # 'transformer_blocks'를 포함하는 이름의 모듈만 학습
                module.requires_grad_(True)
    
    vae.requires_grad_(False) # VAE는 학습하지 않음 (사전 학습된 가중치 사용)

    # 가중치 데이터 타입 설정
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=torch.float32) # VAE는 float32로 유지 (안정성)

    # Flux Transformer 입력 채널 수정 (제어 이미지 입력을 위해)
    with torch.no_grad():
        initial_input_channels = flux_transformer.config.in_channels # 기존 입력 채널 수
        # 제어 잠재 벡터를 이어붙이기 위해 x_embedder의 입력 차원을 2배로 늘림
        new_in_features = flux_transformer.x_embedder.in_features * 2
        new_linear = torch.nn.Linear(
            new_in_features,
            flux_transformer.x_embedder.out_features,
            bias=flux_transformer.x_embedder.bias is not None,
            dtype=flux_transformer.dtype, # 모델의 기존 dtype 사용
            device=flux_transformer.device, # 모델의 기존 device 사용
        )
        # 새 Linear 레이어 가중치 초기화:
        # 앞부분은 기존 가중치 복사, 뒷부분(제어 잠재 벡터용)은 0으로 초기화
        new_linear.weight.data.zero_() 
        new_linear.weight.data[:, :initial_input_channels] = flux_transformer.x_embedder.weight.data.clone()
        if flux_transformer.x_embedder.bias is not None:
            new_linear.bias.data.copy_(flux_transformer.x_embedder.bias.data)
        
        flux_transformer.x_embedder = new_linear # 기존 x_embedder를 새 레이어로 교체
        logger.info(f"Modified flux_transformer.x_embedder for control inputs. New in_features: {new_in_features}")

    # 변경된 입력 채널 수 config에 등록
    flux_transformer.register_to_config(in_channels=initial_input_channels * 2, out_channels=initial_input_channels)


    # 모델 언래핑 함수 (Accelerator 사용 시 필요)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 사용자 정의 모델 저장 및 로드 훅 (Accelerate 0.16.0 이상)
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model_idx, model in enumerate(models):
                    # FluxTransformer2DModel 인스턴스만 따로 저장
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        unwrapped_flux_model = unwrap_model(model)
                        # upcast_before_saving 옵션에 따라 float32로 업캐스팅 후 저장
                        if args.upcast_before_saving:
                             logger.info(f"Upcasting model to float32 before saving to {output_dir}/transformer")
                             unwrapped_flux_model.to(torch.float32)
                        unwrapped_flux_model.save_pretrained(os.path.join(output_dir, "transformer"))
                        logger.info(f"Saved FluxTransformer2DModel to {os.path.join(output_dir, 'transformer')}")
                        # 원래 데이터 타입으로 되돌리기 (메모리 절약 및 다음 학습 스텝을 위해)
                        if args.upcast_before_saving:
                            unwrapped_flux_model.to(dtype=weight_dtype)

                    else:
                        logger.warning(f"Unexpected model type found in save_model_hook: {model.__class__}. Skipping custom save.")
                    
                    if weights: # Accelerator 내부 로직을 위해 weights에서 해당 모델 제거
                       weights.pop(model_idx)


        def load_model_hook(models, input_dir):
            # DeepSpeed를 사용하지 않을 때의 로드 로직
            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        # FluxTransformer2DModel.from_pretrained를 사용하여 로드
                        # 이 부분은 accelerator가 내부적으로 처리하므로,
                        # 실제로는 해당 타입의 모델을 식별하여 accelerator가 올바르게 로드하도록 돕는 역할
                        logger.info(f"Identified FluxTransformer2DModel for loading from {input_dir}")
                    else:
                        logger.warning(f"Unexpected model type found in load_model_hook: {unwrap_model(model).__class__}")
            else: # DeepSpeed 사용 시
                # DeepSpeed는 모델을 직접 로드해야 할 수 있음
                loaded_transformer = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                # 로드된 모델을 accelerator가 관리하는 모델 리스트에 추가하거나 대체하는 로직 필요
                # (이 부분은 Accelerate 버전에 따라 구현이 다를 수 있어 주의 필요)
                logger.info(f"Loaded FluxTransformer2DModel for DeepSpeed from {os.path.join(input_dir, 'transformer')}")
                # `models` 리스트를 직접 수정해야 할 수 있음. 예: models[0] = loaded_transformer

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        logger.info("Registered custom save and load hooks for Accelerator.")

    # 그래디언트 체크포인팅 활성화 (메모리 절약, 속도 저하)
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")

    # TF32 활성화 (Ampere GPU에서 학습 속도 향상)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("TF32 allowed on CUDA matmul.")

    # 학습률 스케일링
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        logger.info(f"Scaled learning rate to: {args.learning_rate}")

    # 8-bit Adam 옵티마이저 사용 설정
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer.")
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`")
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer.")

    # 옵티마이저 파라미터 설정
    # 학습 가능한 파라미터만 옵티마이저에 전달
    params_to_optimize = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 데이터셋 및 데이터로더 준비
    train_dataset = get_train_dataset(args, accelerator)
    train_dataset = prepare_train_dataset(train_dataset, args, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True, # 에폭마다 데이터 셔플
        collate_fn=collate_fn, # 배치 생성 함수
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, # 데이터 로딩 워커 수
    )
    logger.info(f"Prepared train dataloader with batch size {args.train_batch_size} and {args.dataloader_num_workers} workers.")

    # 학습 스텝 수 계산
    if args.max_train_steps is None:
        # num_processes는 DDP 사용 시 GPU 수, TP/PP 등 사용 시 그에 맞게 조정됨
        # len(train_dataloader)는 이미 accelerator.prepare 이후의 길이(샤딩된 길이)를 사용해야 함
        # 따라서, get_scheduler 호출 전에 accelerator.prepare가 호출되어야 이 계산이 정확함.
        # 여기서는 일단 플레이스홀더로 두고, prepare 이후에 재계산하는 로직을 따름.
        # 이 값은 스케줄러 초기화 시 사용되므로, prepare 이전에 계산해야 함.
        # len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes) # 이 계산은 prepare 전에는 부정확
        
        # get_scheduler에 필요한 num_training_steps는 전체 학습 스텝 수 (모든 프로세스, 모든 에폭, gradient_accumulation 고려)
        # Accelerator는 내부적으로 num_processes를 알고 있으므로, 1개 프로세스 기준 스텝 수를 전달하면 됨.
        # 하지만, get_scheduler는 전체 스텝 수를 기대하는 경우가 많으므로, num_processes를 곱해주는 것이 안전할 수 있음.
        # Diffusers의 get_scheduler는 num_training_steps에 (num_epochs * num_update_steps_per_epoch * num_processes)를 기대
        
        # 먼저 1개 프로세스 기준 1 에폭당 업데이트 스텝 수 계산
        # len(train_dataloader)는 accelerator.prepare 이후 변경될 수 있으므로 주의
        # 여기서는 최초 dataloader 길이를 기준으로 계산하고, prepare 이후 다시 조정
        num_update_steps_per_epoch_approx = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler_approx = args.num_train_epochs * num_update_steps_per_epoch_approx
        # num_processes를 곱하는 것은 Accelerator가 내부적으로 처리해주길 기대하거나, 
        # 혹은 명시적으로 곱해줘야 할 수 있음. Diffusers는 곱해진 값을 기대.
        # 여기서는 Accelerator 내부 로직을 믿고, num_processes를 곱하지 않고, lr_scheduler.step() 시점에서 처리되도록 함.
        # 또는, Diffusers 예제처럼 명시적으로 num_processes를 곱함.
        num_training_steps_for_scheduler_approx_global = num_training_steps_for_scheduler_approx * accelerator.num_processes
        
    else:
        # max_train_steps가 주어지면, 해당 값을 전체 스텝 수로 사용
        num_training_steps_for_scheduler_approx_global = args.max_train_steps * accelerator.num_processes

    # 학습률 스케줄러 설정
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, # 웜업 스텝 (모든 프로세스 합산)
        num_training_steps=num_training_steps_for_scheduler_approx_global, # 총 학습 스텝 (모든 프로세스 합산)
        num_cycles=args.lr_num_cycles, # 코사인 스케줄러의 주기 수
        power=args.lr_power, # 다항식 스케줄러의 차수
    )
    logger.info(f"Initialized LR scheduler: {args.lr_scheduler} with {args.lr_warmup_steps * accelerator.num_processes} warmup steps and {num_training_steps_for_scheduler_approx_global} total training steps.")

    # Accelerator로 학습 구성 요소 준비
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Called accelerator.prepare() on model, optimizer, dataloader, and LR scheduler.")

    # 데이터로더 길이 변경에 따른 총 학습 스텝 수 재계산
    # accelerator.prepare 이후 train_dataloader의 길이는 샤딩된 길이로 변경됨
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"max_train_steps not provided, calculated as: {args.max_train_steps} (epochs: {args.num_train_epochs}, steps_per_epoch: {num_update_steps_per_epoch})")

    # 총 학습 에폭 수 재계산
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(f"Recalculated num_train_epochs: {args.num_train_epochs}")

    # 트래커 초기화 (메인 프로세스에서만)
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args)) # args를 복사하여 사용
        # TensorBoard는 리스트 타입의 설정을 처리하지 못하므로 제거
        if "validation_prompt" in tracker_config: tracker_config.pop("validation_prompt")
        if "validation_image" in tracker_config: tracker_config.pop("validation_image")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        logger.info(f"Initialized trackers (e.g., TensorBoard, WandB) with project name: {args.tracker_project_name}")

    # 학습 시작!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}") # 원본 데이터셋 크기 (샤딩 전)
    logger.info(f"  Num batches each epoch (per process) = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # 텍스트 인코딩을 위한 파이프라인 (VAE, Transformer는 사용 안 함)
    # 학습 루프 내에서 GPU/CPU로 이동시키며 사용
    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        transformer=None, # Transformer 불필요
        vae=None,         # VAE 불필요
        torch_dtype=weight_dtype # 정확도 설정
    )
    logger.info("Created text encoding pipeline.")

    # 체크포인트에서 학습 재개 로직
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else: # "latest"인 경우 가장 최근 체크포인트 검색
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None or not os.path.exists(os.path.join(args.output_dir, path)):
            logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' not found or path invalid. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            resume_path = os.path.join(args.output_dir, path)
            logger.info(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path) # Accelerator 상태 로드
            global_step = int(path.split("-")[1]) # 체크포인트 이름에서 global_step 복원
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed training from global_step: {global_step}, epoch: {first_epoch + 1}")
    else:
        initial_global_step = 0

    # WandB 사용 시 데이터셋 샘플 로깅
    if accelerator.is_main_process and args.report_to == "wandb" and args.log_dataset_samples:
        logger.info("Logging some dataset samples to WandB.")
        # ... (wandb 데이터셋 샘플 로깅 코드 - 기존 코드와 유사하게 유지)
        # 이 부분은 train_dataloader가 prepare된 이후에 실행되어야 함
        formatted_images_log = []
        formatted_control_images_log = []
        all_prompts_log = []
        
        # 데이터로더에서 몇 개 배치만 가져와서 로깅
        log_counter = 0
        max_log_samples = 16 # 로깅할 최대 샘플 수
        
        for i, batch_log in enumerate(train_dataloader):
            if log_counter >= max_log_samples:
                break
            
            images_log = (batch_log["pixel_values"] + 1) / 2 # [-1, 1] -> [0, 1]
            control_images_log = (batch_log["conditioning_pixel_values"] + 1) / 2
            prompts_log = batch_log["captions"]

            for img_log, control_img_log, prompt_log in zip(images_log, control_images_log, prompts_log):
                if log_counter >= max_log_samples:
                    break
                formatted_images_log.append(img_log)
                formatted_control_images_log.append(control_img_log)
                all_prompts_log.append(prompt_log)
                log_counter +=1
        
        logged_artifacts = []
        for img_l, control_img_l, prompt_l in zip(formatted_images_log, formatted_control_images_log, all_prompts_log):
            logged_artifacts.append(wandb.Image(control_img_l, caption="Conditioning Sample"))
            logged_artifacts.append(wandb.Image(img_l, caption=f"Target Sample: {prompt_l}"))

        try:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
            wandb_tracker.log({"dataset_samples": logged_artifacts})
            logger.info(f"Logged {len(logged_artifacts)//2} dataset samples to WandB.")
        except Exception as e:
            logger.warning(f"Failed to log dataset samples to WandB: {e}")


    # 학습 진행 표시줄 설정
    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process, # 메인 프로세스에서만 표시
    )

    # 시그마 값 계산 함수 (Flow Matching 용)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas_val = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps_val = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps_val = timesteps.to(accelerator.device)
        
        # timesteps에 해당하는 인덱스 찾기
        step_indices = torch.searchsorted(schedule_timesteps_val, timesteps_val, right=True) -1
        # Clamp indices to be within the valid range for sigmas_val
        step_indices = torch.clamp(step_indices, 0, len(sigmas_val) - 1)

        sigma = sigmas_val[step_indices].flatten()
        while len(sigma.shape) < n_dim: # 차원 확장
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None # 검증 이미지 로그 저장 변수

    # 학습 루프 시작
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train() # 모델을 학습 모드로 설정
        for step, batch in enumerate(train_dataloader):
            # 그래디언트 누적 컨텍스트
            with accelerator.accumulate(flux_transformer):
                # 1. VAE를 사용하여 이미지 인코딩 (픽셀 -> 잠재 공간)
                if not args.offload or vae.device != accelerator.device: # 오프로드 설정 또는 VAE가 CPU에 있을 경우
                    vae.to(accelerator.device) # VAE를 GPU로 이동
                
                pixel_latents = encode_images(batch["pixel_values"], vae, weight_dtype)
                control_latents = encode_images(batch["conditioning_pixel_values"], vae, weight_dtype)
                
                if args.offload: # VAE 오프로드 설정 시 CPU로 이동
                    vae.cpu()
                    logger.debug("Offloaded VAE to CPU.")

                bsz = pixel_latents.shape[0] # 배치 크기

                # 2. Flow Matching을 위한 타임스텝 및 노이즈 샘플링
                # 가중치 방식에 따라 타임스텝 샘플링 밀도 계산
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=accelerator.device # u 텐서 생성 장치 명시
                )
                # u 값을 사용하여 타임스텝 인덱스 샘플링
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                # 인덱스가 범위를 벗어나지 않도록 clamp
                indices = torch.clamp(indices, 0, len(noise_scheduler_copy.timesteps) - 1)
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                
                # 노이즈 생성
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                # 노이즈 추가 (Flow Matching 식)
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # 3. 제어 잠재 벡터와 노이즈 추가된 잠재 벡터 결합
                # 채널 차원을 따라 결합 (cat)
                concatenated_noisy_model_input = torch.cat([noisy_model_input, control_latents], dim=1)

                # FLUX 모델 입력을 위한 패킹 (2x2 패치로 묶음)
                packed_noisy_model_input = FluxControlPipeline._pack_latents(
                    concatenated_noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=concatenated_noisy_model_input.shape[1],
                    height=concatenated_noisy_model_input.shape[2],
                    width=concatenated_noisy_model_input.shape[3],
                )

                # RoPE (Rotary Positional Embedding)를 위한 이미지 ID 준비
                latent_image_ids = FluxControlPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated_noisy_model_input.shape[2] // 2, # 패킹된 높이
                    concatenated_noisy_model_input.shape[3] // 2, # 패킹된 너비
                    accelerator.device,
                    weight_dtype,
                )

                # 4. 텍스트 인코딩
                captions = batch["captions"]
                if not args.offload or text_encoding_pipeline.device != accelerator.device:
                     text_encoding_pipeline.to(accelerator.device) # GPU로 이동
                
                with torch.no_grad(): # 텍스트 인코더는 학습하지 않음
                    # proportion_empty_prompts 확률로 빈 프롬프트 사용 (CFG 학습에 도움)
                    if args.proportion_empty_prompts > 0 and random.random() < args.proportion_empty_prompts:
                        # 빈 캡션 생성 (실제로는 특정 빈 임베딩을 사용해야 할 수 있음)
                        # Diffusers의 encode_prompt는 None이나 빈 문자열을 적절히 처리
                        current_captions = [""] * bsz 
                        logger.debug("Using empty prompts for this batch.")
                    else:
                        current_captions = captions

                    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                        prompt=current_captions, # 캡션 사용
                        prompt_2=None,         # FLUX는 prompt_2를 사용하지 않음 (T5가 주 인코더)
                        device=accelerator.device, # 명시적 장치 설정
                        num_images_per_prompt=1, # 이미지당 프롬프트 수
                        max_sequence_length=flux_transformer.config.max_text_len # Transformer 설정 따름
                    )
                
                if args.offload: # 텍스트 인코딩 파이프라인 오프로드
                    text_encoding_pipeline.to("cpu")
                    logger.debug("Offloaded text encoding pipeline to CPU.")

                # 가이던스 벡터 준비 (FLUX 모델 설정에 따라)
                if unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale, # 학습 시 고정된 값 사용 또는 스케줄링 가능
                        device=accelerator.device, # noisy_model_input.device 대신 accelerator.device 사용
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # 5. 모델 예측 (Flux Transformer Forward Pass)
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000, # 타임스텝 정규화 (0~1 범위)
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds, # CLIP pooler 출력
                    encoder_hidden_states=prompt_embeds,      # T5 출력
                    txt_ids=text_ids,                         # 텍스트 ID (RoPE 용)
                    img_ids=latent_image_ids,                 # 이미지 ID (RoPE 용)
                    return_dict=False,
                )[0]
                
                # 예측된 잠재 벡터 언패킹
                model_pred = FluxControlPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor, # 원본 VAE 잠재 공간 크기
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # 6. 손실 계산 (Flow Matching Loss)
                # 타겟: noise - original_latents (노이즈에서 원본 잠재 벡터를 뺀 값)
                target = noise - pixel_latents
                
                # 손실 가중치 계산 (SD3 스타일)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # MSE 손실 계산
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1, # 각 샘플별 손실 계산
                )
                loss = loss.mean() # 배치 평균 손실

                # 7. 역전파 및 옵티마이저 스텝
                accelerator.backward(loss) # 그래디언트 계산

                if accelerator.sync_gradients: # 그래디언트 동기화 시점에 클리핑 수행
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm) # 그래디언트 클리핑
                
                optimizer.step() # 옵티마이저 스텝 (가중치 업데이트)
                lr_scheduler.step() # 학습률 스케줄러 스텝
                optimizer.zero_grad() # 그래디언트 초기화

            # 그래디언트 동기화 및 스텝 업데이트 (Accelerator가 내부적으로 처리)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 체크포인트 저장 (메인 프로세스에서만 또는 DeepSpeed 사용 시 모든 장치에서)
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # 체크포인트 수 제한 로직
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing old checkpoints: {', '.join(removing_checkpoints)}")
                                for ckpt_to_remove in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt_to_remove))
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path) # Accelerator 상태 저장 (모델, 옵티마이저, 스케줄러 등)
                        logger.info(f"Saved checkpoint to {save_path}")

                # 검증 실행 (메인 프로세스에서만)
                if accelerator.is_main_process:
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        if args.offload: # 검증 전 모델들을 다시 GPU로
                            if vae.device.type == 'cpu': vae.to(accelerator.device)
                            if text_encoding_pipeline.device.type == 'cpu': text_encoding_pipeline.to(accelerator.device)

                        image_logs = log_validation( # 생성된 이미지 로깅
                            flux_transformer=flux_transformer, # 학습 중인 모델 사용
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )
                        logger.info(f"Logged validation images at step {global_step}")

            # 로그 기록
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps: # 최대 학습 스텝 도달 시 종료
                break
        
        if global_step >= args.max_train_steps:
            break

    # 학습 종료 후 처리
    accelerator.wait_for_everyone() # 모든 프로세스 동기화

    if accelerator.is_main_process:
        flux_transformer_unwrapped = unwrap_model(flux_transformer) # 최종 모델 언래핑
        
        # 저장 전 float32로 업캐스팅 옵션
        if args.upcast_before_saving:
            logger.info("Upcasting final model to float32 before saving.")
            flux_transformer_unwrapped.to(torch.float32)

        # 최종 학습된 Transformer 모델 저장 (Hugging Face PretrainedModel 형식)
        flux_transformer_unwrapped.save_pretrained(os.path.join(args.output_dir, "transformer"))
        logger.info(f"Saved final trained FluxTransformer2DModel to {os.path.join(args.output_dir, 'transformer')}")
        
        # 사용한 모델 및 파이프라인 메모리에서 해제
        del flux_transformer
        del flux_transformer_unwrapped # 명시적 삭제
        if 'text_encoding_pipeline' in locals(): del text_encoding_pipeline
        if 'vae' in locals(): del vae
        free_memory() # 가비지 컬렉션 및 캐시 비우기
        logger.info("Cleaned up models from memory.")

        # 최종 검증 실행
        if args.validation_prompt is not None:
            logger.info("Running final validation on the saved model.")
            image_logs = log_validation(
                flux_transformer=None, # 저장된 모델을 사용하므로 None 전달
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype, # 최종 모델 로드 시 사용할 dtype
                step=global_step,
                is_final_validation=True, # 최종 검증 플래그
            )

        # Hugging Face Hub에 업로드
        if args.push_to_hub:
            logger.info(f"Pushing model to Hub repository: {repo_id}")
            save_model_card( # 모델 카드 생성 및 저장
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
                args=args,
            )
            upload_folder( # 모델 파일 및 카드 업로드
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*"], # 체크포인트는 제외
            )
            logger.info(f"Successfully pushed model to Hub: {repo_id}")

    accelerator.end_training() # Accelerator 학습 종료
    logger.info("Training finished.")


if __name__ == "__main__":
    # 명령줄에서 YAML 설정 파일 경로를 받음
    parser = argparse.ArgumentParser(description="FLUX Control Training with YAML Config")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the YAML configuration file (e.g., train_control_flux_config.yaml)"
    )
    # python train_control_flux_custom.py --config path/to/your_config.yaml
    cli_args = parser.parse_args()
    
    main(cli_args.config)
