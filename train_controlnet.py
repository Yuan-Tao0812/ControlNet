import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from accelerate import Accelerator
from huggingface_hub import login
from torchvision import transforms

# 登录 huggingface（首次执行需要输入 token）
login()

# 设置路径
image_dir = "/content/drive/MyDrive/ControlNet_train/images"
mask_dir = "/content/drive/MyDrive/ControlNet_train/masks"
save_dir = "/content/drive/MyDrive/controlnet_finetuned"

# 数据读取
def load_dataset(image_dir, mask_dir, max_samples=2000):
    images, masks = [], []
    file_names = sorted(os.listdir(image_dir))[:max_samples]
    for fname in file_names:
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, base + ".png")
        if os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)
    return Dataset.from_dict({"image": images, "condition": masks})

ds = load_dataset(image_dir, mask_dir)

# 加载模型
# ControlNet 语义分割模型路径
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_seg",
    torch_dtype=torch.float16
)

# Stable Diffusion v1.5 基础模型路径（从官方模型库加载）
vae = AutoencoderKL.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
)

# 加 LoRA 到 UNet
def add_lora_to_unet(unet):
    for _, module in unet.named_modules():
        if hasattr(module, "set_processor") and hasattr(module, "to_q"):
            module.set_processor(LoRAAttnProcessor())
    return unet

unet = add_lora_to_unet(unet)

# 优化器
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)

# 加速器
accelerator = Accelerator()
unet, optimizer = accelerator.prepare(unet, optimizer)

# 预处理函数
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    condition = Image.open(example["condition"]).convert("RGB")

    # image = transform(image).to(dtype=torch.float16)
    # condition = transform(condition).to(dtype=torch.float16)
    image = transform(image).numpy().astype("float32").tolist()
    condition = transform(condition).numpy().astype("float32").tolist()

    return {
        "pixel_values": image,
        "conditioning_pixel_values": condition
    }


ds = ds.map(preprocess)
print("预处理")
ds = ds.shuffle(seed=42)
ds.set_format(type="torch", columns=["pixel_values", "conditioning_pixel_values"])
dataloader = DataLoader(ds, batch_size=2)

# 噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
print("准备进入训练循环")

# 训练 loop
unet.train()
controlnet.train()
batch = next(iter(dataloader))
print(type(batch["pixel_values"]), batch["pixel_values"].dtype)
print(type(batch["conditioning_pixel_values"]), batch["conditioning_pixel_values"].dtype)
for epoch in range(3):
    for i, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=accelerator.device)
        condition = batch["conditioning_pixel_values"].to(dtype=torch.float16, device=accelerator.device)

        # 编码为 latent
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        # 加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 预测噪声
        controlnet_output = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=None,
            controlnet_cond=condition
        )

        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=None,
            down_block_additional_residuals=controlnet_output.down_block_res_samples,
            mid_block_additional_residual=controlnet_output.mid_block_res_sample
        ).sample

        # 损失函数
        loss = F.mse_loss(noise_pred.float(), noise.float())

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(f"[Epoch {epoch} | Step {i}] Loss: {loss.item():.4f}")

# 保存模型
unet.save_pretrained(os.path.join(save_dir, "unet_lora"))
controlnet.save_pretrained(os.path.join(save_dir, "controlnet"))
