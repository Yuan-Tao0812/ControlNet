import torch
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image

# 加载微调后的模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet="/content/drive/MyDrive/controlnet_finetuned/controlnet",
    unet="/content/drive/MyDrive/controlnet_finetuned/unet_lora",
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")

# 加载语义图进行推理
cond = Image.open("/content/drive/MyDrive/ControlNet_train/masks/0000002_00448_d_0000015.png").resize((512, 512))
prompt = "aerial drone view of people and cars on streets"

result = pipe(prompt=prompt, image=cond, num_inference_steps=20).images[0]
result.save("generated.png")
print("✅ 推理完成，保存为 generated.png")
