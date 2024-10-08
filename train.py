import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import get_peft_model_state_dict

from models import EnvironmentLight, ToneMapping
from utils import load_virtual_object_from_blend, create_plane, generate_concept_images
from rendering import setup_renderer, compute_visibility_mask
from diffusion import personalize_diffusion_model, lds_loss
from loss import consistency_loss, regularization_loss, fuse_environment_maps
from diffusers import StableDiffusionPipeline,DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from pytorch3d.structures import join_meshes_as_scene

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from lora_utils import inject_trainable_LoRA, fuse_LoRA_into_linear, unfreeze_all_LoRA_layers, ATTENTION_MODULES

from pytorch3d.renderer import DirectionalLights

# Load configuration
config = OmegaConf.load('config.yaml')

# Device configuration
device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

# Load background image
background_image = Image.open(config.background_image_path).convert('RGB')
bg_transform = T.Compose([
    T.Resize((config.image_size, config.image_size)),
    T.ToTensor(),
])
Ibg = bg_transform(background_image).unsqueeze(0).to(device)

# Load virtual object mesh
virtual_object_mesh = load_virtual_object_from_blend(config.blend_file_path, config.object_name, device)

# Create plane
plane_mesh = create_plane(size=config.plane_size, device=device)

# Create scene mesh
scene_mesh = join_meshes_as_scene([virtual_object_mesh, plane_mesh])

# Setup renderer
renderer = setup_renderer(config, device)

# Initialize environment lights and tone mapping
env_light_fg = EnvironmentLight(config.num_lobes, device)
env_light_shadow = EnvironmentLight(config.num_lobes, device)
tone_mapping_fg = ToneMapping(num_bins=config.num_bins, device=device)
tone_mapping_shadow = ToneMapping(num_bins=config.num_bins, device=device)

# Setup optimizer
params = (list(env_light_fg.parameters()) + list(env_light_shadow.parameters()) +
          list(tone_mapping_fg.parameters()) + list(tone_mapping_shadow.parameters()))
optimizer = optim.Adam(params, lr=config.learning_rate)

# Load and personalize diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    requires_safety_checker=False
) 
pipe.enable_attention_slicing()  # For memory efficiency


# Initialize Accelerator
accelerator = Accelerator()
pipe,env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow, optimizer = accelerator.prepare(
    pipe,env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow, optimizer
)
concept_images = generate_concept_images(pipe, config.num_concept_images, config.concept_image_prompt).to(device)
personalized_pipe = personalize_diffusion_model(config, bg_transform, device)

# Create directories
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.image_output_dir, exist_ok=True)
os.makedirs(config.lora_weights_dir, exist_ok=True)

# Define checkpoint functions
def save_checkpoint(iteration, env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow, optimizer, personalized_pipe):
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt")
    torch.save({
        "model_fg": env_light_fg.state_dict(),
        "model_shadow": env_light_shadow.state_dict(),
        "tone_mapping_fg": tone_mapping_fg.state_dict(),
        "tone_mapping_shadow": tone_mapping_shadow.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {iteration}")
    
    # Save LoRA weights
    lora_state_dict = {}
    for name, module in personalized_pipe.unet.named_modules():
        if 'lora_' in name:
            lora_state_dict[name] = module.state_dict()
    
    if lora_state_dict:
        lora_path = os.path.join(config.lora_weights_dir, f"lora_weights_{iteration}.pt")
        torch.save(lora_state_dict, lora_path)
        print(f"LoRA weights saved at iteration {iteration}")
    else:
        print("No LoRA weights found to save.")

def load_checkpoint():
    checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not checkpoints:
        return 0
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_path = os.path.join(config.checkpoint_dir, latest_checkpoint)
    loaded_state = torch.load(checkpoint_path, map_location=device)
    
    env_light_fg.load_state_dict(loaded_state["model_fg"])
    env_light_shadow.load_state_dict(loaded_state["model_shadow"])
    tone_mapping_fg.load_state_dict(loaded_state["tone_mapping_fg"])
    tone_mapping_shadow.load_state_dict(loaded_state["tone_mapping_shadow"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    
    iteration = loaded_state["iteration"]
    lora_path = os.path.join(config.lora_weights_dir, f"lora_weights_{iteration}.pt")
    if os.path.exists(lora_path):
        lora_state_dict = torch.load(lora_path, map_location=device)
        personalized_pipe.unet.load_state_dict(lora_state_dict, strict=False)
        print(f"LoRA weights loaded from iteration {iteration}")
    
    start_iteration = iteration + 1
    print(f"Resumed from checkpoint at iteration {start_iteration - 1}")
    return start_iteration


def check_lora_application(model):
    lora_modules = [module for name, module in model.named_modules() if 'lora_' in name]
    if lora_modules:
        print(f"Found {len(lora_modules)} LoRA modules in the model.")
        for module in lora_modules[:5]:  # Print details of first 5 LoRA modules
            print(f"✅ LoRA module: {module}")
    else:
        print("❌ No LoRA modules found in the model.")

# Call this function before starting the training loop
check_lora_application(personalized_pipe.unet)

def save_icomp_image(Icomp, iteration):
    output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(output_image).save(os.path.join(config.image_output_dir, f'Icomp_iteration_{iteration}.png'))


# Constants for regularization
lambda_consistency = 0.03
lambda_reg = 0.01

# Optimization loop
start_iteration = load_checkpoint()
for iteration in range(start_iteration, config.num_iterations):
    with accelerator.accumulate(env_light_fg):
        optimizer.zero_grad()
    

        # Rendering and compositing
        fg_light_direction, fg_light_color = env_light_fg.get_aggregated_light()
        shadow_light_direction, shadow_light_color = env_light_shadow.get_aggregated_light()

        # Get light directions from EnvironmentLight        
        fg_lights =  DirectionalLights(device=device, direction=fg_light_direction, ambient_color=fg_light_color)
        shadow_lights = DirectionalLights(device=device, direction=shadow_light_direction, ambient_color=shadow_light_color)

        # Render foreground        
        Ifg = renderer(scene_mesh, lights=fg_lights)
        Ifg = Ifg[..., :3].permute(0, 3, 1, 2).to(device)

        # Render shadow        
        shadow_render = renderer(plane_mesh, lights=shadow_lights)
        shadow_render = shadow_render[..., :3].permute(0, 3, 1, 2)
        
        bg_render = renderer(plane_mesh, lights=fg_lights)
        bg_render = bg_render[..., :3].permute(0, 3, 1, 2)
        
        beta_shadow = shadow_render / (bg_render + 1e-6)

        # Apply tone mapping        
        Ifg_tone = tone_mapping_fg(Ifg)
        beta_shadow_tone = tone_mapping_shadow(beta_shadow)

        # Compute visibility mask        
        V = compute_visibility_mask(renderer, scene_mesh, plane_mesh)
        

        # Print shapes for debugging
        print(f"V shape: {V.shape}")
        print(f"beta_shadow_tone shape: {beta_shadow_tone.shape}")
        print(f"Ibg shape: {Ibg.shape}")
        print(f"Ifg_tone shape: {Ifg_tone.shape}")

        # Ensure all tensors have the same number of dimensions
        V = V.unsqueeze(1) if V.dim() == 3 else V

        
        beta_shadow_tone = beta_shadow_tone.unsqueeze(1) if beta_shadow_tone.dim() == 3 else beta_shadow_tone
        Ibg = Ibg.unsqueeze(0) if Ibg.dim() == 3 else Ibg
        Ifg_tone = Ifg_tone.unsqueeze(0) if Ifg_tone.dim() == 3 else Ifg_tone

        # Ensure all tensors have the same spatial dimensions
        target_size = (Ibg.shape[2], Ibg.shape[3])
        V = F.interpolate(V, size=target_size, mode='nearest')
        beta_shadow_tone = F.interpolate(beta_shadow_tone, size=target_size, mode='bilinear', align_corners=False)
        Ifg_tone = F.interpolate(Ifg_tone, size=target_size, mode='bilinear', align_corners=False)



       
        # Composite  
        Icomp = (1 - V) * beta_shadow_tone * Ibg + V * Ifg_tone
        Icomp = Icomp.to(device)
        pipe.unet = pipe.unet.to(device)
        pipe.vae = pipe.vae.to(device)
        pipe.text_encoder = pipe.text_encoder.to(device)

        print(f"Icomp shape: {Icomp.shape}")
        # Compute LDS loss
        prompt = "A photo of a car in a realistic scene"
        t = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,), device=device)
        loss = lds_loss(pipe, personalized_pipe, Icomp, prompt, t)
        
        # Add regularization terms
        loss += lambda_consistency * consistency_loss(env_light_fg, env_light_shadow)
        loss += lambda_reg * regularization_loss(env_light_shadow)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Optimizer step
        optimizer.step()
        
        save_icomp_image(Icomp, iteration)
    
    fuse_environment_maps(env_light_fg, env_light_shadow, iteration / config.num_iterations)
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration} Loss: {loss.item()}")
        save_checkpoint(iteration, env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow, optimizer, personalized_pipe)

        save_icomp_image(Icomp, iteration)

# Save final results
save_icomp_image(Icomp, config.num_iterations - 1)
save_checkpoint(config.num_iterations - 1, env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow, optimizer, personalized_pipe)
