import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model
import random
from pytorch3d.renderer import TexturesVertex
import bpy
import trimesh
import os
from diffusers import UNet2DConditionModel
from lora_utils import inject_trainable_LoRA, fuse_LoRA_into_linear, unfreeze_all_LoRA_layers, ATTENTION_MODULES
import os
from accelerate import Accelerator
from peft import get_peft_model_state_dict

# PyTorch3D for differentiable rendering
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, TexturesUV, PointLights, DirectionalLights,
    Materials, BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_scene


# Diffusers library for Stable Diffusion
from diffusers import StableDiffusionPipeline,DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import fnmatch
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
# from lora_utils import inject_trainable_lora



import diffusers
print("🎯 diffusers:",diffusers.__version__)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    

# --------------------------
# Step 1: Load Background Image
# --------------------------
# Load the background image
background_image = Image.open('multi_bg.jpg').convert('RGB')
bg_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])
Ibg = bg_transform(background_image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]

# --------------------------
# Step 2: Load Virtual Object Mesh
# --------------------------
# Load the virtual object mesh
# Function to load virtual object from .blend file
def load_virtual_object_from_blend(blend_file_path, object_name, device='cuda'):
    import bpy
    import numpy as np
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex

    # Open the .blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    # Find the object in the scene
    if object_name not in bpy.data.objects:
        raise ValueError(f"Object '{object_name}' not found in file '{blend_file_path}'")
    
    obj = bpy.data.objects[object_name]
    
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        raise ValueError(f"Object '{object_name}' is not a mesh")
    
    # Triangulate the mesh
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Get mesh data
    mesh = obj.data

    # Apply object transformations to vertices
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices], dtype=np.float32)
    
    # Extract faces (triangles)
    faces = [list(p.vertices) for p in mesh.polygons]

    # Verify that all faces are triangles
    if not all(len(face) == 3 for face in faces):
        raise ValueError("Not all faces are triangles after triangulation.")
    
    # Convert faces to a NumPy array
    faces = np.array(faces, dtype=np.int64)
    
    # Convert to PyTorch tensors
    verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # Create a Meshes object
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    
    # Add default white vertex colors as textures
    num_verts = pytorch3d_mesh.num_verts_per_mesh().item()
    verts_features = torch.ones((1, num_verts, 3), device=device)
    pytorch3d_mesh.textures = TexturesVertex(verts_features=verts_features)
    
    return pytorch3d_mesh



# --------------------------
# Step 3: Define Proxy Geometry (Plane)
# --------------------------
def create_plane(size=5.0, device='cpu'):
    verts = torch.tensor([
        [-size, 0, -size],
        [size, 0, -size],
        [size, 0, size],
        [-size, 0, size],
    ], device=device, dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], device=device, dtype=torch.int64)

    # Create a simple texture for the plane
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    plane_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return plane_mesh

# Create the plane
plane_mesh = create_plane(size=5.0, device=device)

# Usage in the main script
blend_file_path = "Television_01_4k/Television_01_4k.blend"  # Replace with the actual path
object_name = "Television_01"  # Replace with the actual object name in the .blend file
virtual_object_mesh = load_virtual_object_from_blend(blend_file_path, object_name, device=device)

# The rest of your script remains the same
scene_mesh = join_meshes_as_scene([virtual_object_mesh, plane_mesh])

# --------------------------
# Step 4: Define Optimizable Environment Lighting
# --------------------------
class EnvironmentLight(nn.Module):
    def __init__(self, num_lobes=32, device='cpu'):
        super(EnvironmentLight, self).__init__()
        self.num_lobes = num_lobes
        self.mu = nn.Parameter(torch.randn(num_lobes, 3, device=device))
        self.c = nn.Parameter(torch.ones(num_lobes, 3, device=device))
        self.sigma = nn.Parameter(torch.ones(num_lobes, device=device) * 0.5)

    def forward(self, directions):
        mu_normalized = F.normalize(self.mu, dim=-1)
        dot_product = torch.matmul(directions, mu_normalized.transpose(0, 1))
        gaussian = torch.exp(-2 * (1 - dot_product) / (self.sigma ** 2).unsqueeze(0))
        return torch.matmul(gaussian, self.c)

    def get_light_directions(self):
        return F.normalize(self.mu, dim=-1)
    

env_light = EnvironmentLight(device=device)

# --------------------------
# Step 5: Define Tone-Mapping Function
# --------------------------
class ToneMapping(nn.Module):
    def __init__(self, num_bins=5):
        super(ToneMapping, self).__init__()
        self.num_bins = num_bins
        self.widths = nn.Parameter(torch.ones(num_bins) / num_bins)
        self.heights = nn.Parameter(torch.linspace(0, 1, num_bins + 1))
        self.slopes = nn.Parameter(torch.ones(num_bins + 1))

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        bin_idx = torch.searchsorted(torch.cumsum(self.widths, 0), x)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        
        x_low = torch.gather(torch.cat([torch.zeros(1), torch.cumsum(self.widths, 0)[:-1]]), 0, bin_idx)
        x_high = x_low + torch.gather(self.widths, 0, bin_idx)
        y_low = torch.gather(self.heights[:-1], 0, bin_idx)
        y_high = torch.gather(self.heights[1:], 0, bin_idx)
        slope_low = torch.gather(self.slopes[:-1], 0, bin_idx)
        slope_high = torch.gather(self.slopes[1:], 0, bin_idx)
        
        t = (x - x_low) / (x_high - x_low)
        y = y_low + (y_high - y_low) * (slope_low * t ** 2 + 2 * t * (1 - t)) / ((slope_low + slope_high) * t ** 2 + 2 * (slope_low + slope_high - 2) * t + 2)
        return y

tone_mapping = ToneMapping()

# --------------------------
# Step 6: Set Up Differentiable Renderer
# --------------------------
# Set up camera parameters
cameras = FoVPerspectiveCameras(device=device)

# Define rasterization settings
# Set up renderer
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

cameras = FoVPerspectiveCameras(device=device)

# Define blend parameters
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=DirectionalLights(device=device),
        materials=Materials(device=device),
        blend_params=blend_params
    )
)




# --------------------------
# Step 7: Load and Personalize Diffusion Model with LoRA
# --------------------------
# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    requires_safety_checker=False
).to(device)
pipe.enable_attention_slicing()  # For memory efficiency

# Implement LoRA for personalization
# Here we use a simplified version; in practice, use a LoRA library or implementation

def personalize_diffusion_model(pipe, target_image, concept_images, num_steps=1000, device=None):
    # Move the entire pipeline to the specified device
    if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pipe = pipe.to(device)
    target_image = target_image.to(device)
    concept_images = concept_images.to(device)
    
    # Inject LoRA into the UNet
    inject_trainable_LoRA(
        model=pipe.unet,
        rank=4,
        scale=1.0,
        target_replace_modules=ATTENTION_MODULES
    )
    
    # Unfreeze LoRA layers
    unfreeze_all_LoRA_layers(pipe.unet)
    
    # Freeze other parameters
    for name, param in pipe.unet.named_parameters():
        if 'lora' not in name:
            param.requires_grad_(False)

    # Prepare optimizer (only optimize LoRA parameters)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=1e-4)

    # Prepare noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Training loop
    for step in range(num_steps):
        # Randomly select an image and its corresponding prompt
        idx = torch.randint(0, len(concept_images), (1,)).item()
        image = concept_images[idx:idx+1].to(device)
        prompt = "a photo of a car"

        # Encode the prompt
        text_input = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        encoder_hidden_states = pipe.text_encoder(text_input.input_ids)[0]

        # Prepare the image
        image = image.to(device=pipe.device, dtype=pipe.unet.dtype)

        # Encode the image into latent space
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    # After training, fuse LoRA weights into the main model
    fuse_LoRA_into_linear(pipe.unet, target_replace_modules=ATTENTION_MODULES)

    return pipe

def train_step(pipe, image, prompt, noise_scheduler, text_encoder):
    # Encode the prompt
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(image.device)
    encoder_hidden_states = text_encoder(text_input.input_ids)[0]

    # Prepare the image
    image = image.to(device=pipe.device, dtype=pipe.unet.dtype)
    
    # Encode the image into latent space
    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = latents * 0.18215

    # Sample noise to add to the latents
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device
    ).long()

    # Add noise to the latents according to the noise magnitude at each timestep
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Predict the noise residual
    noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Compute loss
    loss = F.mse_loss(noise_pred, noise)

    return loss



def lds_loss(pipe, personalized_pipe, image, prompt, t):
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
        
    text_embeddings = pipe.text_encoder(prompt)[0]
    
    noise_pred = personalized_pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_original = pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
    
    return F.mse_loss(noise_pred - noise_pred_original, noise)


def generate_concept_images(pipe, num_images=5, prompt="a photo of a car"):
    concept_images = []
    for _ in range(num_images):
        attributes = ["red", "blue", "black", "white", "SUV", "sedan", "sports car"]
        random_attribute = random.choice(attributes)
        full_prompt = f"{prompt}, {random_attribute}"
        
        with torch.no_grad():
            image = pipe(full_prompt, num_inference_steps=5).images[0]
        
        image = T.ToTensor()(image).unsqueeze(0)
        concept_images.append(image)
    
    return torch.cat(concept_images, dim=0).to(device)

# Generate concept images
concept_images = generate_concept_images(pipe).to(device) 



# --------------------------
# Step 8: Define Optimization Loop
# --------------------------
# Optimizable parameters
# Initialize separate environment maps for foreground and shadow
env_light_fg = EnvironmentLight(device=device)
env_light_shadow = EnvironmentLight(device=device)

# Initialize separate tone mapping for foreground and shadow
tone_mapping_fg = ToneMapping()
tone_mapping_shadow = ToneMapping()

# Optimizable parameters
params = (list(env_light_fg.parameters()) + list(env_light_shadow.parameters()) +
          list(tone_mapping_fg.parameters()) + list(tone_mapping_shadow.parameters()))

optimizer = optim.Adam(params, lr=1e-2)

pipe = pipe.to(device)
Ibg = Ibg.to(device)

# Personalize the diffusion model (assuming this has been done)
personalized_pipe = personalize_diffusion_model(pipe, Ibg, concept_images,1000,device)

# Constants for regularization
lambda_consistency = 0.03
lambda_reg = 0.01

def consistency_loss(env_fg, env_shadow):
    # Compute normalized luminance for both environment maps
    L_fg = torch.sum(env_fg(torch.ones_like(env_fg.mu)), dim=-1)  # [num_lobes]
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)  # [num_lobes]
    
    L_fg_norm = L_fg / L_fg.sum()
    L_shadow_norm = L_shadow / L_shadow.sum()
    
    # Compute KL divergence
    loss = F.kl_div(L_shadow_norm.log(), L_fg_norm, reduction='batchmean')
    
    return loss


def regularization_loss(env_shadow):
    # Compute luminance
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)  # [num_lobes]
    
    # Compute Cauchy loss in log-space
    loss = torch.log(1 + 2 * (L_shadow ** 2)).mean()
    
    return loss

def fuse_environment_maps(env_fg, env_shadow, ratio):
    # Compute luminance for both environment maps
    L_fg = torch.sum(env_fg(torch.ones_like(env_fg.mu)), dim=-1)  # [num_lobes]
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)  # [num_lobes]
    
    # Compute blending ratio
    r = (L_fg / L_fg.max()) * (L_shadow / (L_fg + L_shadow))
    
    # Compute fused luminance
    L_fused = (1 - r) * L_fg + r * L_shadow
    
    # Adjust foreground environment map
    scale_factor = L_fused / L_fg
    env_fg.c.data *= scale_factor.unsqueeze(-1)
    
    # Linearly interpolate between original and fused environment maps
    env_fg.c.data = ratio * env_fg.c.data + (1 - ratio) * env_shadow.c.data
    env_shadow.c.data = env_fg.c.data

def compute_visibility_mask(renderer, scene_mesh, background_mesh):
    # Render depth for the full scene
    scene_depth = renderer.rasterizer(scene_mesh)
    
    # Render depth for the background only
    bg_depth = renderer.rasterizer(background_mesh)
    
    # Create visibility mask
    V = (scene_depth.zbuf < bg_depth.zbuf).float()
    
    return V   


# Initialize the Accelerator
accelerator = Accelerator()

# Wrap the model, optimizer, and data loader (if any) with accelerator
env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow = accelerator.prepare(
    env_light_fg, env_light_shadow, tone_mapping_fg, tone_mapping_shadow
)
optimizer = accelerator.prepare(optimizer)


# Define the checkpoint directory, image output directory, and LoRA weights directory
checkpoint_dir = "./checkpoints"
image_output_dir = "./output_images"
lora_weights_dir = "./lora_weights"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(lora_weights_dir, exist_ok=True)

# Function to save checkpoint
def save_checkpoint(iteration):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
    accelerator.save({
        "model_fg": env_light_fg.state_dict(),
        "model_shadow": env_light_shadow.state_dict(),
        "tone_mapping_fg": tone_mapping_fg.state_dict(),
        "tone_mapping_shadow": tone_mapping_shadow.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {iteration}")
    
    # Save LoRA weights
    save_lora_weights(iteration)

# Function to save LoRA weights
def save_lora_weights(iteration):
    lora_state_dict = get_peft_model_state_dict(personalized_pipe.unet)
    lora_path = os.path.join(lora_weights_dir, f"lora_weights_{iteration}.pt")
    accelerator.save(lora_state_dict, lora_path)
    print(f"LoRA weights saved at iteration {iteration}")

# Function to load checkpoint (update to load LoRA weights as well)
def load_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not checkpoints:
        return 0  # Start from the beginning if no checkpoints found
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    loaded_state = accelerator.load(checkpoint_path)
    
    env_light_fg.load_state_dict(loaded_state["model_fg"])
    env_light_shadow.load_state_dict(loaded_state["model_shadow"])
    tone_mapping_fg.load_state_dict(loaded_state["tone_mapping_fg"])
    tone_mapping_shadow.load_state_dict(loaded_state["tone_mapping_shadow"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    
    # Load LoRA weights
    iteration = loaded_state["iteration"]
    lora_path = os.path.join(lora_weights_dir, f"lora_weights_{iteration}.pt")
    if os.path.exists(lora_path):
        lora_state_dict = accelerator.load(lora_path)
        personalized_pipe.unet.load_state_dict(lora_state_dict, strict=False)
        print(f"LoRA weights loaded from iteration {iteration}")
    
    start_iteration = iteration + 1
    print(f"Resumed from checkpoint at iteration {start_iteration - 1}")
    return start_iteration


# Function to save Icomp as an image
def save_icomp_image(Icomp, iteration):
    print("save_icomp_image")
    output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(output_image).save(os.path.join(image_output_dir, f'Icomp_iteration_{iteration}.png'))

# Optimization loop
num_iterations = 500
start_iteration = load_checkpoint()  # Load the latest checkpoint if it exists

for iteration in range(start_iteration, num_iterations):
    with accelerator.accumulate(env_light_fg):
        optimizer.zero_grad()
        
        # Get light directions from EnvironmentLight
        fg_light_directions = env_light_fg.get_light_directions()
        shadow_light_directions = env_light_shadow.get_light_directions()
        
        # Render foreground
        fg_lights = DirectionalLights(device=device, direction=fg_light_directions, ambient_color=env_light_fg(fg_light_directions))
        Ifg = renderer(scene_mesh, lights=fg_lights)
        Ifg = Ifg[..., :3].permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
        
        # Render shadow
        shadow_lights = DirectionalLights(device=device, direction=shadow_light_directions, ambient_color=env_light_shadow(shadow_light_directions))
        shadow_render = renderer(plane_mesh, lights=shadow_lights)
        shadow_render = shadow_render[..., :3].permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
        
        bg_lights = DirectionalLights(device=device, direction=fg_light_directions, ambient_color=env_light_fg(fg_light_directions))
        bg_render = renderer(plane_mesh, lights=bg_lights)
        bg_render = bg_render[..., :3].permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
        
        beta_shadow = shadow_render / (bg_render + 1e-6)
        
        # Apply tone mapping
        Ifg_tone = tone_mapping_fg(Ifg)
        beta_shadow_tone = tone_mapping_shadow(beta_shadow)
        
        # Compute visibility mask
        V = compute_visibility_mask(renderer, scene_mesh, plane_mesh)
        
        # Composite
        Icomp = (1 - V) * beta_shadow_tone * Ibg + V * Ifg_tone
        
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
    
    # Progressively fuse environment maps
    fuse_environment_maps(env_light_fg, env_light_shadow, iteration / num_iterations)
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration} Loss: {loss.item()}")
        
        # Save checkpoint
        save_checkpoint(iteration)
        
        # Optionally save intermediate images
        output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        Image.fromarray(output_image).save(f'output_{iteration}.png')


        
        
# Save the final composite image
output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
output_image = (output_image * 255).astype(np.uint8)
Image.fromarray(output_image).save('final_output.png')

# Save final checkpoint
save_checkpoint(num_iterations - 1)