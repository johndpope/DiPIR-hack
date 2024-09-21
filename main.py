import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model
import random


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
# Replace 'virtual_object.obj' with your object file path
virtual_object_mesh = load_objs_as_meshes(['virtual_object.obj'], device=device)

# --------------------------
# Step 3: Define Proxy Geometry (Plane)
# --------------------------
def create_plane(size=5.0, device='cpu'):
    # Create a simple square plane in the XZ plane
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

    textures = TexturesUV(
        maps=torch.ones((1, 1, 1, 3), device=device),
        faces_uvs=faces.unsqueeze(0),
        verts_uvs=torch.zeros_like(verts[:, :2]).unsqueeze(0),
    )

    plane_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return plane_mesh

plane_mesh = create_plane(size=5.0, device=device)

# Combine the object mesh and the plane mesh into a scene
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
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Define blend parameters
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Create the renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=None,  # We'll set lights dynamically
        materials=Materials(device=device),
        blend_params=blend_params
    )
)

# --------------------------
# Step 7: Load and Personalize Diffusion Model with LoRA
# --------------------------
# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.enable_attention_slicing()  # For memory efficiency

# Implement LoRA for personalization
# Here we use a simplified version; in practice, use a LoRA library or implementation
def personalize_diffusion_model(pipe, target_image, concept_images, num_steps=1000):
    config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, config)
    
    # Freeze text encoder and VAE
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    # Combine target image and concept images
    all_images = torch.cat([target_image, concept_images], dim=0)
    
    # Create corresponding prompts
    target_prompt = "a scene in the style of sks rendering"
    concept_prompts = ["a photo of a car"] * len(concept_images)
    all_prompts = [target_prompt] + concept_prompts
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    
    # Prepare noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(pipe.scheduler.config.name)
    
    # Training loop
    for step in range(num_steps):
        # Randomly select an image and its corresponding prompt
        idx = torch.randint(0, len(all_images), (1,)).item()
        image = all_images[idx:idx+1]
        prompt = all_prompts[idx]
        
        # Perform a training step
        loss = train_step(pipe, image, prompt, noise_scheduler, pipe.text_encoder)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
    
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


def generate_concept_images(pipe, num_images=40, prompt="a photo of a car"):
    concept_images = []
    for _ in range(num_images):
        # Add random attributes to diversify the generated images
        attributes = ["red", "blue", "black", "white", "SUV", "sedan", "sports car"]
        random_attribute = random.choice(attributes)
        full_prompt = f"{prompt}, {random_attribute}"
        
        # Generate image
        with torch.no_grad():
            image = pipe(full_prompt).images[0]
        
        # Convert to tensor and normalize
        image = T.ToTensor()(image).unsqueeze(0)
        concept_images.append(image)
    
    return torch.cat(concept_images, dim=0)

# Generate concept images
concept_images = generate_concept_images(pipe)



# For simplicity, we'll proceed without personalization

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

# Personalize the diffusion model (assuming this has been done)
personalized_pipe = personalize_diffusion_model(pipe, Ibg, concept_images)

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
num_iterations = 500
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # Render foreground
    Ifg = renderer(scene_mesh, lights=env_light_fg())
    Ifg = Ifg.permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
    
    # Render shadow
    shadow_render = renderer(plane_mesh, lights=env_light_shadow())
    shadow_render = shadow_render.permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
    bg_render = renderer(plane_mesh, lights=env_light_fg())
    bg_render = bg_render.permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
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
    
    loss.backward()
    optimizer.step()
    
    # Progressively fuse environment maps
    fuse_environment_maps(env_light_fg, env_light_shadow, iteration / num_iterations)
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration} Loss: {loss.item()}")
        
        # Optionally save intermediate images
        output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        Image.fromarray(output_image).save(f'output_{iteration}.png')

# Save the final composite image
output_image = Icomp.detach().cpu().squeeze().permute(1, 2, 0).numpy()
output_image = (output_image * 255).astype(np.uint8)
Image.fromarray(output_image).save('final_output.png')