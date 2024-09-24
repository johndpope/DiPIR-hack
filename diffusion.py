import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from lora_utils import inject_trainable_LoRA, fuse_LoRA_into_linear, unfreeze_all_LoRA_layers, ATTENTION_MODULES


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

def lds_loss(pipe, personalized_pipe, image, prompt, t):
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
    
    text_embeddings = pipe.text_encoder(prompt)[0]
    
    noise_pred = personalized_pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_original = pipe.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
    
    return F.mse_loss(noise_pred - noise_pred_original, noise)