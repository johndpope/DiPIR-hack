import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from lora_utils import inject_trainable_LoRA, fuse_LoRA_into_linear, unfreeze_all_LoRA_layers, ATTENTION_MODULES


def personalize_diffusion_model(config, device):
    # Load the pre-trained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        config.diffusion_model_path,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.enable_attention_slicing()

    # Inject trainable LoRA modules into the model
    inject_trainable_LoRA(
        model=pipe.unet,
        rank=config.lora_rank,
        scale=config.lora_scale,
        target_replace_modules=ATTENTION_MODULES
    )

    # Unfreeze LoRA parameters for fine-tuning
    unfreeze_all_LoRA_layers(pipe.unet)

    # Freeze all other parameters
    for name, param in pipe.unet.named_parameters():
        if 'lora' not in name:
            param.requires_grad_(False)

    # Prepare the fine-tuning dataset
    # Generate synthetic images for concept preservation
    concept_images = generate_concept_images(
        pipe,
        num_images=config.num_concept_images,
        prompt=config.concept_image_prompt
    ).to(device)

    # Include the target scene image
    target_image = Image.open(config.background_image_path).convert('RGB')
    target_image = bg_transform(target_image).unsqueeze(0).to(device)

    # Combine images into a dataset
    fine_tuning_images = torch.cat([target_image, concept_images], dim=0)

    # Define the fine-tuning prompts
    prompts = [config.scene_style_token] * 1 + [config.concept_class_prompt] * config.num_concept_images

    # Prepare the optimizer
    optimizer = torch.optim.Adam(
        [p for p in pipe.unet.parameters() if p.requires_grad],
        lr=config.fine_tuning_learning_rate
    )

    # Fine-tune the model
    pipe.unet.train()
    num_epochs = config.num_fine_tuning_epochs
    for epoch in range(num_epochs):
        for img, prompt in zip(fine_tuning_images, prompts):
            img = img.unsqueeze(0)  # Add batch dimension
            # Encode the image
            latents = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            # Get text embeddings
            text_input = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embeddings = pipe.text_encoder(text_input.input_ids)[0]
            # Predict the noise residual
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            # Compute the loss
            loss = F.mse_loss(noise_pred, noise)
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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