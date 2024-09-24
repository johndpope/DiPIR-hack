from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, Materials, BlendParams
)
from pytorch3d.structures import Meshes



def setup_renderer(config, device):
    cameras = FoVPerspectiveCameras(device=device)
    
    raster_settings = RasterizationSettings(
        image_size=config.raster_settings.image_size,
        blur_radius=config.raster_settings.blur_radius,
        faces_per_pixel=config.raster_settings.faces_per_pixel,
    )
    
    blend_params = BlendParams(
        sigma=config.blend_params.sigma,
        gamma=config.blend_params.gamma,
    )
    
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
    
    return renderer
def compute_visibility_mask(renderer, scene_mesh: Meshes, background_mesh: Meshes):
    print("Computing visibility mask...")
    
    print(f"Scene mesh: {scene_mesh}")
    print(f"Background mesh: {background_mesh}")
    
    # Rasterize both meshes
    scene_fragments = renderer.rasterizer(scene_mesh)
    bg_fragments = renderer.rasterizer(background_mesh)
    
    print(f"Scene fragments type: {type(scene_fragments)}")
    print(f"Background fragments type: {type(bg_fragments)}")
    
    # Get depth buffers
    scene_zbuf = scene_fragments.zbuf
    bg_zbuf = bg_fragments.zbuf
    
    print(f"Original scene_zbuf shape: {scene_zbuf.shape}")
    print(f"Original bg_zbuf shape: {bg_zbuf.shape}")
    
    # Ensure zbuf has the correct shape [batch_size, image_height, image_width, 1]
    if scene_zbuf.dim() == 3:
        scene_zbuf = scene_zbuf.unsqueeze(-1)
    if bg_zbuf.dim() == 3:
        bg_zbuf = bg_zbuf.unsqueeze(-1)
    
    print(f"Adjusted scene_zbuf shape: {scene_zbuf.shape}")
    print(f"Adjusted bg_zbuf shape: {bg_zbuf.shape}")
    
    # Compute visibility mask
    V = (scene_zbuf < bg_zbuf).float()
    print(f"Initial V shape: {V.shape}")
    
    # Additional shape adjustments
    V = V.squeeze(-1)  # Remove last dimension if it's singleton
    print(f"V shape after squeeze: {V.shape}")
    
    V = V.unsqueeze(1)  # Add channel dimension
    print(f"V shape after unsqueeze: {V.shape}")
    
    V = V.expand(-1, 3, -1, -1)  # Expand to 3 channels
    print(f"Final V shape: {V.shape}")
    
    print(f"V min: {V.min()}, V max: {V.max()}")
    
    return V
