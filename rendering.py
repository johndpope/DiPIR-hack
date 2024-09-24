from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, Materials, BlendParams
)

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

def compute_visibility_mask(renderer, scene_mesh, background_mesh):
    scene_depth = renderer.rasterizer(scene_mesh)
    bg_depth = renderer.rasterizer(background_mesh)
    V = (scene_depth.zbuf < bg_depth.zbuf).float()
    return V