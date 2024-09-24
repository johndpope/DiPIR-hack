import bpy
import numpy as np
import torch
import random
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import torchvision.transforms as T

def load_virtual_object_from_blend(blend_file_path, object_name, device='cuda'):
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    if object_name not in bpy.data.objects:
        raise ValueError(f"Object '{object_name}' not found in file '{blend_file_path}'")
    
    obj = bpy.data.objects[object_name]
    
    if obj.type != 'MESH':
        raise ValueError(f"Object '{object_name}' is not a mesh")
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices], dtype=np.float32)
    faces = np.array([list(p.vertices) for p in mesh.polygons], dtype=np.int64)
    
    verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    
    num_verts = pytorch3d_mesh.num_verts_per_mesh().item()
    verts_features = torch.ones((1, num_verts, 3), device=device)
    pytorch3d_mesh.textures = TexturesVertex(verts_features=verts_features)
    
    return pytorch3d_mesh

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

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    plane_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return plane_mesh

def generate_concept_images(pipe, num_images=5, prompt="a photo of a car"):
    concept_images = []
    attributes = ["red", "blue", "black", "white", "SUV", "sedan", "sports car"]
    
    for _ in range(num_images):
        random_attribute = random.choice(attributes)
        full_prompt = f"{prompt}, {random_attribute}"
        
        with torch.no_grad():
            image = pipe(full_prompt, num_inference_steps=5).images[0]
        
        image = T.ToTensor()(image).unsqueeze(0)
        concept_images.append(image)
    
    return torch.cat(concept_images, dim=0)