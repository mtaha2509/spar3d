"""
High-quality texture blending for multi-view reconstruction.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch.nn.functional as F

from spar3d.models.mesh import Mesh
from .view_processor import ViewData


@dataclass
class BlendingConfig:
    """Configuration for texture blending."""
    texture_size: int = 1024
    blur_radius: int = 2
    feather_radius: int = 5
    max_memory_chunk: int = 1000000  # Maximum number of vertices to process at once


class TextureBlender:
    def __init__(self, config: Optional[BlendingConfig] = None):
        """Initialize texture blender with configuration."""
        self.config = config or BlendingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def blend_textures(
        self,
        mesh: Mesh,
        views: List[ViewData],
        images: List[torch.Tensor]
    ) -> Mesh:
        """
        Blend textures from multiple views onto the mesh.
        
        Args:
            mesh: Base mesh to texture
            views: List of view data
            images: List of view images (can be PIL Images or tensors)
            
        Returns:
            Mesh with blended textures
        """
        try:
            # Generate UV coordinates if they don't exist
            if not hasattr(mesh, 'uv_coords'):
                print("Generating UV coordinates for mesh...")
                mesh.uv_coords = self._generate_uv_coords(mesh)
            
            # Initialize texture maps
            texture_maps = []
            weight_maps = []
            
            # Process each view
            for view, image in zip(views, images):
                # Convert image to tensor if needed
                if not isinstance(image, torch.Tensor):
                    # Convert PIL Image to tensor
                    img_array = np.array(image)
                    if img_array.ndim == 2:  # Grayscale
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[-1] == 4:  # RGBA
                        img_array = img_array[..., :3]  # Keep only RGB
                    # Normalize and convert to tensor
                    img_tensor = torch.from_numpy(img_array.astype(np.float32)).permute(2, 0, 1) / 255.0
                else:
                    img_tensor = image
                
                texture_map, weight_map = self._generate_view_texture(
                    mesh, view, img_tensor
                )
                texture_maps.append(texture_map)
                weight_maps.append(weight_map)

            # Blend textures
            final_texture = self._blend_texture_maps(texture_maps, weight_maps)
            
            # Apply to mesh
            mesh.texture = final_texture
            mesh.has_texture = True  # Mark mesh as textured
            
            return mesh

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                # Reduce texture size and retry
                old_size = self.config.texture_size
                self.config.texture_size = old_size // 2
                try:
                    result = self.blend_textures(mesh, views, images)
                    self.config.texture_size = old_size
                    return result
                except:
                    self.config.texture_size = old_size
                    raise
            raise e

    def _to_tensor(self, data, dtype=torch.float64):
        """Convert various data types to PyTorch tensor with specified dtype."""
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        elif hasattr(data, 'numpy'):  # For TrackedArray or similar
            return torch.from_numpy(np.array(data.numpy(), copy=True)).to(dtype=dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(np.array(data, copy=True)).to(dtype=dtype)
        else:
            try:
                # Try converting to numpy first
                return torch.from_numpy(np.array(data, copy=True)).to(dtype=dtype)
            except:
                raise TypeError(f"Cannot convert {type(data)} to tensor")

    def _generate_uv_coords(self, mesh: Mesh) -> torch.Tensor:
        """Generate UV coordinates for mesh using spherical projection."""
        vertices = self._to_tensor(mesh.vertices).to(self.device)
        
        # Center the mesh
        center = vertices.mean(dim=0, keepdim=True)
        centered_verts = vertices - center
        
        # Compute spherical coordinates
        x, y, z = centered_verts.unbind(-1)
        r = torch.sqrt(x*x + y*y + z*z)
        theta = torch.arccos(z / (r + 1e-8))  # Add epsilon to avoid division by zero
        phi = torch.arctan2(y, x)
        
        # Convert to UV coordinates
        u = (phi + torch.pi) / (2 * torch.pi)
        v = theta / torch.pi
        
        return torch.stack([u, v], dim=-1)

    def _generate_view_texture(
        self,
        mesh: Mesh,
        view: ViewData,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate texture map for a single view."""
        # Process vertices in chunks to manage memory
        vertices = mesh.vertices
        chunk_size = self.config.max_memory_chunk
        
        texture_map = torch.zeros(
            (self.config.texture_size, self.config.texture_size, 3),
            device=self.device,
            dtype=torch.float64
        )
        weight_map = torch.zeros(
            (self.config.texture_size, self.config.texture_size),
            device=self.device,
            dtype=torch.float64
        )
        
        # Ensure UV coordinates exist
        if not hasattr(mesh, 'uv_coords'):
            mesh.uv_coords = self._generate_uv_coords(mesh)
        
        for i in range(0, len(vertices), chunk_size):
            chunk_verts = vertices[i:i + chunk_size]
            
            # Project vertices to view space
            proj_verts = self._project_vertices(chunk_verts, view.camera_params)
            
            # Calculate visibility and weights
            visibility, weights = self._compute_visibility_weights(
                proj_verts,
                view.view_direction,
                mesh.vertex_normals[i:i + chunk_size] if hasattr(mesh, 'vertex_normals') else None
            )
            
            # Sample image colors
            colors = self._sample_image_colors(proj_verts, image)
            
            # Accumulate to texture map
            self._accumulate_texture(
                texture_map,
                weight_map,
                colors,
                weights * visibility,
                mesh.uv_coords[i:i + chunk_size]
            )
            
            # Clear temporary tensors
            del proj_verts, visibility, weights, colors
            torch.cuda.empty_cache()
        
        return texture_map, weight_map

    def _project_vertices(
        self,
        vertices: torch.Tensor,
        camera_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Project vertices to view space."""
        # Convert vertices and camera parameters to tensors if needed
        vertices_tensor = self._to_tensor(vertices).to(self.device)
        extrinsic = self._to_tensor(camera_params['extrinsic']).to(self.device)
        intrinsic = self._to_tensor(camera_params['intrinsic']).to(self.device)
        
        # Transform to camera space
        cam_verts = torch.einsum(
            'ij,bj->bi',
            extrinsic[:3, :3],
            vertices_tensor
        ) + extrinsic[:3, 3]
        
        # Project to image space
        proj_verts = torch.einsum(
            'ij,bj->bi',
            intrinsic,
            cam_verts
        )
        
        return proj_verts

    def _compute_visibility_weights(
        self,
        proj_verts: torch.Tensor,
        view_direction: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute visibility and blending weights."""
        # Convert inputs to tensors if needed
        view_direction_tensor = self._to_tensor(view_direction).to(self.device)
        
        if normals is not None:
            normals_tensor = self._to_tensor(normals).to(self.device)
            # Compute view-dependent weights
            view_dot = torch.einsum(
                'bi,i->b',
                normals_tensor,
                view_direction_tensor
            )
        else:
            # If no normals available, use simpler weighting
            view_dot = torch.ones(proj_verts.shape[0], device=self.device)
        
        # Visibility based on normal orientation
        visibility = (view_dot > 0).to(dtype=torch.float64)
        
        # Smooth falloff for grazing angles
        weights = torch.pow(torch.clamp(view_dot, 0, 1), 2).to(dtype=torch.float64)
        
        return visibility, weights

    def _sample_image_colors(
        self,
        proj_verts: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """Sample colors from image at projected vertex positions."""
        # Normalize coordinates to [-1, 1]
        coords = proj_verts[:, :2] / proj_verts[:, 2:3]
        coords = coords * 2 - 1
        
        # Ensure image is in the right format (CxHxW)
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            image_tensor = image
        else:
            # Assume HxWxC format, convert to CxHxW
            image_tensor = image.permute(2, 0, 1)
        
        # If image has 4 channels (RGBA), keep only RGB
        if image_tensor.shape[0] == 4:
            image_tensor = image_tensor[:3]
        
        # Convert to double for consistency
        image_double = image_tensor.to(dtype=torch.float64, device=self.device)
        
        # Sample using grid_sample
        colors = F.grid_sample(
            image_double.unsqueeze(0),
            coords.unsqueeze(0).unsqueeze(1),
            align_corners=True
        )
        
        # Reshape colors to (N, 3) format
        colors = colors.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        
        return colors

    def _accumulate_texture(
        self,
        texture_map: torch.Tensor,
        weight_map: torch.Tensor,
        colors: torch.Tensor,
        weights: torch.Tensor,
        uv_coords: torch.Tensor
    ):
        """Accumulate colors and weights to texture map."""
        # Convert UV coordinates to tensor if needed
        uv_coords_tensor = self._to_tensor(uv_coords).to(self.device)
        
        # Scale UV coordinates to texture resolution
        uv_pixels = (uv_coords_tensor * (self.config.texture_size - 1)).long()
        
        # Convert colors and weights to double
        colors = colors.to(dtype=torch.float64)
        weights = weights.to(dtype=torch.float64)
        
        # Ensure colors are in the right shape (N, 3)
        if colors.dim() == 3:  # If colors are in (3, N) format
            colors = colors.permute(1, 0)
        elif colors.dim() == 1:  # If colors are flattened
            colors = colors.view(-1, 3)
            
        # Ensure weights are the right shape
        weights = weights.view(-1)
        
        # Accumulate colors and weights
        for i in range(len(colors)):
            if weights[i] > 0:
                u, v = uv_pixels[i]
                if 0 <= u < self.config.texture_size and 0 <= v < self.config.texture_size:
                    weighted_color = colors[i] * weights[i]
                    texture_map[v, u] += weighted_color
                    weight_map[v, u] += weights[i]

    def _blend_texture_maps(
        self,
        texture_maps: List[torch.Tensor],
        weight_maps: List[torch.Tensor]
    ) -> torch.Tensor:
        """Blend multiple texture maps using weights."""
        # Stack maps
        textures = torch.stack(texture_maps, dim=0)
        weights = torch.stack(weight_maps, dim=0).unsqueeze(-1)
        
        # Normalize weights
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
        
        # Blend
        blended = (textures * weights).sum(dim=0)
        
        # Apply feathering
        if self.config.feather_radius > 0:
            kernel_size = 2 * self.config.feather_radius + 1
            blended = F.avg_pool2d(
                blended.permute(2, 0, 1).unsqueeze(0),
                kernel_size,
                stride=1,
                padding=self.config.feather_radius
            ).squeeze(0).permute(1, 2, 0)
        
        return blended 