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
            images: List of view images as tensors
            
        Returns:
            Mesh with blended textures
        """
        try:
            # Initialize texture maps
            texture_maps = []
            weight_maps = []
            
            # Process each view
            for view, image in zip(views, images):
                texture_map, weight_map = self._generate_view_texture(
                    mesh, view, image
                )
                texture_maps.append(texture_map)
                weight_maps.append(weight_map)

            # Blend textures
            final_texture = self._blend_texture_maps(texture_maps, weight_maps)
            
            # Apply to mesh
            mesh.texture = final_texture
            
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
            return torch.from_numpy(data.numpy()).to(dtype=dtype)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype=dtype)
        else:
            try:
                # Try converting to numpy first
                return torch.from_numpy(np.array(data)).to(dtype=dtype)
            except:
                raise TypeError(f"Cannot convert {type(data)} to tensor")

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
        
        for i in range(0, len(vertices), chunk_size):
            chunk_verts = vertices[i:i + chunk_size]
            
            # Project vertices to view space
            proj_verts = self._project_vertices(chunk_verts, view.camera_params)
            
            # Calculate visibility and weights
            visibility, weights = self._compute_visibility_weights(
                proj_verts,
                view.view_direction,
                mesh.vertex_normals[i:i + chunk_size]
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
        normals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute visibility and blending weights."""
        # Convert inputs to tensors if needed
        normals_tensor = self._to_tensor(normals).to(self.device)
        view_direction_tensor = self._to_tensor(view_direction).to(self.device)
        
        # Compute view-dependent weights
        view_dot = torch.einsum(
            'bi,i->b',
            normals_tensor,
            view_direction_tensor
        )
        
        # Visibility based on normal orientation
        visibility = (view_dot > 0).to(dtype=torch.float64)
        
        # Smooth falloff for grazing angles
        weights = torch.pow(torch.clamp(view_dot, 0, 1), 2)
        
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
        
        # Convert image to double for consistency
        image_double = image.to(dtype=torch.float64, device=self.device)
        
        # Sample using grid_sample
        colors = F.grid_sample(
            image_double.unsqueeze(0),
            coords.unsqueeze(0).unsqueeze(1),
            align_corners=True
        )
        
        return colors.squeeze()

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
        
        # Accumulate colors and weights
        for i in range(len(colors)):
            if weights[i] > 0:
                u, v = uv_pixels[i]
                if 0 <= u < self.config.texture_size and 0 <= v < self.config.texture_size:
                    texture_map[v, u] += colors[i] * weights[i]
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