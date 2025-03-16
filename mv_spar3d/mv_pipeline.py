"""
Main pipeline for multi-view 3D reconstruction using SPAR3D.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image
import os

from spar3d.models.mesh import Mesh
from .view_processor import ViewProcessor, ViewData
from .point_fusion import PointCloudFuser, FusionConfig
from .texture_blender import TextureBlender, BlendingConfig


@dataclass
class PipelineConfig:
    """Configuration for the multi-view pipeline."""
    texture_resolution: int = 1024
    low_vram_mode: bool = True
    device: str = "cuda"
    voxel_size: float = 0.02
    max_points_per_view: int = 100000
    max_image_size: int = 1024  # Maximum size for the longer edge of input images


class MultiViewPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the multi-view reconstruction pipeline."""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.view_processor = ViewProcessor(
            device=self.config.device,
            low_vram_mode=self.config.low_vram_mode,
            bake_resolution=self.config.texture_resolution
        )
        
        self.point_fuser = PointCloudFuser(
            FusionConfig(
                voxel_size=self.config.voxel_size,
                max_points_per_view=self.config.max_points_per_view
            )
        )
        
        self.texture_blender = TextureBlender(
            BlendingConfig(
                texture_size=self.config.texture_resolution
            )
        )

    def process_views(
        self,
        views: Dict[str, Union[str, Image.Image]],
        output_dir: Optional[str] = None
    ) -> Mesh:
        """
        Process multiple views to create a textured 3D mesh.
        
        Args:
            views: Dictionary of view type to image path/PIL Image
            output_dir: Optional directory to save intermediate results
            
        Returns:
            Textured mesh
        """
        try:
            # Process views sequentially
            processed_views = []
            images = []
            pil_images = []  # Store original PIL images
            
            for view_type, image_source in views.items():
                # Load and preprocess image
                if isinstance(image_source, str):
                    image = self._load_and_preprocess_image(image_source)
                else:
                    image = self._preprocess_image(image_source)
                
                # Process view
                print(f"Processing {view_type} view...")
                view_data = self.view_processor.process_view(
                    image,
                    view_type
                )
                processed_views.append(view_data)
                images.append(self._prepare_image_tensor(image))
                pil_images.append(image)  # Store the PIL image
                
                # Save intermediate results if requested
                if output_dir:
                    self._save_intermediate(view_data, view_type, output_dir)
                
                # Clear GPU memory after each view
                torch.cuda.empty_cache()
            
            # Fuse point clouds
            print("Fusing point clouds...")
            fused_points = self.point_fuser.fuse_views(processed_views)
            
            # Ensure fused points are in the correct format
            if not isinstance(fused_points, torch.Tensor):
                fused_points = torch.tensor(fused_points, dtype=torch.float32, device=self.config.device)
            
            if output_dir:
                points_path = os.path.join(output_dir, "fused_points.ply")
                self._save_point_cloud(fused_points, points_path)
            
            # Generate base mesh using front view
            print("Generating base mesh...")
            front_view = processed_views[0]  # Assuming front is first
            
            # Ensure points are in the correct format for the model
            model_points = fused_points.to(device=self.config.device, dtype=torch.float32)
            
            # Use the original PIL image for mesh generation
            base_mesh = self.view_processor.model.run_image(
                pil_images[0],  # Use the stored PIL image
                pointcloud=model_points,
                bake_resolution=self.config.texture_resolution
            )[0]
            
            # Blend textures from all views
            print("Blending textures...")
            final_mesh = self.texture_blender.blend_textures(
                base_mesh,
                processed_views,
                pil_images  # Use PIL images for texture blending
            )
            
            # Save final result if requested
            if output_dir:
                self._save_mesh(final_mesh, output_dir)
            
            return final_mesh
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            # Clean up
            self.cleanup()
            raise e

    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image from path."""
        try:
            image = Image.open(image_path).convert('RGBA')
            return self._preprocess_image(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for reconstruction."""
        # Resize if needed
        if max(image.size) > self.config.max_image_size:
            ratio = self.config.max_image_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Ensure image is in RGBA format
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        return image

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor while maintaining RGBA format."""
        # Ensure image is in RGBA format
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        
        # Ensure array is float32 and normalized to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Convert to CxHxW format (4xHxW for RGBA)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Move to device if needed
        if self.config.device != 'cpu':
            img_tensor = img_tensor.to(self.config.device)
        
        return img_tensor

    def _save_intermediate(
        self,
        view_data: ViewData,
        view_type: str,
        output_dir: str
    ):
        """Save intermediate results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save point cloud
        points_path = os.path.join(output_dir, f"{view_type}_points.ply")
        self._save_point_cloud(view_data.point_cloud, points_path)
        
        # Save confidence map
        conf_path = os.path.join(output_dir, f"{view_type}_confidence.png")
        conf_map = (view_data.confidence_map.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(conf_map).save(conf_path)

    def _save_point_cloud(self, points: torch.Tensor, path: str):
        """Save point cloud to PLY file."""
        import open3d as o3d
        
        # Ensure points are on CPU and in float64 format (Open3D preference)
        points_np = points.detach().cpu().numpy().astype(np.float64)
        
        # If points have more than 3 dimensions (e.g., including normals),
        # extract just the XYZ coordinates
        if points_np.shape[-1] > 3:
            points_np = points_np[:, :3]  # Take first 3 dimensions (XYZ)
        elif points_np.shape[-1] < 3:
            raise ValueError(f"Points must have at least 3 coordinates, got shape {points_np.shape}")
        
        # Ensure points are in the correct shape (Nx3)
        if len(points_np.shape) > 2:
            points_np = points_np.reshape(-1, 3)
        
        # Create point cloud and set points
        pcd = o3d.geometry.PointCloud()
        try:
            pcd.points = o3d.utility.Vector3dVector(points_np)
            
            # If we have normals in the original data, try to set them
            if points.shape[-1] == 6:
                normals_np = points.detach().cpu().numpy()[:, 3:].astype(np.float64)
                pcd.normals = o3d.utility.Vector3dVector(normals_np)
        except Exception as e:
            print(f"Debug info - points shape: {points_np.shape}, dtype: {points_np.dtype}")
            print(f"First few points: {points_np[:5]}")
            raise RuntimeError(f"Failed to create point cloud: {str(e)}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # Save point cloud
            success = o3d.io.write_point_cloud(path, pcd, write_ascii=True)
            if not success:
                raise RuntimeError("Failed to write point cloud file")
        except Exception as e:
            raise RuntimeError(f"Error saving point cloud to {path}: {str(e)}")

    def _save_mesh(self, mesh: Mesh, output_dir: str):
        """Save mesh to file."""
        mesh_path = os.path.join(output_dir, "final_mesh.glb")
        mesh.export(mesh_path, include_normals=True)

    def cleanup(self):
        """Clean up resources."""
        self.view_processor.cleanup()
        torch.cuda.empty_cache() 