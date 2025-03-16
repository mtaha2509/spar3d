"""
Main pipeline for multi-view 3D reconstruction using SPAR3D.
"""

import torch
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
    batch_size: int = 1
    voxel_size: float = 0.02
    max_points_per_view: int = 100000


class MultiViewPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the multi-view reconstruction pipeline."""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.view_processor = ViewProcessor(
            device=self.config.device,
            low_vram_mode=self.config.low_vram_mode
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
            
            for view_type, image_source in views.items():
                # Load image if path provided
                if isinstance(image_source, str):
                    image = Image.open(image_source).convert('RGBA')
                else:
                    image = image_source
                
                # Process view
                print(f"Processing {view_type} view...")
                view_data = self.view_processor.process_view(
                    image,
                    view_type,
                    self.config.batch_size
                )
                processed_views.append(view_data)
                images.append(self._prepare_image(image))
                
                # Save intermediate results if requested
                if output_dir:
                    self._save_intermediate(view_data, view_type, output_dir)
                
                # Clear GPU memory after each view
                torch.cuda.empty_cache()
            
            # Fuse point clouds
            print("Fusing point clouds...")
            fused_points = self.point_fuser.fuse_views(processed_views)
            
            if output_dir:
                self._save_point_cloud(fused_points, output_dir)
            
            # Generate base mesh using front view
            print("Generating base mesh...")
            front_view = processed_views[0]  # Assuming front is first
            base_mesh = self.view_processor.model.run_image(
                images[0],
                pointcloud=fused_points,
                bake_resolution=self.config.texture_resolution
            )[0]
            
            # Blend textures from all views
            print("Blending textures...")
            final_mesh = self.texture_blender.blend_textures(
                base_mesh,
                processed_views,
                images
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

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        # Convert to tensor and normalize
        tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        return tensor.permute(2, 0, 1)  # Convert to CxHxW format

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
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        o3d.io.write_point_cloud(path, pcd)

    def _save_mesh(self, mesh: Mesh, output_dir: str):
        """Save mesh to file."""
        mesh_path = os.path.join(output_dir, "final_mesh.glb")
        mesh.export(mesh_path, include_normals=True)

    def cleanup(self):
        """Clean up resources."""
        self.view_processor.cleanup()
        torch.cuda.empty_cache() 