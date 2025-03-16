"""
Efficient point cloud fusion with memory management and quality-aware merging.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import open3d as o3d

from .view_processor import ViewData


@dataclass
class FusionConfig:
    """Configuration for point cloud fusion."""
    voxel_size: float = 0.02
    distance_threshold: float = 0.05
    normal_threshold: float = 0.8
    max_points_per_view: int = 100000
    min_confidence: float = 0.3


class PointCloudFuser:
    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize point cloud fuser with configuration."""
        self.config = config or FusionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fuse_views(self, views: List[ViewData]) -> torch.Tensor:
        """
        Fuse multiple views' point clouds efficiently.
        
        Args:
            views: List of ViewData objects containing point clouds and metadata
            
        Returns:
            Fused point cloud tensor
        """
        try:
            # Process views sequentially to manage memory
            base_cloud = self._preprocess_view(views[0])
            
            for i in range(1, len(views)):
                # Load and preprocess current view
                current_cloud = self._preprocess_view(views[i])
                
                # Align and merge with base cloud
                base_cloud = self._merge_clouds(base_cloud, current_cloud, views[i])
                
                # Clear current view data
                del current_cloud
                torch.cuda.empty_cache()

            # Final cleanup and optimization
            fused_cloud = self._optimize_final_cloud(base_cloud)
            return fused_cloud

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                # Reduce point count and retry
                old_max = self.config.max_points_per_view
                self.config.max_points_per_view = old_max // 2
                try:
                    result = self.fuse_views(views)
                    self.config.max_points_per_view = old_max
                    return result
                except:
                    self.config.max_points_per_view = old_max
                    raise
            raise e

    def _preprocess_view(self, view: ViewData) -> o3d.geometry.PointCloud:
        """Preprocess single view point cloud."""
        # Filter by confidence
        mask = view.confidence_map >= self.config.min_confidence
        points = view.point_cloud[mask]
        
        # Subsample if needed
        if len(points) > self.config.max_points_per_view:
            indices = torch.randperm(len(points))[:self.config.max_points_per_view]
            points = points[indices]
        
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        return pcd

    def _merge_clouds(
        self,
        base_cloud: o3d.geometry.PointCloud,
        current_cloud: o3d.geometry.PointCloud,
        current_view: ViewData
    ) -> o3d.geometry.PointCloud:
        """Merge two point clouds with ICP alignment."""
        # Initial alignment based on view direction
        init_transform = self._compute_initial_transform(current_view.view_direction)
        current_cloud.transform(init_transform)
        
        # Fine alignment with ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_cloud, base_cloud,
            self.config.distance_threshold,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Apply refined transformation
        current_cloud.transform(reg_p2p.transformation)
        
        # Merge clouds
        merged_cloud = base_cloud + current_cloud
        
        # Voxel downsample to reduce memory
        merged_cloud = merged_cloud.voxel_down_sample(self.config.voxel_size)
        
        return merged_cloud

    def _optimize_final_cloud(self, cloud: o3d.geometry.PointCloud) -> torch.Tensor:
        """Optimize and convert final point cloud."""
        # Remove outliers
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Ensure normals are consistent
        cloud.normalize_normals()
        
        # Convert back to tensor
        points = torch.tensor(np.asarray(cloud.points), dtype=torch.float32)
        normals = torch.tensor(np.asarray(cloud.normals), dtype=torch.float32)
        
        # Combine points and normals
        return torch.cat([points, normals], dim=1)

    def _compute_initial_transform(self, view_direction: torch.Tensor) -> np.ndarray:
        """Compute initial transformation matrix based on view direction."""
        view_dir = view_direction.cpu().numpy()
        up = np.array([0, 1, 0])
        
        # Compute rotation matrix
        z_axis = view_dir / np.linalg.norm(view_dir)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        return transform 