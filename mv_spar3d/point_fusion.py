"""
Efficient point cloud fusion with memory management and quality-aware merging.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from .view_processor import ViewData

# Try importing Open3D, fallback to numpy-based operations if not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available, falling back to numpy-based operations. This might be slower.")


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
            if OPEN3D_AVAILABLE:
                return self._fuse_views_open3d(views)
            else:
                return self._fuse_views_numpy(views)

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

    def _fuse_views_numpy(self, views: List[ViewData]) -> torch.Tensor:
        """Numpy-based point cloud fusion implementation."""
        # Process views sequentially to manage memory
        base_points = self._preprocess_view_numpy(views[0])
        
        for i in range(1, len(views)):
            # Load and preprocess current view
            current_points = self._preprocess_view_numpy(views[i])
            
            # Align and merge with base points
            base_points = self._merge_clouds_numpy(base_points, current_points, views[i])
            
            # Clear current view data
            del current_points
            torch.cuda.empty_cache()

        # Final cleanup and optimization
        return self._optimize_final_cloud_numpy(base_points)

    def _preprocess_view_numpy(self, view: ViewData) -> np.ndarray:
        """Preprocess single view point cloud using numpy."""
        # Filter by confidence
        mask = view.confidence_map >= self.config.min_confidence
        points = view.point_cloud[mask].cpu().numpy()
        
        # Subsample if needed
        if len(points) > self.config.max_points_per_view:
            indices = np.random.choice(
                len(points),
                self.config.max_points_per_view,
                replace=False
            )
            points = points[indices]
        
        return points

    def _merge_clouds_numpy(
        self,
        base_points: np.ndarray,
        current_points: np.ndarray,
        current_view: ViewData
    ) -> np.ndarray:
        """Merge two point clouds using numpy operations."""
        # Initial alignment based on view direction
        transform = self._compute_initial_transform(current_view.view_direction)
        current_points = (transform[:3, :3] @ current_points.T).T + transform[:3, 3]
        
        # Simple concatenation for now (could be enhanced with ICP-like alignment)
        merged_points = np.vstack([base_points, current_points])
        
        # Voxel downsample
        merged_points = self._voxel_downsample_numpy(
            merged_points,
            self.config.voxel_size
        )
        
        return merged_points

    def _voxel_downsample_numpy(
        self,
        points: np.ndarray,
        voxel_size: float
    ) -> np.ndarray:
        """Simple voxel downsampling using numpy."""
        # Quantize points to voxel coordinates
        voxel_coords = np.floor(points / voxel_size)
        
        # Use dictionary to keep track of points in each voxel
        voxel_dict = {}
        for i, coord in enumerate(voxel_coords):
            key = tuple(coord)
            if key in voxel_dict:
                voxel_dict[key].append(i)
            else:
                voxel_dict[key] = [i]
        
        # Take centroid of points in each voxel
        downsampled = []
        for indices in voxel_dict.values():
            centroid = points[indices].mean(axis=0)
            downsampled.append(centroid)
        
        return np.array(downsampled)

    def _optimize_final_cloud_numpy(self, points: np.ndarray) -> torch.Tensor:
        """Optimize and convert final point cloud using numpy."""
        # Convert back to tensor
        points_tensor = torch.tensor(points, dtype=torch.float32)
        
        # Estimate normals using cross products of neighboring points
        normals = self._estimate_normals_numpy(points)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        
        # Combine points and normals
        return torch.cat([points_tensor, normals_tensor], dim=1)

    def _estimate_normals_numpy(self, points: np.ndarray) -> np.ndarray:
        """Estimate point normals using local neighborhoods."""
        from sklearn.neighbors import NearestNeighbors
        
        # Find k nearest neighbors for each point
        k = min(20, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Compute normals using PCA
        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            normals[i] = eigenvecs[:, 0]  # Smallest eigenvector
        
        # Orient normals consistently
        center = points.mean(axis=0)
        for i in range(len(normals)):
            if np.dot(points[i] - center, normals[i]) < 0:
                normals[i] = -normals[i]
        
        return normals

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

    def _fuse_views_open3d(self, views: List[ViewData]) -> torch.Tensor:
        """Original Open3D-based fusion implementation."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available")
            
        # Process views sequentially to manage memory
        base_cloud = self._preprocess_view_open3d(views[0])
        
        for i in range(1, len(views)):
            # Load and preprocess current view
            current_cloud = self._preprocess_view_open3d(views[i])
            
            # Align and merge with base cloud
            base_cloud = self._merge_clouds_open3d(base_cloud, current_cloud, views[i])
            
            # Clear current view data
            del current_cloud
            torch.cuda.empty_cache()

        # Final cleanup and optimization
        return self._optimize_final_cloud_open3d(base_cloud)

    def _preprocess_view_open3d(self, view: ViewData) -> 'o3d.geometry.PointCloud':
        """Original Open3D preprocessing implementation."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available")
            
        # Filter by confidence
        mask = view.confidence_map >= self.config.min_confidence
        points = view.point_cloud[mask]
        
        # Subsample if needed
        if len(points) > self.config.max_points_per_view:
            indices = torch.randperm(len(points))[:self.config.max_points_per_view]
            points = points[indices]
        
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        return pcd

    def _merge_clouds_open3d(
        self,
        base_cloud: 'o3d.geometry.PointCloud',
        current_cloud: 'o3d.geometry.PointCloud',
        current_view: ViewData
    ) -> 'o3d.geometry.PointCloud':
        """Original Open3D merging implementation."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available")
            
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

    def _optimize_final_cloud_open3d(
        self,
        cloud: 'o3d.geometry.PointCloud'
    ) -> torch.Tensor:
        """Original Open3D optimization implementation."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available")
            
        # Remove outliers
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Ensure normals are consistent
        cloud.normalize_normals()
        
        # Convert back to tensor
        points = torch.tensor(np.asarray(cloud.points), dtype=torch.float32)
        normals = torch.tensor(np.asarray(cloud.normals), dtype=torch.float32)
        
        # Combine points and normals
        return torch.cat([points, normals], dim=1) 