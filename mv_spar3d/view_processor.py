"""
Memory-efficient view processor for SPAR3D multi-view pipeline.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from PIL import Image

from spar3d.system import SPAR3D
from spar3d.models.mesh import Mesh


@dataclass
class ViewData:
    """Container for view-specific data."""
    point_cloud: torch.Tensor
    camera_params: Dict[str, torch.Tensor]
    confidence_map: torch.Tensor
    view_direction: torch.Tensor


class ViewProcessor:
    def __init__(
        self,
        model: Optional[SPAR3D] = None,
        device: str = "cuda",
        low_vram_mode: bool = True
    ):
        """Initialize view processor with optional model instance."""
        self.device = device
        if model is None:
            self.model = SPAR3D.from_pretrained(
                "stabilityai/stable-point-aware-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=low_vram_mode
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        self.model.eval()

    def process_view(
        self,
        image: Image.Image,
        view_type: str,
        batch_size: int = 1  # Kept for API compatibility but not used
    ) -> ViewData:
        """
        Process a single view efficiently.
        
        Args:
            image: Input image
            view_type: One of 'front', 'back', 'left', 'right'
            batch_size: Kept for API compatibility but not used
            
        Returns:
            ViewData containing point cloud and associated data
        """
        # Define view-specific camera parameters
        view_directions = {
            'front': torch.tensor([0, 0, -1], device=self.device),
            'back': torch.tensor([0, 0, 1], device=self.device),
            'left': torch.tensor([-1, 0, 0], device=self.device),
            'right': torch.tensor([1, 0, 0], device=self.device)
        }

        try:
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        # Get point cloud and camera parameters
                        point_cloud, glob_dict = self.model.run_image(
                            image,
                            return_points=True
                        )
                else:
                    # Get point cloud and camera parameters
                    point_cloud, glob_dict = self.model.run_image(
                        image,
                        return_points=True
                    )
                
                # Extract confidence from model's internal features
                confidence_map = self._compute_confidence_map(glob_dict)
                
                # Create view data
                view_data = ViewData(
                    point_cloud=glob_dict['point_clouds'][0].points.to('cpu'),
                    camera_params=self._extract_camera_params(glob_dict),
                    confidence_map=confidence_map.to('cpu'),
                    view_direction=view_directions[view_type].to('cpu')
                )
                
                # Clear unnecessary tensors
                del glob_dict
                torch.cuda.empty_cache()
                
                return view_data
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory. Try reducing the image size or using low_vram_mode=True")
            raise e

    def _compute_confidence_map(self, glob_dict: Dict) -> torch.Tensor:
        """Compute confidence map for point cloud quality."""
        # Use feature activations to estimate confidence
        features = glob_dict.get('features', None)
        if features is not None:
            confidence = torch.norm(features, dim=1)
            return torch.sigmoid(confidence)
        return torch.ones(glob_dict['point_clouds'][0].points.shape[0], device=self.device)

    def _extract_camera_params(self, glob_dict: Dict) -> Dict[str, torch.Tensor]:
        """Extract and store relevant camera parameters."""
        return {
            'extrinsic': glob_dict.get('camera_extrinsic', torch.eye(4, device=self.device)),
            'intrinsic': glob_dict.get('camera_intrinsic', torch.eye(3, device=self.device)),
            'fov': glob_dict.get('fov', torch.tensor(0.8, device=self.device))
        }

    def cleanup(self):
        """Clean up GPU memory."""
        torch.cuda.empty_cache()
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() 