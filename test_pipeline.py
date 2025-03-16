"""
Test script for multi-view SPAR3D pipeline.
"""

import os
import torch
from PIL import Image
from mv_spar3d import MultiViewPipeline
from mv_spar3d.mv_pipeline import PipelineConfig


def main():
    # Configure pipeline
    config = PipelineConfig(
        texture_resolution=1024,
        low_vram_mode=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        voxel_size=0.02,
        max_points_per_view=100000,
        max_image_size=1024  # Maximum size for the longer edge of input images
    )
    
    # Initialize pipeline
    pipeline = MultiViewPipeline(config)
    
    # Define views
    views = {
    'front': os.path.join('/home', 'taha', 'Downloads', 'doll', 'front.png'),
    'back': os.path.join('/home', 'taha', 'Downloads', 'doll', 'back.png'),
    'left': os.path.join('/home', 'taha', 'Downloads', 'doll', 'left.png'),
    'right': os.path.join('/home', 'taha', 'Downloads', 'doll', 'right.png')
    }

    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process views
        print("Starting multi-view reconstruction...")
        print("Using device:", config.device)
        print("Low VRAM mode:", config.low_vram_mode)
        print("Max image size:", config.max_image_size)
        
        mesh = pipeline.process_views(views, output_dir=output_dir)
        
        print(f"\nReconstruction completed successfully!")
        print(f"Output files saved to: {output_dir}")
        print(f"Mesh statistics:")
        print(f"- Vertices: {len(mesh.vertices)}")
        print(f"- Faces: {len(mesh.faces)}")
        print(f"- Texture resolution: {mesh.texture.shape[:2]}")
        
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        pipeline.cleanup()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 
