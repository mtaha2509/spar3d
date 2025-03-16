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
        batch_size=1,
        voxel_size=0.02,
        max_points_per_view=100000
    )
    
    # Initialize pipeline
    pipeline = MultiViewPipeline(config)
    
    # Define views
    views = {
        'front': os.path.join('C:', 'Users', 'DELL', 'Downloads', 'doll', 'front.png'),
        'back': os.path.join('C:', 'Users', 'DELL', 'Downloads', 'doll', 'back.png'),
        'left': os.path.join('C:', 'Users', 'DELL', 'Downloads', 'doll', 'left.png'),
        'right': os.path.join('C:', 'Users', 'DELL', 'Downloads', 'doll', 'right.png')
    }
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process views
        print("Starting multi-view reconstruction...")
        mesh = pipeline.process_views(views, output_dir=output_dir)
        
        print(f"\nReconstruction completed successfully!")
        print(f"Output files saved to: {output_dir}")
        print(f"Mesh statistics:")
        print(f"- Vertices: {len(mesh.vertices)}")
        print(f"- Faces: {len(mesh.faces)}")
        print(f"- Texture resolution: {mesh.texture.shape[:2]}")
        
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
    finally:
        # Clean up
        pipeline.cleanup()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 