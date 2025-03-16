from mv_spar3d import MultiViewPipeline

# Initialize pipeline
pipeline = MultiViewPipeline(PipelineConfig(
    texture_resolution=1024,
    low_vram_mode=True,
    voxel_size=0.02,
    max_points_per_view=100000
))

# Process views
views = {
    'front': '/home/taha/Downloads/doll/front.png',
    'back': '/home/taha/Downloads/doll/back.png',
    'left': '/home/taha/Downloads/doll/left.png',
    'right': '/home/taha/Downloads/doll/right.png'
}

# Generate textured mesh
mesh = pipeline.process_views(views, output_dir='output')
