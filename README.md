# Crystal Python - Video Effects Pipeline

A sophisticated video generation system that creates crystalline animated effects using Manim and organic deformation algorithms.

## Project Structure

```
CrystalPython/
├── main.py                          # 🚀 Main entry point
├── version_manager.py               # 📋 Version control utilities
├── test_manim_deformation.py        # 🧪 Test Manim deformation integration  
├── test_pipeline_deformed.py        # 🧪 Test the deformed pipeline
├── test_slow_deformation.py         # 🧪 Test slow deformation parameters
├── components/                      # 📦 Core components
│   ├── __init__.py
│   ├── config.py                   # ⚙️  Global configuration
│   ├── logger.py                   # 📝 Logging utilities
│   ├── vector_animation.py         # 🎨 Manim vector animations
│   ├── manim_deformed_pipeline.py  # 🌀 Main deformation pipeline
│   ├── manim_deformed_scene.py     # 🎬 Manim scenes with deformation
│   ├── hybrid_pipeline.py          # 🔀 Hybrid vector/raster pipeline
│   ├── deformations.py             # 🌊 Organic deformation algorithms
│   ├── effects.py                  # ✨ Visual effects
│   ├── texture_manager.py          # 🖼️  Texture management
│   ├── svg_parser.py               # 📄 SVG parsing utilities
│   ├── extract_svg_from_pdf.py     # 🔄 PDF to SVG conversion
│   └── utils.py                    # 🛠️  General utilities
├── input/                          # 📥 Input files (logos, textures, videos)
├── output/                         # 📤 Generated videos and history
└── trash/                          # 🗑️  Unused/old files (ignored by git)
```

## Usage

### Quick Start
```bash
python main.py
```

### Current Pipeline Modes

The main script supports different rendering modes:

1. **🌀 Manim + Deformazione** (DEFAULT)
   - Uses Manim for vector graphics
   - Applies organic deformation to each frame
   - Creates fluid, natural movement effects

2. **🔀 Hybrid Pipeline** 
   - Combines vector and raster effects
   - Advanced post-processing

3. **🎨 Pure Vector**
   - Clean Manim vector output only

## Key Features

- **🌊 Organic Deformation**: Fluid, natural movement using Perlin noise
- **💎 Crystal Effects**: Sophisticated crystalline visual effects  
- **🎬 Professional Output**: High-quality MP4 video generation
- **📊 Progress Tracking**: Real-time progress bars and logging
- **🧪 Comprehensive Testing**: Multiple test scenarios for different effects

## Configuration

Edit `components/config.py` to customize:
- Video resolution and frame rate
- Deformation parameters
- Effect intensities
- Output settings

## Testing

Run individual test files to validate specific functionality:

```bash
# Test the main deformation pipeline
python test_pipeline_deformed.py

# Test slow deformation parameters
python test_slow_deformation.py

# Test Manim integration
python test_manim_deformation.py
```

## Requirements

- Python 3.8+
- Manim Community Edition
- OpenCV (cv2)
- NumPy
- FFmpeg (for video processing)

## Output

Generated videos are saved in the `output/` directory with timestamps and Greek letters for easy identification:

```
crystal_water_20250715_123456_α.mp4
```
