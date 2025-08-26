# Crystal Plant Machine

Svalvoliamo il vettoriale in modi nuovi.
python --version

# Install the sacred dependencies
pip install opencv-python numpy scipy librosa Pillow noise svgpathtools PyMuPDF
```

### Basic Invocation
```bash
# Clone the mystical repository
git clone https://github.com/psonic/crystal_dream_machine.git
cd crystal_dream_machine

# Place your logo (SVG/PDF) in the input/ folder
cp logo.svg input/
cp logo.pdf input/

# Optional: Add background video, texture and audio
cp background.mov input/sfondo.MOV
cp audio.aif input/
cp texture.jpg input/

# Cast the spell - transform your Alex Ortiga logos into flowing journeys
python natisone_trip_generator.py
```

## 🎛️ Configuration Magic

The generator is controlled through the `Config` class with **60+ mystical parameters**:

### 🔮 Core Effects
```python
# Organic deformation (the logo "breathes")
DEFORMATION_ENABLED = True
DEFORMATION_SPEED = 0.01        # How fast the breathing
DEFORMATION_INTENSITY = 11.0    # How deep the breath

# Audio reactivity (bass makes it dance)
DEFORMATION_AUDIO_REACTIVE = True
DEFORMATION_BASS_INTENSITY = 0.15  # Gentle bass response
DEFORMATION_BASS_SPEED = 0.02      # Speed modulation
```

### 🌊 Lens System
```python
# Mystical lens effects
LENS_DEFORMATION_ENABLED = True
NUM_LENSES = 30                 # Number of reality-bending lenses
LENS_MIN_STRENGTH = -1.2        # Gentle distortion range
LENS_MAX_STRENGTH = 1.3         # Maximum mystical power
```

### 📱 Format Options
```python
# Instagram Stories format
INSTAGRAM_STORIES_MODE = True   # Perfect 9:16 vertical videos
TEST_MODE = True               # Quick 5-second tests
```

## 🎨 Blending Presets

Choose from magical blending modes:

- **`glow`** - Ethereal luminous effect
- **`cinematic`** - Hollywood-grade composition  
- **`mystical`** - Deep spiritual vibes
- **`neon`** - Cyberpunk energy
- **`crystal`** - Pure crystalline clarity
- **`fire`** - Passionate flame effects
- **`water`** - Flowing liquid dreams
- **`earth`** - Grounded natural tones

```python
Config.BLENDING_PRESET = 'mystical'  # Let the magic flow
```

## 🎵 Audio Integration

The generator **listens to your audio** and makes the visuals dance:

```python
# Audio files in input/ folder
AUDIO_FILES = ['input/audio1.aif', 'input/audio2.aif']
AUDIO_RANDOM_SELECTION = True    # Randomly pick audio
AUDIO_RANDOM_START = True        # Start from random position

# Sensitivity controls (gentle by default)
AUDIO_BASS_SENSITIVITY = 0.3     # Bass affects lens movement
AUDIO_MID_SENSITIVITY = 0.2      # Mids control strength  
AUDIO_HIGH_SENSITIVITY = 0.15    # Highs add sparkle
```

## 🌟 Advanced Features

### Ghost Tracers
```python
TRACER_ENABLED = True           # Leave mystical trails
TRACER_MAX_OPACITY = 0.8        # Trail visibility
TRACER_FADE_STEPS = 20          # How long trails last
```

### Texture Magic  
```python
TEXTURE_ENABLED = True          # Apply mystical textures
TEXTURE_TARGET = 'both'         # Logo + background
TEXTURE_BLENDING_MODE = 'screen' # How texture blends
```

### Video Background
```python
BACKGROUND_VIDEO_PATH = 'input/sfondo.MOV'
BG_SLOWDOWN_FACTOR = 1.3        # Slow-motion effect
BG_RANDOM_START = True          # Random starting point
```

## 📁 Project Structure

```
CrystalDreamMachine/
├── 🌊 natisone_trip_generator.py         # Main mystical engine
├── 🎨 blending_presets.py                # Visual magic presets  
├── 📝 version_manager.py                 # Git integration
├── input/                                # Your source materials
│   ├── logo.svg                         # Your logo (SVG/PDF)
│   ├── sfondo.MOV                       # Background video
│   ├── texture.jpg                      # Mystical texture
│   └── audio1.aif                       # Sound energy
└── output/                              # Generated magic
    └── crystalpy_YYYYMMDD_HHMMSS.mp4   # Your creation
```

*Born from the creative collaboration of Alex Ortiga (logo design), TV Int (mystical textures), and Iaia & Friend (ice video backgrounds) - united by the flowing waters of the Natisone.*

## 🤝 Contributing to the Magic

We welcome fellow digital mystics! To contribute:

1. Fork the sacred repository
2. Create your mystical branch (`git checkout -b feature/new-magic`)
3. Commit your spells (`git commit -m 'Add mystical feature'`)
4. Push to the branch (`git push origin feature/new-magic`)
5. Open a Pull Request with your magical contribution

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **The Natisone River** - Source of mystical inspiration and flowing energy
- **Alex Ortiga** - Visionary logo designer and creative collaborator
- **TV Int** - Mystical texture provider and digital artist
- **Iaia & Friend** - Ice video creators and visual storytellers
- **The Digital Spirits** - For guiding our algorithms through the mystical waters
- **OpenCV Community** - For the foundational magic
- **All Natisone Trip Users** - For believing in digital mysticism and flowing experiences

---

<div align="center">

*"Technology is the new magic, and magic flows like the eternal Natisone"*

**Made with 💜 and 🌊 by the Natisone Trip Collective**

[🌟 Star this repo](https://github.com/psonic/CrystalPython2) • [🐛 Report bugs](https://github.com/psonic/CrystalPython2/issues) • [💡 Request features](https://github.com/psonic/CrystalPython2/discussions)

</div>
