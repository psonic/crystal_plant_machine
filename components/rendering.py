"""
🎬 Rendering Module - Crystal Plant Generator
Gestisce la generazione di video del logo.
"""

import cv2
import os
from datetime import datetime
from . import animations


def render_video(color_img, config, animation_type="static", output_filename=None):
    """
    Renderizza video del logo.
    
    Args:
        color_img: Immagine colorata del logo
        config: Configurazione del progetto
        animation_type: Tipo di animazione
        output_filename: Nome file output (opzionale)
    
    Returns:
        str: Percorso del file video generato
    """
    # Crea animator
    animator = animations.create_animator(color_img, config)
    total_frames = animator.get_total_frames()
    
    # Parametri video
    video_config = config.get('video', {})
    fps = video_config.get('fps', 30)
    duration = video_config.get('duration_seconds', 30)
    
    height, width = color_img.shape[:2]
    
    # Setup percorso output
    os.makedirs('output', exist_ok=True)
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'crystal_plant_{animation_type}_{timestamp}.mp4'
    
    output_path = os.path.join('output', output_filename)
    
    print(f"🎬 Renderizzando video:")
    print(f"   🎭 Animazione: {animation_type}")
    print(f"   📐 Risoluzione: {width}x{height}")
    print(f"   🎞️ FPS: {fps}")
    print(f"   ⏱️ Durata: {duration}s ({total_frames} frames)")
    print(f"   📁 Output: {output_path}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception("Impossibile aprire video writer")
    
    try:
        for frame_num in range(total_frames):
            # Genera frame tramite animator
            frame = animator.get_frame(frame_num, animation_type)
            out.write(frame)
            
            # Progress bar ogni 10%
            if frame_num % max(1, total_frames // 10) == 0:
                progress = (frame_num / total_frames) * 100
                print(f"📹 Progresso: {progress:.1f}%")
        
        print(f"📹 Progresso: 100.0%")
        
    except Exception as e:
        print(f"❌ Errore durante rendering: {e}")
        raise
    finally:
        out.release()
    
    print(f"✅ Video salvato: {output_path}")
    return output_path
