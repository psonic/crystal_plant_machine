"""
🖼️ Preview Module - Crystal Plant Generator
Gestisce la visualizzazione in anteprima del logo.
"""

import cv2
import time
from . import animations


def show_preview(color_img, config=None, animation_type="static"):
    """
    Mostra preview del logo in una finestra OpenCV.
    
    Args:
        color_img: Immagine colorata da mostrare
        config: Configurazione (per animazioni)
        animation_type: Tipo di animazione da mostrare
    """
    print(f"🖼️ Mostrando preview ({animation_type})...")
    
    # Se è statico, mostra solo l'immagine
    if animation_type == "static" or config is None:
        cv2.imshow('Crystal Plant Logo - Preview', color_img)
        print("⌨️ Premi ESC per chiudere la finestra...")
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
    else:
        # Mostra animazione in tempo reale
        animator = animations.create_animator(color_img, config)
        fps = config.get('video', {}).get('fps', 30)
        frame_delay = int(1000 / fps)  # millisecondi
        
        print("⌨️ Controlli:")
        print("   ESC - Chiudi")
        print("   SPACE - Pausa/Play")
        
        frame_num = 0
        total_frames = animator.get_total_frames()
        paused = False
        
        while True:
            if not paused:
                # Genera frame animato
                animated_frame = animator.get_frame(frame_num, animation_type)
                cv2.imshow('Crystal Plant Logo - Preview (Animato)', animated_frame)
                
                frame_num = (frame_num + 1) % total_frames
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                paused = not paused
                status = "PAUSA" if paused else "PLAY"
                print(f"🎬 {status}")
    
    cv2.destroyAllWindows()
    print("✅ Preview chiusa")
