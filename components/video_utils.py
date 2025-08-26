"""
Utility per la gestione di texture e video output per Crystal Plant Machine.
"""

import cv2
import numpy as np
import random
import sys
from datetime import datetime


def load_and_prepare_texture(texture_path, width, height):
    """Carica la texture, la ridimensiona e la prepara per la fusione."""
    try:
        texture = cv2.imread(texture_path)
        if texture is None:
            print(f"⚠️ Attenzione: Texture non trovata in '{texture_path}'. L'animazione procederà senza.")
            return None
        # Ridimensiona la texture per adattarla al canvas dell'animazione
        resized_texture = cv2.resize(texture, (width, height))
        print("🎨 Texture caricata e ridimensionata con successo.")
        return resized_texture
    except Exception as e:
        print(f"❌ Errore durante il caricamento della texture: {e}")
        return None


def apply_texture_to_frames(frames, texture, config):
    """
    Applica una texture ai frame dell'animazione.
    
    Args:
        frames: Lista di frame dell'animazione
        texture: Texture da applicare (None se non disponibile)
        config: Configurazione
    
    Returns:
        Lista di frame con texture applicata
    """
    if not frames:
        return []
    
    final_frames = []
    
    print("\n🎨 Applicazione della texture e compositing finale...")
    from .animation import print_progress_bar
    import time
    start_time = time.time()

    for i, unified_frame in enumerate(frames):
        if texture is not None:
            # Convertiamo il frame unificato in una maschera alpha (in scala di grigi)
            alpha_mask = cv2.cvtColor(unified_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            alpha_mask = alpha_mask[:, :, np.newaxis]  # Adatta le dimensioni per il calcolo

            # Applichiamo la texture usando la maschera alpha
            texture_float = texture.astype(np.float32)
            blended_frame_float = texture_float * alpha_mask
            
            # Riconvertiamo in formato 8-bit per il video finale
            final_frame = blended_frame_float.astype(np.uint8)
            final_frames.append(final_frame)
        else:
            # Se non c'è texture, usiamo direttamente il frame unificato
            final_frames.append(unified_frame)
        
        print_progress_bar(i + 1, len(frames), start_time, prefix='🔮 Compositing', suffix='Completato')
    print()
    
    return final_frames


def save_animation_to_mp4(frames, width, height, config):
    """
    Salva i frame come video MP4 compatibile con WhatsApp.
    """
    if not frames:
        print("❌ Nessun frame da salvare.")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    magic_symbol = random.choice(config['output']['magic_symbols'])
    output_filename = f"{config['output']['directory']}/{config['output']['filename_prefix']}_{timestamp}_{magic_symbol}.mp4"
    
    # Codec H.264 (avc1) è ampiamente compatibile
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_filename, fourcc, config['video']['fps'], (width, height))

    if not out.isOpened():
        print("❌ Errore: Impossibile creare il file video. Controlla i codec.")
        return None

    print(f"\n💾 Salvataggio animazione in corso: {output_filename}")
    for i, frame in enumerate(frames):
        out.write(frame)
        # Stampa una semplice progress bar per il salvataggio
        progress = (i + 1) / len(frames)
        bar = '█' * int(progress * 20) + '-' * (20 - int(progress * 20))
        sys.stdout.write(f'\rSalvataggio: [{bar}] {progress:.0%}')
        sys.stdout.flush()

    out.release()
    print(f"\n✅ Video salvato con successo!")
    return output_filename


def create_color_gradient_frame(width, height, colors, step, total_steps):
    """
    Crea un frame con gradiente di colori che cambia nel tempo.
    
    Args:
        width, height: Dimensioni del frame
        colors: Lista di colori RGB per il gradiente
        step: Step corrente
        total_steps: Numero totale di step
    
    Returns:
        Frame con gradiente colorato
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calcola la progressione temporale
    time_factor = (step / total_steps) * 2 * np.pi
    
    # Crea gradiente animato
    for y in range(height):
        for x in range(width):
            # Calcola l'indice del colore basato su posizione e tempo
            color_index = (np.sin(x * 0.01 + time_factor) + np.cos(y * 0.01 + time_factor * 1.3)) * 0.5 + 0.5
            color_index = int(color_index * (len(colors) - 1))
            color_index = np.clip(color_index, 0, len(colors) - 1)
            
            frame[y, x] = colors[color_index]
    
    return frame


def blend_frames(frame1, frame2, alpha=0.5):
    """
    Miscela due frame con un fattore alpha.
    
    Args:
        frame1, frame2: Frame da miscelare
        alpha: Fattore di miscela (0.0 = solo frame1, 1.0 = solo frame2)
    
    Returns:
        Frame miscelato
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    return blended
