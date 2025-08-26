"""
Crystal Plant Machine - Main Script
Generatore di animazioni         print("🎬 Avvio preview in tempo reale...")
    print("📺 Premere 'q' per uscire") print("🎬 Avvio preview in tempo reale...")
    print("📺 Premere 'q' per uscire")rint("🎬 Avvio preview in tempo reale...")
    print("📺 Premere 'e' per uscire, 'SPACE' per pausa/play, 'r' per riavviare")ganiche con crescita di rami e deformazioni elastiche.
"""

import cv2
import numpy as np
import sys
import yaml
import argparse
from components import svg_pdf
from components.animation import animate_organic_growth
from components.branch_system import BranchManager
from components.deformations import apply_elastic_deformation
from components.video_utils import load_and_prepare_texture, apply_texture_to_frames, save_animation_to_mp4


def load_config(config_path='config.yaml'):
    """Carica la configurazione dal file YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configurazione caricata da {config_path}")
        return config
    except Exception as e:
        print(f"❌ Errore nel caricamento della configurazione: {e}")
        sys.exit(1)


def extract_contours_from_input(config):
    """Estrae i contorni dal file di input (PDF o SVG)."""
    contours = None
    hierarchy = None
    
    if config['input']['type'].upper() == 'PDF':
        try:
            contours, hierarchy = svg_pdf.extract_contours_from_pdf(
                config['input']['path'], 
                config['video']['width'], 
                config['video']['height'], 
                padding=config['input']['padding']
            )
        except Exception as e:
            print(f"❌ Errore durante l'estrazione dal PDF: {e}")
            return None, None
    elif config['input']['type'].upper() == 'SVG':
        try:
            contours, hierarchy = svg_pdf.extract_contours_from_svg(
                config['input']['path'], 
                config['video']['width'], 
                config['video']['height'], 
                padding=config['input']['padding']
            )
        except Exception as e:
            print(f"❌ Errore durante l'estrazione dall'SVG: {e}")
            return None, None
    else:
        print(f"❌ Tipo di input non valido: {config['input']['type']}. Scegliere 'PDF' o 'SVG'.")
        return None, None

    return contours, hierarchy


def preview_animation(contours, width, height, logo_mask, hierarchy, config):
    """
    Mostra una preview in tempo reale dell'animazione in una finestra OpenCV.
    """
    print("� Avvio preview in tempo reale...")
    print("📺 Premere 'q' per uscire, 'SPACE' per pausa/play, 'r' per riavviare")
    
    # Crea la finestra
    cv2.namedWindow('Crystal Plant Preview', cv2.WINDOW_NORMAL)
    
    # Scala per la preview (riduciamo per performance migliori)
    preview_scale = 0.6
    preview_width = int(width * preview_scale)
    preview_height = int(height * preview_scale)
    
    # Ridimensiona il logo per la preview
    preview_logo_mask = cv2.resize(logo_mask, (preview_width, preview_height))
    
    # Ridimensiona i contorni per la preview
    preview_contours = []
    for contour in contours:
        scaled_contour = (contour * preview_scale).astype(np.int32)
        preview_contours.append(scaled_contour)
    
    # Inizializza il branch manager per la preview
    from components.branch_system import BranchManager
    branch_manager = BranchManager(config)
    branch_manager.initialize_branches_from_contours(preview_contours, hierarchy)
    
    # Variabili per FPS e timing
    import time
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    step = 0
    
    # Target FPS per la preview
    target_fps = config.get('video', {}).get('fps', 30)
    frame_time = 1.0 / target_fps
    
    while True:
        frame_start = time.time()
        
        # Genera la maschera corrente (senza deformazioni)
        temp_mask = preview_logo_mask.copy()
        
        # Aggiorna e disegna i rami sulla maschera
        branch_manager.grow_and_ramify()
        branch_manager.draw_all_on_mask(temp_mask)
        
        # Applica deformazioni alla maschera (stessa logica del render)
        deformed_mask = apply_elastic_deformation(
            temp_mask,      # Passa direttamente la maschera in scala di grigi
            step, 
            config['animation']['steps'],  # Usa gli step totali dal config
            preview_width,  # Usa le dimensioni della preview, non originali
            preview_height, # Usa le dimensioni della preview, non originali  
            config
        )
        
        # Converte la maschera finale in RGB per il frame
        frame = cv2.cvtColor(deformed_mask, cv2.COLOR_GRAY2BGR)
        
        # Calcola FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        
        # Aggiungi info essenziali
        info_color = (100, 255, 100)
        cv2.putText(frame, f"FPS: {current_fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
        cv2.putText(frame, f"Step: {step}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
        
        cv2.imshow('Crystal Plant Preview', frame)
        
        # Gestione input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        step += 1
        
        # Controlla il timing per mantenere il target FPS
        elapsed = time.time() - frame_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
    
    cv2.destroyAllWindows()
    print("👋 Preview terminata")


def render_animation(contours, width, height, logo_mask, hierarchy, config):
    """
    Renderizza l'animazione completa e la salva come video.
    """
    print("🎬 Avvio rendering completo...")
    
    # Genera le maschere dell'animazione con crescita organica
    mask_frames = animate_organic_growth(
        contours, 
        width, 
        height, 
        logo_mask, 
        hierarchy, 
        config
    )

    if not mask_frames:
        print("❌ Nessun frame generato, impossibile salvare il video.")
        return

    # Applica deformazioni e converte in frame RGB
    final_frames = []
    print("🌊 Applicazione deformazioni e conversione RGB...")
    
    for step, mask in enumerate(mask_frames):
        # Applica deformazioni elastiche
        deformed_mask = apply_elastic_deformation(mask, step, len(mask_frames), width, height, config)
        
        # Converte in frame RGB
        rgb_frame = cv2.cvtColor(deformed_mask, cv2.COLOR_GRAY2BGR)
        final_frames.append(rgb_frame)

    # Salva l'animazione finale
    output_file = save_animation_to_mp4(
        final_frames, 
        width, 
        height, 
        config
    )
    
    if output_file:
        print(f"🎉 Animazione completata: {output_file}")


def parse_arguments():
    """Parsing degli argomenti da command line."""
    parser = argparse.ArgumentParser(description='Crystal Plant Machine - Generatore di animazioni organiche')
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--preview', action='store_true', 
                      help='Mostra preview in tempo reale dell\'animazione')
    group.add_argument('--render', action='store_true', 
                      help='Renderizza l\'animazione completa come video')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Percorso del file di configurazione (default: config.yaml)')
    
    return parser.parse_args()


def main():
    """Funzione principale del programma."""
    args = parse_arguments()
    
    # Se non è specificato niente, usa render come default
    if not args.preview and not args.render:
        args.render = True
    
    print("🌊 Avvio Crystal Plant Machine...")
    print(f"📋 Modalità: {'Preview' if args.preview else 'Render'}")

    # Carica la configurazione
    config = load_config(args.config)
    
    # Estrai i contorni del logo
    contours, hierarchy = extract_contours_from_input(config)
    if not contours:
        print("❌ Nessun contorno estratto. Impossibile procedere.")
        return

    # Crea la maschera unificata del logo
    logo_mask = svg_pdf.create_unified_mask(
        contours, 
        hierarchy, 
        config['video']['width'], 
        config['video']['height'], 
        config['smoothing']['enabled'], 
        config['smoothing']['factor']
    )

    # Esegui la modalità appropriata
    if args.preview:
        preview_animation(contours, config['video']['width'], config['video']['height'], logo_mask, hierarchy, config)
    else:
        render_animation(contours, config['video']['width'], config['video']['height'], logo_mask, hierarchy, config)


if __name__ == "__main__":
    main()
