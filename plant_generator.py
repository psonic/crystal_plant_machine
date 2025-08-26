#!/usr/bin/env python3
"""
🌱 Crystal Plant Generator - Generatore di Video Logo Crystal Therapy
Legge config, carica SVG/PDF e mostra il tracciato colorato.
"""

import sys
import os
import cv2
import numpy as np
import yaml
import argparse
from components import svg_pdf, preview, rendering, animations


def read_config(config_path):
    """Legge la configurazione dal file YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Errore lettura config: {e}")
        return {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crystal Plant Generator')
    
    # Gruppo mutualmente esclusivo per preview/render
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--preview', action='store_true', 
                             help='Mostra preview del logo')
    action_group.add_argument('--render', action='store_true',
                             help='Renderizza video del logo')
    
    # Animazione (opzionale)
    available_animations = animations.get_available_animations()
    parser.add_argument('--animation', choices=available_animations, default='static',
                       help=f'Tipo di animazione ({", ".join(available_animations)})')
    
    return parser.parse_args()
def main():
    """Funzione principale del generatore."""
    print("🌱 Crystal Plant Generator - Avvio...")
    
    # Parse arguments
    args = parse_arguments()
    
    # Leggi configurazione
    config_path = 'config.yaml'
    config = read_config(config_path)
    
    # Parametri da config
    use_svg_source = config.get('use_svg_source', 'PDF').upper()
    svg_path = config.get('paths', {}).get('svg_path', 'input/logo.svg')
    pdf_path = config.get('paths', {}).get('pdf_path', 'input/logo.pdf')
    padding = config.get('video', {}).get('svg_padding', 40)
    logo_zoom_factor = config.get('logo', {}).get('zoom_factor', 1.0)
    smoothing_enabled = config.get('smoothing', {}).get('enabled', True)
    smoothing_factor = config.get('smoothing', {}).get('factor', 0.001)
    
    # Colori logo (BGR per OpenCV)
    logo_color_config = config.get('logo', {}).get('color', {})
    logo_color = (
        logo_color_config.get('b', 255),  # Blue
        logo_color_config.get('g', 255),  # Green
        logo_color_config.get('r', 255)   # Red
    )
    logo_alpha = config.get('logo', {}).get('alpha', 1.0)
    
    mode = "PREVIEW" if args.preview else "RENDER"
    print(f"🎯 Modalità: {mode}")
    if hasattr(args, 'animation'):
        print(f"🎭 Animazione: {args.animation}")
    print(f"📂 Sorgente: {use_svg_source}")
    print(f"🎨 Colore logo: {logo_color}, Alpha: {logo_alpha}")
    
    # Dimensioni video (fisso per ora)
    width, height = 800, 800
    
    try:
        # Estrai contorni dal file scelto
        if use_svg_source == 'SVG':
            print(f"📄 Caricamento SVG: {svg_path}")
            contours, hierarchy = svg_pdf.extract_contours_from_svg(
                svg_path, width, height, padding, logo_zoom_factor=logo_zoom_factor)
        else:
            print(f"📄 Caricamento PDF: {pdf_path}")
            contours, hierarchy = svg_pdf.extract_contours_from_pdf(
                pdf_path, width, height, padding, logo_zoom_factor=logo_zoom_factor)
        
        print(f"✅ Estratti {len(contours)} contorni")
        
        # Crea maschera unificata
        mask = svg_pdf.create_unified_mask(
            contours, hierarchy, width, height, smoothing_enabled, smoothing_factor)
        
        # Crea immagine colorata
        color_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Applica il colore solo dove c'è la maschera
        color_img[mask == 255] = logo_color
        
        # Applica trasparenza se necessaria
        if logo_alpha < 1.0:
            # Crea sfondo bianco
            bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            # Mescola con alpha blending
            color_img = cv2.addWeighted(color_img, logo_alpha, bg, 1-logo_alpha, 0)
        
        # Esegui azione richiesta
        if args.preview:
            preview.show_preview(color_img, config, args.animation)
        elif args.render:
            rendering.render_video(color_img, config, args.animation)
        
        print("✅ Operazione completata!")
        
    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())