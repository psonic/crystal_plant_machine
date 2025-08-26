"""
Sistema di deformazioni elastiche per Crystal Plant Machine.
Applica effetti di deformazione fluida alle maschere come se fossero tessuti elastici.
"""

import cv2
import numpy as np


def apply_elastic_deformation(mask, step, total_steps, width, height, config):
    """
    Applica una deformazione elastica fluida alla maschera, come un tessuto che si allunga e si contrae.
    
    Args:
        mask: Maschera da deformare
        step: Step corrente dell'animazione
        total_steps: Numero totale di step
        width, height: Dimensioni dell'immagine
        config: Configurazione con parametri di deformazione
    
    Returns:
        Maschera deformata
    """
    # Controlla se le deformazioni sono abilitate
    if not config['deformation']['enabled']:
        return mask
    
    # Crea una griglia di coordinate
    y_indices, x_indices = np.indices((height, width), dtype=np.float32)
    
    # Parametri per le onde di deformazione da configurazione
    time_factor = (step / total_steps) * 2 * np.pi * config['deformation']['wave_speed']
    intensity = config['deformation']['intensity']
    
    # Onde multiple con frequenze e ampiezze diverse per un effetto più naturale
    displacement_x = np.zeros_like(x_indices)
    displacement_y = np.zeros_like(y_indices)
    
    # Crea onde multiple basate sul numero di layer configurato
    for layer in range(config['deformation']['wave_layers']):
        layer_factor = (layer + 1) * 0.5  # Ogni layer ha frequenza diversa
        
        # Onde con pattern diversi per ogni layer
        wave_x = np.sin(y_indices * (0.008 + layer * 0.004) + time_factor * (2 + layer)) * (8 - layer * 2) * intensity
        wave_y = np.cos(x_indices * (0.008 + layer * 0.004) + time_factor * (1.5 + layer * 0.5)) * (6 - layer * 1.5) * intensity
        
        # Onde diagonali per maggiore complessità
        diag_wave_x = np.sin(y_indices * 0.015 + x_indices * 0.01 + time_factor * (3 + layer * 0.8)) * (4 - layer) * intensity
        diag_wave_y = np.cos(x_indices * 0.015 + y_indices * 0.01 + time_factor * (2.5 + layer * 0.7)) * (5 - layer * 1.2) * intensity
        
        displacement_x += wave_x + diag_wave_x
        displacement_y += wave_y + diag_wave_y
    
    # Applica le deformazioni alle coordinate
    new_x = x_indices + displacement_x
    new_y = y_indices + displacement_y
    
    # Assicurati che le coordinate rimangano nei bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    # Applica la deformazione usando remap
    deformed_mask = cv2.remap(mask, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return deformed_mask


def create_wave_deformation(width, height, time_factor, wave_config):
    """
    Crea un campo di deformazione basato su onde sinusoidali.
    
    Args:
        width, height: Dimensioni dell'immagine
        time_factor: Fattore temporale per l'animazione
        wave_config: Configurazione delle onde
    
    Returns:
        Tuple di displacement_x, displacement_y
    """
    y_indices, x_indices = np.indices((height, width), dtype=np.float32)
    
    displacement_x = np.zeros_like(x_indices)
    displacement_y = np.zeros_like(y_indices)
    
    # Onda principale
    displacement_x += np.sin(y_indices * wave_config['freq_y'] + time_factor * wave_config['speed_main']) * wave_config['amplitude_main']
    displacement_y += np.cos(x_indices * wave_config['freq_x'] + time_factor * wave_config['speed_main']) * wave_config['amplitude_main']
    
    # Onda secondaria
    displacement_x += np.sin(x_indices * wave_config['freq_x'] * 1.5 + time_factor * wave_config['speed_secondary']) * wave_config['amplitude_secondary']
    displacement_y += np.cos(y_indices * wave_config['freq_y'] * 1.5 + time_factor * wave_config['speed_secondary']) * wave_config['amplitude_secondary']
    
    return displacement_x, displacement_y


def apply_turbulence_deformation(mask, step, total_steps, width, height, turbulence_intensity=0.5):
    """
    Applica una deformazione di turbolenza più caotica alla maschera.
    
    Args:
        mask: Maschera da deformare
        step: Step corrente dell'animazione
        total_steps: Numero totale di step
        width, height: Dimensioni dell'immagine
        turbulence_intensity: Intensità della turbolenza (0.0-1.0)
    
    Returns:
        Maschera deformata
    """
    if turbulence_intensity == 0:
        return mask
    
    y_indices, x_indices = np.indices((height, width), dtype=np.float32)
    time_factor = (step / total_steps) * 2 * np.pi
    
    # Crea rumore turbolento con multiple ottave
    displacement_x = np.zeros_like(x_indices)
    displacement_y = np.zeros_like(y_indices)
    
    # Multiple ottave di rumore
    for octave in range(3):
        scale = 2 ** octave
        freq = 0.01 * scale
        amplitude = turbulence_intensity * (8 / scale)
        
        displacement_x += np.sin(x_indices * freq + time_factor * (1 + octave * 0.3)) * amplitude
        displacement_y += np.cos(y_indices * freq + time_factor * (1.2 + octave * 0.4)) * amplitude
        
        # Rumore diagonale
        displacement_x += np.sin((x_indices + y_indices) * freq * 0.7 + time_factor * (2 + octave * 0.5)) * amplitude * 0.5
        displacement_y += np.cos((x_indices - y_indices) * freq * 0.7 + time_factor * (1.8 + octave * 0.6)) * amplitude * 0.5
    
    # Applica le deformazioni
    new_x = x_indices + displacement_x
    new_y = y_indices + displacement_y
    
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    deformed_mask = cv2.remap(mask, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return deformed_mask
