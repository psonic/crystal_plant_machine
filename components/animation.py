"""
Sistema di animazione organica per Crystal Plant Machine.
Coordina la crescita dei rami e le deformazioni elastiche.
"""

import cv2
import numpy as np
import random
import time
import sys
from .branch_system import BranchManager
from .deformations import apply_elastic_deformation


def print_progress_bar(iteration, total, start_time, prefix='✨ Animazione in corso', suffix='Completato', length=30):
    """
    Stampa una barra di avanzamento magica e colorata nel terminale, con numeri colorati.
    """
    # Definizione colori ANSI
    COLOR = {
        "reset": "\033[0m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "magenta": "\033[95m",
    }

    percent = iteration / total
    
    # Calcolo FPS e ETA
    elapsed_time = time.time() - start_time
    fps = iteration / elapsed_time if elapsed_time > 0 else 0
    eta_seconds = (elapsed_time / iteration) * (total - iteration) if iteration > 0 else 0
    eta = time.strftime("%M:%S", time.gmtime(eta_seconds))

    # Emoticon che cambiano con il progresso
    emoticons = ['🌱', '🌿', '🌳', '✨', '🔮', '🌟']
    emoticon = emoticons[int(percent * (len(emoticons) - 1))]

    # Barra colorata (gradiente da ciano a magenta)
    filled_length = int(length * percent)
    bar = ''
    for i in range(length):
        if i < filled_length:
            # Calcola il colore in base alla posizione per il gradiente
            r = int(0 + (255 * (i / length)))
            g = int(255 - (255 * (i / length)))
            b = 255
            bar += f'\033[38;2;{r};{g};{b}m█\033[0m'
        else:
            bar += '░'

    # Costruisce la stringa finale con numeri colorati
    progress_str = (f'\r{prefix} {emoticon} |{bar}| {COLOR["magenta"]}{percent:.1%}{COLOR["reset"]} | '
                    f'{COLOR["cyan"]}{iteration}/{total}{COLOR["reset"]} Frames | '
                    f'{COLOR["green"]}{fps:.1f}{COLOR["reset"]} FPS | '
                    f'ETA: {COLOR["yellow"]}{eta}{COLOR["reset"]} | {suffix}')
    
    sys.stdout.write(progress_str)
    sys.stdout.flush()


def find_sharp_tips(contour, angle_threshold_deg=130, step=15):
    """
    Trova i punti di un contorno che sono "punte" (angoli acuti) e calcola la loro direzione.
    Restituisce una lista di tuple (punto, angolo_di_uscita).
    """
    tips = []
    contour_squeezed = contour.squeeze()
    num_points = len(contour_squeezed)

    if num_points < 2 * step:
        return []

    for i in range(num_points):
        # Prendi tre punti: corrente, precedente e successivo, a una certa distanza 'step'
        # per calcolare l'angolo in modo più stabile e ignorare il rumore.
        p_prev = contour_squeezed[(i - step + num_points) % num_points]
        p_curr = contour_squeezed[i]
        p_next = contour_squeezed[(i + step) % num_points]

        # Calcola i vettori dal punto corrente ai suoi vicini
        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        # Calcola l'angolo tra i due vettori
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue

        dot_product = np.dot(v1, v2) / (norm_v1 * norm_v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))

        # Se l'angolo è acuto, abbiamo trovato una punta
        if angle < angle_threshold_deg:
            # Calcola la direzione di uscita della punta (la bisettrice dell'angolo, rivolta verso l'esterno)
            # Sommiamo i vettori normalizzati e invertiamo la direzione per puntare all'esterno.
            outgoing_vector = -(v1 / norm_v1 + v2 / norm_v2)
            outgoing_angle = np.degrees(np.arctan2(outgoing_vector[1], outgoing_vector[0]))
            
            tips.append((tuple(p_curr), outgoing_angle))
            
    # Filtra le punte troppo vicine per evitare cluster
    if not tips:
        return []
        
    filtered_tips = [tips[0]]
    min_dist_sq = 50**2 # Distanza minima al quadrato tra le punte
    for i in range(1, len(tips)):
        is_far_enough = True
        for j in range(len(filtered_tips)):
            dist_sq = (tips[i][0][0] - filtered_tips[j][0][0])**2 + (tips[i][0][1] - filtered_tips[j][0][1])**2
            if dist_sq < min_dist_sq:
                is_far_enough = False
                break
        if is_far_enough:
            filtered_tips.append(tips[i])

    print(f"✨ Trovate {len(filtered_tips)} punte adatte per la crescita.")
    return filtered_tips


# Variabile globale per mantenere lo stato del branch manager tra le chiamate
_global_branch_manager = None

def animate_organic_growth(contours, width, height, base_logo_mask, hierarchy, config, single_step=None, preview_mode=False):
    """
    Anima una crescita organica complessa che si aggiunge alla maschera del logo.
    I rami crescono e si uniscono progressivamente alla forma del logo.
    Include deformazioni elastiche fluide per un effetto più organico.
    
    Args:
        contours: Lista dei contorni estratti
        width, height: Dimensioni del canvas
        base_logo_mask: Maschera base del logo
        hierarchy: Gerarchia dei contorni (opzionale, per gestire buchi e relazioni)
        config: Configurazione dell'animazione
        single_step: Se specificato, genera solo il frame per questo step (per preview)
        preview_mode: Se True, ottimizza per la preview in tempo reale
    """
    global _global_branch_manager
    
    frames = []
    
    # In modalità preview, usa un branch manager globale per mantenere lo stato
    if preview_mode:
        if _global_branch_manager is None:
            _global_branch_manager = BranchManager(config)
            _global_branch_manager.initialize_branches_from_contours(contours, hierarchy)
            print(f"🌿 Inizializzati {len(_global_branch_manager.branches)} rami per la preview")
        
        branch_manager = _global_branch_manager
    else:
        # Per il render normale, crea un nuovo branch manager
        branch_manager = BranchManager(config)
        branch_manager.initialize_branches_from_contours(contours, hierarchy)

    # Se non sono state trovate punte, ritorna solo il logo base per tutti i frame
    if not branch_manager.branches:
        if not preview_mode:
            print("⚠️ Nessuna punta trovata, usando solo il logo base.")
        # Converte la maschera base in immagine a 3 canali e la replica per tutti i frame
        base_logo_bgr = cv2.cvtColor(base_logo_mask, cv2.COLOR_GRAY2BGR)
        if single_step is not None:
            return [base_logo_bgr.copy()]
        return [base_logo_bgr.copy() for _ in range(config['animation']['steps'])]

    # Ciclo di animazione completo (sia per render che per preview)
    start_time = time.time() if not preview_mode else None
    
    for step in range(config['animation']['steps']):
        # Inizia con la maschera del logo base
        current_mask = base_logo_mask.copy()
        
        # Fa crescere e ramificare tutti i rami
        branch_manager.grow_and_ramify()
        
        # Disegna tutti i rami sulla maschera
        branch_manager.draw_all_on_mask(current_mask)

        # Ritorna solo la maschera, senza deformazioni (saranno applicate dal chiamante)
        frames.append(current_mask.copy())
        
        # Mostra progress solo in modalità render
        if not preview_mode and start_time:
            print_progress_bar(step + 1, config['animation']['steps'], start_time)

    if not preview_mode:
        print()
    return frames


def reset_global_branch_manager():
    """Resetta il branch manager globale (utile per riavviare la preview)."""
    global _global_branch_manager
    _global_branch_manager = None
