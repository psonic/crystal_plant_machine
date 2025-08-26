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


def find_sharp_tips(contour, angle_threshold_deg=150, step=10):
    """
    Trova i punti finali (endpoints) di un contorno usando la scheletrizzazione.
    Questo metodo è molto più robusto e preciso del calcolo geometrico degli angoli.
    """
    try:
        from skimage.morphology import skeletonize
        from skimage import measure
    except ImportError:
        print("⚠️ scikit-image non disponibile, uso algoritmo geometrico di fallback")
        return find_sharp_tips_geometric(contour, angle_threshold_deg, step)
    
    # Crea una maschera binaria dal contorno
    # Prima determiniamo le dimensioni necessarie
    contour_squeezed = contour.squeeze()
    if contour_squeezed.ndim == 1:
        # Se è un singolo punto, non possiamo fare scheletrizzazione
        return []
    
    mins = np.min(contour_squeezed, axis=0).astype(int)
    maxs = np.max(contour_squeezed, axis=0).astype(int)
    min_x, min_y = mins[0], mins[1]
    max_x, max_y = maxs[0], maxs[1]
    
    # Aggiungi un po' di padding
    padding = 50
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding
    
    # Crea la maschera
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Trasla il contorno nell'origine della maschera
    translated_contour = contour_squeezed - [min_x - padding, min_y - padding]
    translated_contour = translated_contour.astype(np.int32)
    
    # Riempi il contorno
    cv2.fillPoly(mask, [translated_contour], 255)
    
    # Converti in binario (True/False) per skimage
    binary_mask = mask > 0
    
    # Applica la scheletrizzazione
    skeleton = skeletonize(binary_mask)
    
    # Trova i punti finali dello scheletro
    endpoints = []
    skeleton_uint8 = skeleton.astype(np.uint8) * 255
    
    # Scansiona ogni pixel dello scheletro
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:  # Se è un pixel dello scheletro
                # Conta i vicini nell'intorno 3x3
                neighbors = skeleton[y-1:y+2, x-1:x+2]
                neighbor_count = np.sum(neighbors) - 1  # -1 per escludere il pixel centrale
                
                # Un endpoint ha esattamente 1 vicino
                if neighbor_count == 1:
                    # Trasforma le coordinate di nuovo nello spazio originale
                    original_x = x + min_x - padding
                    original_y = y + min_y - padding
                    
                    # Calcola la direzione di crescita guardando l'unico vicino
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            if skeleton[y + dy, x + dx]:
                                # Direzione opposta al vicino (per crescere "verso l'esterno")
                                angle = np.degrees(np.arctan2(-dy, -dx))
                                endpoints.append(((original_x, original_y), angle))
                                break
                        else:
                            continue
                        break
    
    print(f"🔬 Scheletrizzazione: trovati {len(endpoints)} endpoints reali")
    return endpoints


def find_sharp_tips_geometric(contour, angle_threshold_deg=150, step=10):
    """
    Algoritmo geometrico di fallback per trovare punte acute.
    Usato quando scikit-image non è disponibile.
    """
    tips = []
    contour_squeezed = contour.squeeze()
    num_points = len(contour_squeezed)

    if num_points < 2 * step:
        return []

    # Calcola il centro di massa per determinare "interno" vs "esterno"
    center = np.mean(contour_squeezed, axis=0)
    
    for i in range(num_points):
        # Prendi tre punti: corrente, precedente e successivo (step ancora più ridotto)
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

        # Prima verifica: è un angolo acuto? (soglia molto alta)
        if angle > angle_threshold_deg:
            continue
            
        # Seconda verifica: verifica che il punto sporga verso l'esterno (minimo assoluto)
        dist_curr = np.linalg.norm(p_curr - center)
        dist_prev = np.linalg.norm(p_prev - center)
        dist_next = np.linalg.norm(p_next - center)
        
        # Deve sporgere anche solo minimamente (1% di tolleranza)
        if dist_curr <= max(dist_prev, dist_next) * 1.01:
            continue
            
        # Calcola la direzione di uscita della punta
        outgoing_vector = -(v1 / norm_v1 + v2 / norm_v2)
        outgoing_angle = np.degrees(np.arctan2(outgoing_vector[1], outgoing_vector[0]))
        
        # Terza verifica: direzione ragionevole (estremamente permissiva)
        to_center = center - p_curr
        center_angle = np.degrees(np.arctan2(to_center[1], to_center[0]))
        
        # L'angolo di uscita deve solo non puntare direttamente verso il centro
        angle_diff = abs(((outgoing_angle - center_angle + 180) % 360) - 180)
        if angle_diff < 10:  # Solo 10 gradi minimi di divergenza
            continue
        
        tips.append((tuple(p_curr), outgoing_angle))
            
    # Filtra le punte troppo vicine per evitare cluster (distanza molto ridotta)
    if not tips:
        return []
        
    filtered_tips = [tips[0]]
    min_dist_sq = 20**2  # Distanza minima ulteriormente ridotta
    for i in range(1, len(tips)):
        is_far_enough = True
        for j in range(len(filtered_tips)):
            dist_sq = (tips[i][0][0] - filtered_tips[j][0][0])**2 + (tips[i][0][1] - filtered_tips[j][0][1])**2
            if dist_sq < min_dist_sq:
                is_far_enough = False
                break
        if is_far_enough:
            filtered_tips.append(tips[i])

    print(f"✨ Algoritmo geometrico: trovate {len(filtered_tips)} punte")
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
