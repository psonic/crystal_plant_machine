import cv2
import numpy as np
import fitz  # PyMuPDF per leggere PDF
import random
import time
import sys
from datetime import datetime

# --- CONFIGURAZIONE DEL PROGETTO ---

class Config:
    # TEST MODE - Imposta a True per video più piccoli e veloci durante i test
    TEST_MODE = True
    
    # SVG EXPORT - Imposta a True per salvare anche un file SVG pulito
    SAVE_SVG_OUTPUT = True
    
    # Parametri video in base al test mode
    WIDTH = 640 if TEST_MODE else 1920
    HEIGHT = 360 if TEST_MODE else 1080
    ANIMATION_STEPS = 120 if TEST_MODE else 450
    FPS = 30
    START_DELAY_SECONDS = 2
    
    PDF_PATH = 'input/logo.pdf'
    LOGO_COLOR = (255, 255, 255)
    INITIAL_BRANCHES_PER_TIP = 1
    # Aggiunta per il nome del file di output
    MAGIC_SYMBOLS = ['🔮', '✨', '🌟', '🌿', '🌊']

# --- CLASSE PER LA CRESCITA ORGANICA ---

class Branch:
    """Rappresenta un singolo ramo che cresce, curva e si ramifica."""
    def __init__(self, origin, initial_angle, max_length, generation=0, speed_factor=1.0):
        self.origin = origin
        self.angle = initial_angle
        self.max_length = int(max_length)
        self.generation = generation
        self.points = [origin]
        self.is_complete = False
        # Aggiunto fattore di velocità per crescita differenziata
        self.speed_factor = speed_factor 
        # Spessore del ramo che si assottiglia con la generazione
        self.thickness = max(1, 5 - self.generation) 
        
        # Inerzia di curvatura per curve più armoniose e attorcigliate
        self.curvature = (random.random() - 0.5) * 10  # Curvatura iniziale ridotta per seguire meglio la punta
        self.curvature_change_rate = random.uniform(0.4, 1.0) # Variazione della curva ridotta

    def grow(self):
        """Estende il ramo di un passo, applicando una curvatura armonica."""
        if self.is_complete:
            return

        last_point = self.points[-1]
        
        # Aggiorna la curvatura in modo morbido
        self.curvature += (random.random() - 0.5) * self.curvature_change_rate
        # Limita la curvatura per evitare spirali troppo strette
        max_curve = 15 + self.generation * 5 # Curva massima ridotta
        self.curvature = max(-max_curve, min(max_curve, self.curvature))
        
        # Applica la curvatura all'angolo
        self.angle += self.curvature

        # La velocità di crescita ora dipende anche dal fattore di velocità individuale
        growth_speed = (5 / (self.generation + 1)**1.5) * self.speed_factor
        rad_angle = np.radians(self.angle)
        new_x = last_point[0] + growth_speed * np.cos(rad_angle)
        new_y = last_point[1] + growth_speed * np.sin(rad_angle)
        
        self.points.append((new_x, new_y))

        if len(self.points) >= self.max_length:
            self.is_complete = True

    def draw(self, canvas):
        """Disegna il ramo sul canvas con spessore affusolato."""
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                progress = (i + 1) / len(self.points)
                
                # Lo spessore si riduce lungo la lunghezza del ramo
                current_thickness = int(self.thickness * (1 - progress**2))
                current_thickness = max(1, current_thickness) # Spessore minimo di 1
                
                # Dissolvenza ancora più marcata (progress^5) per un effetto più etereo
                brightness = int(200 * (1 - progress**5)) 
                color = (brightness, brightness, brightness)
                
                p1 = (int(self.points[i][0]), int(self.points[i][1]))
                p2 = (int(self.points[i+1][0]), int(self.points[i+1][1]))
                
                cv2.line(canvas, p1, p2, color, current_thickness)

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

def animate_organic_growth(contours, hierarchy, width, height):
    """
    Anima una crescita organica complessa con ramificazioni multiple, curve e intrecci.
    Fa crescere i rami solo dai contorni esterni, non dai buchi interni.
    """
    frames = []
    
    # Inizializza i rami principali dalle punte identificate dei contorni ESTERNI
    all_branches = []
    print("🌿 Inizio identificazione delle punte per la crescita...")
    
    # Se non c'è hierarchy, usa tutti i contorni
    if hierarchy is None:
        external_contours = contours
    else:
        # Filtra solo i contorni esterni (parent == -1)
        hierarchy = hierarchy[0]  # OpenCV restituisce hierarchy in un array extra
        external_contours = [contours[i] for i, h in enumerate(hierarchy) if h[3] == -1]
    
    for contour in external_contours:
        # Trova le punte e i loro angoli di crescita
        tips = find_sharp_tips(contour)
        
        # Rende la crescita casuale: non tutte le punte germoglieranno sempre
        random.shuffle(tips)
        num_tips_to_grow = random.randint(len(tips) // 2, len(tips))
        selected_tips = tips[:num_tips_to_grow]

        for tip_origin, tip_angle in selected_tips:
            for i in range(Config.INITIAL_BRANCHES_PER_TIP):
                # L'angolo iniziale ora segue la curva della punta, con una variazione minima
                angle = tip_angle + random.uniform(-5, 5)
                max_len = random.uniform(80, 220) # Aumentata la lunghezza massima potenziale
                # Assegna una velocità di crescita casuale a ogni ramo
                speed = random.uniform(0.7, 1.2)
                branch = Branch(origin=tip_origin, initial_angle=angle, max_length=max_len, generation=0, speed_factor=speed)
                all_branches.append(branch)

    # Se non sono state trovate punte, non procedere
    if not all_branches:
        print("⚠️ Nessuna punta trovata, l'animazione della crescita non può partire.")
        # Crea N frame neri per evitare errori successivi
        return [np.zeros((height, width, 3), dtype=np.uint8)] * Config.ANIMATION_STEPS

    # Aggiungi un ritardo iniziale
    delay_frames = int(Config.START_DELAY_SECONDS * Config.FPS)
    total_animation_steps = Config.ANIMATION_STEPS - delay_frames
    
    # Frame statici per il ritardo
    for _ in range(delay_frames):
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # Ciclo di animazione
    start_time = time.time()
    for step in range(total_animation_steps):
        # Inizia con uno sfondo nero per disegnare i rami
        branch_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        newly_created_branches = []
        for branch in all_branches:
            if not branch.is_complete:
                branch.grow()

                # Logica di ramificazione: ancora più probabile e con rametti più corti e curvi
                ramification_chance = 0.08 + (branch.generation * 0.1) 
                if len(branch.points) > 5 and random.random() < ramification_chance and branch.generation < 4:
                    new_origin = branch.points[-random.randint(2, 5)]
                    # Angoli più estremi per un look più intricato
                    new_angle = branch.angle + random.choice([-75, 75, -100, 100])
                    # Rametti secondari molto più corti
                    new_max_len = branch.max_length * random.uniform(0.15, 0.4)
                    # Anche i nuovi rami hanno velocità random
                    new_speed = random.uniform(0.6, 1.1)
                    
                    if new_max_len > 3:
                        new_branch = Branch(origin=new_origin, initial_angle=new_angle, max_length=new_max_len, generation=branch.generation + 1, speed_factor=new_speed)
                        newly_created_branches.append(new_branch)
            
            branch.draw(branch_canvas)

        all_branches.extend(newly_created_branches)

        # Aggiunge semplicemente il canvas con i rami. La fusione con la texture avverrà dopo.
        frames.append(branch_canvas)
        print_progress_bar(step + 1, total_animation_steps, start_time)

    print()
    return frames