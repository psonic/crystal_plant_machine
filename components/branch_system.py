"""
Sistema di rami organici per Crystal Plant Machine.
Gestisce la crescita, ramificazione e rendering dei rami.
"""

import cv2
import numpy as np
import random


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
        # Rami secondari ancora più fini
        self.thickness = max(1, 4 - self.generation) 
        
        # Inerzia di curvatura per curve più armoniose e attorcigliate
        if generation == 0:
            # Rami principali: crescita dritta e deterministica
            self.curvature = 0  # Nessuna curvatura iniziale
            self.curvature_change_rate = 0.2  # Variazione minima e fissa
        else:
            # Rami secondari: più casuali
            self.curvature = (random.random() - 0.5) * 15  # Curvatura iniziale aumentata
            self.curvature_change_rate = random.uniform(0.5, 1.5) # Variazione della curva aumentata
        
        # Metadati per la gerarchia
        self.is_from_hole = False  # True se nasce da un buco (contorno interno)
        self.contour_index = -1    # Indice del contorno di origine

    def grow(self):
        """Estende il ramo di un passo, applicando una curvatura armonica."""
        if self.is_complete:
            return

        last_point = self.points[-1]
        
        # Aggiorna la curvatura in modo morbido
        if self.generation == 0:
            # Rami principali: crescita più dritta e deterministica
            self.curvature += (random.random() - 0.5) * self.curvature_change_rate * 0.5  # Variazione ridotta
        else:
            # Rami secondari: più casuali
            self.curvature += (random.random() - 0.5) * self.curvature_change_rate
        # Limita la curvatura per evitare spirali troppo strette
        max_curve = 20 + self.generation * 7 # Curva massima aumentata per più "attorcigliamento"
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
        """Disegna il ramo sul canvas con un effetto di dissolvenza più marcato."""
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                progress = (i + 1) / len(self.points)
                # Dissolvenza ancora più marcata (progress^5) per un effetto più etereo
                brightness = int(200 * (1 - progress**5)) 
                color = (brightness, brightness, brightness)
                
                p1 = (int(self.points[i][0]), int(self.points[i][1]))
                p2 = (int(self.points[i+1][0]), int(self.points[i+1][1]))
                
                cv2.line(canvas, p1, p2, color, self.thickness)

    def draw_on_mask(self, mask):
        """Disegna il ramo direttamente sulla maschera in scala di grigi."""
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                progress = (i + 1) / len(self.points)
                # Intensità che decresce verso la punta del ramo
                intensity = int(255 * (1 - progress**3))  # Dissolvenza più graduale per le maschere
                
                p1 = (int(self.points[i][0]), int(self.points[i][1]))
                p2 = (int(self.points[i+1][0]), int(self.points[i+1][1]))
                
                # Disegna sulla maschera in scala di grigi
                cv2.line(mask, p1, p2, intensity, self.thickness)


class BranchManager:
    """Gestisce l'insieme di tutti i rami e la loro crescita coordinata."""
    
    def __init__(self, config):
        self.config = config
        self.branches = []
        self.initialized = False
    
    def initialize_branches_from_contours(self, contours, hierarchy):
        """Inizializza i rami una sola volta dalle punte dei contorni."""
        if self.initialized:
            return
            
        from .animation import find_sharp_tips
        import random
        
        for i, contour in enumerate(contours):
            # Determina il tipo di contorno usando la gerarchia
            is_hole = False
            if hierarchy is not None and len(hierarchy[0]) > i:
                parent_idx = hierarchy[0][i][3]
                is_hole = parent_idx != -1
            
            # Salta i buchi interni
            if is_hole:
                continue
            
            # Trova le punte
            tips = find_sharp_tips(contour)
            self.create_branches_from_tips(tips, i)
        
        self.initialized = True
    
    def create_branches_from_tips(self, tips, contour_index):
        """Crea rami iniziali dalle punte identificate di un contorno."""
        import random
        
        if not tips:
            return
            
        # Ottieni la probabilità di crescita dal config
        growth_prob = self.config.get('growth', {}).get('tip_growth_probability', 1.0)
        
        # Seleziona le punte che cresceranno in base alla probabilità
        selected_tips = []
        for tip in tips:
            if random.random() < growth_prob:
                selected_tips.append(tip)
        
        print(f"🌱 Punte trovate: {len(tips)}, selezionate per crescita: {len(selected_tips)} (prob: {growth_prob})")

        for tip_origin, tip_angle in selected_tips:
            for j in range(self.config['animation']['initial_branches_per_tip']):
                # L'angolo iniziale segue esattamente la direzione della punta (no randomizzazione)
                angle = tip_angle
                
                # Lunghezza fissa per rami principiali (no randomizzazione)
                max_len = (self.config['growth']['min_length'] + self.config['growth']['max_length']) / 2
                
                # Velocità fissa per rami principali (no randomizzazione)
                speed = (self.config['growth']['min_speed'] + self.config['growth']['max_speed']) / 2
                
                # Crea il ramo (sempre da forma esterna ora)
                branch = Branch(origin=tip_origin, initial_angle=angle, max_length=max_len, generation=0, speed_factor=speed)
                branch.is_from_hole = False  # Sempre False ora
                branch.contour_index = contour_index  # Indice del contorno di origine
                self.branches.append(branch)
    
    def grow_and_ramify(self):
        """Fa crescere tutti i rami e gestisce la ramificazione."""
        import random
        
        newly_created_branches = []
        
        for branch in self.branches:
            if not branch.is_complete:
                branch.grow()

                # Logica di ramificazione semplificata (tutti i rami sono da forme esterne)
                ramification_chance = self.config['growth']['ramification_chance'] + (branch.generation * 0.1)
                
                if (len(branch.points) > 5 and 
                    random.random() < ramification_chance and 
                    branch.generation < self.config['growth']['max_generations']):
                    
                    new_origin = branch.points[-random.randint(2, 5)]
                    new_angle = branch.angle + random.choice([-75, 75, -100, 100])
                    new_max_len = branch.max_length * random.uniform(0.15, 0.4)
                    new_speed = random.uniform(self.config['growth']['min_speed'], self.config['growth']['max_speed'])
                    
                    if new_max_len > 3:
                        new_branch = Branch(
                            origin=new_origin, 
                            initial_angle=new_angle, 
                            max_length=new_max_len, 
                            generation=branch.generation + 1, 
                            speed_factor=new_speed
                        )
                        # Eredita le proprietà del parent (sempre forma esterna)
                        new_branch.is_from_hole = False
                        new_branch.contour_index = branch.contour_index
                        newly_created_branches.append(new_branch)
        
        self.branches.extend(newly_created_branches)
    
    def draw_all_on_mask(self, mask):
        """Disegna tutti i rami sulla maschera."""
        for branch in self.branches:
            branch.draw_on_mask(mask)
