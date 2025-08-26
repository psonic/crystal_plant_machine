"""
🎭 Animations Module - Crystal Plant Generator
Gestisce tutta la logica di animazione del logo.
"""

import cv2
import numpy as np
import math


class LogoAnimator:
    """Classe per generare frame animati del logo."""
    
    def __init__(self, base_img, config):
        """
        Inizializza l'animatore.
        
        Args:
            base_img: Immagine base del logo
            config: Configurazione del progetto
        """
        self.base_img = base_img
        self.config = config
        self.height, self.width = base_img.shape[:2]
        
        # Parametri video
        video_config = config.get('video', {})
        self.fps = video_config.get('fps', 30)
        self.duration = video_config.get('duration_seconds', 30)
        self.total_frames = self.fps * self.duration
    
    def get_frame(self, frame_num, animation_type="static"):
        """
        Genera un singolo frame animato.
        
        Args:
            frame_num: Numero del frame (0 to total_frames-1)
            animation_type: Tipo di animazione
        
        Returns:
            np.array: Frame animato
        """
        # Normalizza progresso (0.0 -> 1.0)
        progress = frame_num / self.total_frames if self.total_frames > 0 else 0
        
        if animation_type == "static":
            return self.base_img.copy()
        elif animation_type == "fade":
            return self._animate_fade(progress)
        elif animation_type == "zoom":
            return self._animate_zoom(progress)
        elif animation_type == "pulse":
            return self._animate_pulse(progress)
        elif animation_type == "rotate":
            return self._animate_rotate(progress)
        elif animation_type == "slide":
            return self._animate_slide(progress)
        else:
            return self.base_img.copy()
    
    def get_total_frames(self):
        """Restituisce il numero totale di frame."""
        return self.total_frames
    
    def _animate_fade(self, progress):
        """Animazione fade in/out."""
        # Fade in primi 25%, statico 50%, fade out ultimi 25%
        if progress < 0.25:
            alpha = progress / 0.25
        elif progress > 0.75:
            alpha = (1.0 - progress) / 0.25
        else:
            alpha = 1.0
        
        # Sfondo nero
        black_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return cv2.addWeighted(self.base_img, alpha, black_bg, 1-alpha, 0)
    
    def _animate_zoom(self, progress):
        """Animazione zoom sinusoidale."""
        # Zoom da 0.8x a 1.2x sinusoidalmente
        zoom_factor = 1.0 + 0.2 * math.sin(progress * 2 * math.pi)
        
        # Calcola nuove dimensioni
        new_width = int(self.width * zoom_factor)
        new_height = int(self.height * zoom_factor)
        
        # Ridimensiona
        resized = cv2.resize(self.base_img, (new_width, new_height))
        
        # Centra su sfondo nero
        result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calcola offset per centrare
        y_offset = (self.height - new_height) // 2
        x_offset = (self.width - new_width) // 2
        
        # Inserisci l'immagine centrata
        if new_height <= self.height and new_width <= self.width:
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        else:
            # Crop se troppo grande
            crop_y = max(0, (new_height - self.height) // 2)
            crop_x = max(0, (new_width - self.width) // 2)
            result = resized[crop_y:crop_y+self.height, crop_x:crop_x+self.width]
        
        return result
    
    def _animate_pulse(self, progress):
        """Animazione pulse (variazione alpha)."""
        # Pulse veloce
        alpha = 0.4 + 0.6 * (math.sin(progress * 8 * math.pi) * 0.5 + 0.5)
        
        black_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return cv2.addWeighted(self.base_img, alpha, black_bg, 1-alpha, 0)
    
    def _animate_rotate(self, progress):
        """Animazione rotazione."""
        # Rotazione completa in 360 gradi
        angle = progress * 360
        
        # Centro di rotazione
        center = (self.width // 2, self.height // 2)
        
        # Matrice di rotazione
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Applica rotazione
        return cv2.warpAffine(self.base_img, rotation_matrix, (self.width, self.height))
    
    def _animate_slide(self, progress):
        """Animazione sliding da sinistra a destra."""
        # Movimento sinusoidale da -width/2 a +width/2
        offset_x = int((self.width / 4) * math.sin(progress * 2 * math.pi))
        
        # Crea sfondo nero
        result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calcola bounds per evitare errori
        src_start_x = max(0, -offset_x)
        src_end_x = min(self.width, self.width - offset_x)
        dst_start_x = max(0, offset_x)
        dst_end_x = min(self.width, self.width + offset_x)
        
        # Copia la porzione visibile
        if src_end_x > src_start_x and dst_end_x > dst_start_x:
            result[:, dst_start_x:dst_end_x] = self.base_img[:, src_start_x:src_end_x]
        
        return result


def get_available_animations():
    """Restituisce lista delle animazioni disponibili."""
    return ["static", "fade", "zoom", "pulse", "rotate", "slide"]


def create_animator(base_img, config):
    """
    Factory function per creare un animator.
    
    Args:
        base_img: Immagine base del logo
        config: Configurazione del progetto
    
    Returns:
        LogoAnimator: Istanza dell'animatore
    """
    return LogoAnimator(base_img, config)
