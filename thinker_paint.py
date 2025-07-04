

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

# Import MediaPipe avec gestion d'erreur
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe disponible - Détection de doigts activée")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe non disponible - Mode couleur rouge seulement")
    print("💡 Installez avec: pip install mediapipe")

class EnhancedColorPaint:
    def __init__(self):
        # Dimensions de la fenêtre (augmentées)
        self.width = 1400
        self.height = 800
        
        # Canvas de peinture
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Paramètres de peinture
        self.brush_size = 8
        self.current_color = (0, 0, 255)  # Rouge par défaut (BGR)
        self.prev_x, self.prev_y = None, None
        
        # Variables de détection
        self.detection_mode = "red"  # "red" ou "finger"
        self.finger_x, self.finger_y = None, None
        self.finger_detected = False
        
        # MediaPipe setup
        if MEDIAPIPE_AVAILABLE:
            self.setup_mediapipe()
        
        # Historique pour l'annulation
        self.canvas_history = []
        self.max_history = 20
        
        # Statistiques de performance
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Palette de couleurs étendue
        self.colors = {
            'Rouge': (0, 0, 255),
            'Vert': (0, 255, 0),
            'Bleu': (255, 0, 0),
            'Jaune': (0, 255, 255),
            'Violet': (255, 0, 255),
            'Orange': (0, 165, 255),
            'Blanc': (255, 255, 255),
            'Noir': (0, 0, 0),
            'Rose': (203, 192, 255),
            'Cyan': (255, 255, 0),
            'Marron': (42, 42, 165),
            'Gris': (128, 128, 128)
        }
        
        # Formes à dessiner
        self.shapes = {
            'Ligne': 'line',
            'Rectangle': 'rectangle',
            'Cercle': 'circle',
            'Libre': 'free'
        }
        self.current_shape = 'free'
        self.shape_zones = []
        self.setup_shape_zones()
        
        # Variables pour les formes géométriques
        self.shape_start_x, self.shape_start_y = None, None
        self.is_drawing_shape = False
        self.temp_canvas = None
        
        # Zones de couleurs
        self.color_zones = []
        self.setup_color_zones()
        
        # Outils étendus
        self.tools = {
            'Fin': 2,
            'Petit': 5,
            'Moyen': 10,
            'Gros': 18,
            'Très Gros': 30,
            'Énorme': 45
        }
        self.tool_zones = []
        self.setup_tool_zones()
        
        # Modes étendus
        self.drawing_mode = True
        self.last_selection = 0
        self.eraser_mode = False
        self.show_help = False
        self.show_stats = True
        
        # Sauvegarde automatique
        self.auto_save_interval = 30  # secondes
        self.last_auto_save = time.time()
        
        # Interface moderne
        self.ui_alpha = 0.8
        self.button_hover_effect = {}
        
        # Charger les paramètres
        self.load_settings()
        
    def setup_mediapipe(self):
        """Initialiser MediaPipe pour la détection des mains"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        print("✅ MediaPipe initialisé")
        
    def setup_color_zones(self):
        """Créer les zones de sélection de couleurs"""
        zone_width = 100
        zone_height = 60
        start_x = self.width - zone_width - 20
        
        for i, (name, color) in enumerate(self.colors.items()):
            y = 20 + i * (zone_height + 10)
            self.color_zones.append({
                'name': name,
                'color': color,
                'rect': (start_x, y, zone_width, zone_height)
            })
    
    def setup_tool_zones(self):
        """Créer les zones d'outils"""
        zone_width = 90
        zone_height = 45
        start_x = 20
        
        for i, (name, size) in enumerate(self.tools.items()):
            y = 20 + i * (zone_height + 8)
            self.tool_zones.append({
                'name': name,
                'size': size,
                'rect': (start_x, y, zone_width, zone_height)
            })
    
    def setup_shape_zones(self):
        """Créer les zones de formes"""
        zone_width = 90
        zone_height = 45
        start_x = 130
        
        for i, (name, shape_type) in enumerate(self.shapes.items()):
            y = 20 + i * (zone_height + 8)
            self.shape_zones.append({
                'name': name,
                'type': shape_type,
                'rect': (start_x, y, zone_width, zone_height)
            })
    
    def load_settings(self):
        """Charger les paramètres depuis un fichier"""
        try:
            if os.path.exists('paint_settings.json'):
                with open('paint_settings.json', 'r') as f:
                    settings = json.load(f)
                    self.detection_sensitivity = settings.get('detection_sensitivity', 20)
                    self.auto_save_interval = settings.get('auto_save_interval', 30)
                    self.show_stats = settings.get('show_stats', True)
        except Exception as e:
            print(f"⚠️ Impossible de charger les paramètres: {e}")
    
    def save_settings(self):
        """Sauvegarder les paramètres"""
        try:
            settings = {
                'detection_sensitivity': self.detection_sensitivity,
                'auto_save_interval': self.auto_save_interval,
                'show_stats': self.show_stats
            }
            with open('paint_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"⚠️ Impossible de sauvegarder les paramètres: {e}")
    
    def add_to_history(self):
        """Ajouter l'état actuel à l'historique"""
        if len(self.canvas_history) >= self.max_history:
            self.canvas_history.pop(0)
        self.canvas_history.append(self.canvas.copy())
    
    def undo_last_action(self):
        """Annuler la dernière action"""
        if self.canvas_history:
            self.canvas = self.canvas_history.pop()
            print("♾️ Action annulée!")
        else:
            print("⚠️ Aucune action à annuler")
    
    def draw_interface(self, frame):
        """Dessiner l'interface utilisateur moderne"""
        # Fond semi-transparent pour l'interface
        overlay = frame.copy()
        
        # Titre avec style moderne
        title_text = "🎨 THINKER PAINT ENHANCED"
        cv2.putText(frame, title_text, (self.width//2 - 250, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(frame, title_text, (self.width//2 - 250, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 200), 2)
        
        # Dessiner les zones de couleurs
        for zone in self.color_zones:
            x, y, w, h = zone['rect']
            color = zone['color']
            
            # Rectangle de couleur
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Bordure
            border_color = (255, 255, 255) if color == self.current_color else (128, 128, 128)
            thickness = 4 if color == self.current_color else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, thickness)
            
            # Nom
            text_color = (0, 0, 0) if color in [(255, 255, 255), (0, 255, 255)] else (255, 255, 255)
            cv2.putText(frame, zone['name'], (x + 5, y + h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Dessiner les zones d'outils avec style moderne
        for zone in self.tool_zones:
            x, y, w, h = zone['rect']
            
            # Couleur de fond avec gradient
            bg_color = (80, 120, 160) if zone['size'] == self.brush_size else (40, 60, 80)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
            
            # Bordure moderne
            border_color = (150, 200, 255) if zone['size'] == self.brush_size else (100, 150, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
            
            # Texte avec ombre
            cv2.putText(frame, zone['name'], (x + 6, y + 26), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Ombre
            cv2.putText(frame, zone['name'], (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"({zone['size']}px)", (x + 6, y + 41), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)  # Ombre
            cv2.putText(frame, f"({zone['size']}px)", (x + 5, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Dessiner les zones de formes
        for zone in self.shape_zones:
            x, y, w, h = zone['rect']
            
            # Couleur de fond
            bg_color = (120, 80, 160) if zone['type'] == self.current_shape else (60, 40, 80)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
            
            # Bordure
            border_color = (200, 150, 255) if zone['type'] == self.current_shape else (150, 100, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
            
            # Texte
            cv2.putText(frame, zone['name'], (x + 6, y + 26), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, zone['name'], (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panneau d'aide moderne
        if self.show_help:
            help_bg = np.zeros((200, 400, 3), dtype=np.uint8)
            help_bg[:] = (40, 40, 40)
            
            help_texts = [
                "🔴 Objet ROUGE = Curseur",
                "🎨 Clic zones = Sélection",
                "ESC=Quitter | S=Sauver",
                "C=Effacer | Z=Annuler",
                "ESPACE=Mode | H=Aide",
                "E=Gomme | R=Reset"
            ]
            
            for i, text in enumerate(help_texts):
                cv2.putText(help_bg, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Position de l'aide
            help_x, help_y = self.width - 420, self.height - 220
            frame[help_y:help_y+200, help_x:help_x+400] = help_bg
        
        # Instructions compactes
        quick_help = "🔴 Objet Rouge = Curseur | H = Aide Complète"
        cv2.putText(frame, quick_help, (20, self.height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # Panneau d'informations moderne
        info_bg = (30, 30, 30)
        info_rect = (self.width//2 - 200, 60, 400, 120)
        cv2.rectangle(frame, (info_rect[0], info_rect[1]), 
                     (info_rect[0] + info_rect[2], info_rect[1] + info_rect[3]), info_bg, -1)
        cv2.rectangle(frame, (info_rect[0], info_rect[1]), 
                     (info_rect[0] + info_rect[2], info_rect[1] + info_rect[3]), (100, 100, 100), 2)
        
        # Mode actuel
        mode_text = "🎨 DESSIN" if self.drawing_mode else "🔍 SÉLECTION"
        if self.eraser_mode:
            mode_text = "🧹 GOMME"
        mode_color = (0, 255, 0) if self.drawing_mode else (0, 255, 255)
        if self.eraser_mode:
            mode_color = (0, 100, 255)
        
        cv2.putText(frame, f"Mode: {mode_text}", (info_rect[0] + 10, info_rect[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Informations actuelles
        color_name = self.get_color_name()
        shape_name = self.get_shape_name()
        detection_name = "DOIGT" if self.detection_mode == "finger" else "OBJET ROUGE"
        
        cv2.putText(frame, f"Couleur: {color_name}", (info_rect[0] + 10, info_rect[1] + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.current_color, 2)
        cv2.putText(frame, f"Taille: {self.brush_size}px | Forme: {shape_name}", 
                   (info_rect[0] + 10, info_rect[1] + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Détection: {detection_name}", 
                   (info_rect[0] + 10, info_rect[1] + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        # Statistiques de performance
        if self.show_stats:
            stats_text = f"FPS: {self.current_fps:.1f} | Détection: {self.detection_sensitivity}"
            cv2.putText(frame, stats_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
    
    def get_color_name(self):
        """Obtenir le nom de la couleur actuelle"""
        for name, color in self.colors.items():
            if color == self.current_color:
                return name
        return "Personnalisée"
    
    def get_shape_name(self):
        """Obtenir le nom de la forme actuelle"""
        for name, shape_type in self.shapes.items():
            if shape_type == self.current_shape:
                return name
        return "Libre"
    
    def detect_finger(self, frame):
        """Détecter l'index avec MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            return None, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        self.finger_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessiner les points de repère
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Obtenir la position de l'index (point 8) et du pouce (point 4)
                index_tip = hand_landmarks.landmark[8]
                index_pip = hand_landmarks.landmark[6]
                thumb_tip = hand_landmarks.landmark[4]
                
                h, w, _ = frame.shape
                
                # Vérifier si l'index est levé (geste de pointage)
                index_up = index_tip.y < index_pip.y
                
                if index_up:
                    # Convertir en coordonnées de canvas
                    self.finger_x = int(index_tip.x * self.width)
                    self.finger_y = int(index_tip.y * self.height)
                    
                    # Vérifier les limites
                    if (0 <= self.finger_x <= self.width and 
                        0 <= self.finger_y <= self.height):
                        self.finger_detected = True
                        
                        # Dessiner curseur sur webcam
                        cursor_x = int(index_tip.x * w)
                        cursor_y = int(index_tip.y * h)
                        
                        cv2.circle(frame, (cursor_x, cursor_y), 15, (0, 255, 0), -1)
                        cv2.circle(frame, (cursor_x, cursor_y), 25, (0, 255, 0), 3)
                        
                        # Coordonnées et état
                        cv2.putText(frame, f"INDEX ({self.finger_x},{self.finger_y})", 
                                  (cursor_x - 80, cursor_y - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cv2.putText(frame, "POINTE", 
                                  (cursor_x - 30, cursor_y + 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        return self.finger_x, self.finger_y
                else:
                    # Index non levé - afficher statut
                    cv2.putText(frame, "MAIN FERMEE - PAS DE DESSIN", 
                              (50, h - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        
        return None, None
    
    def detect_color_object(self, frame):
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Définir la plage de rouge
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Créer les masques pour le rouge
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Nettoyer le masque
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Prendre le plus grand contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Vérifier si le contour est assez grand
            if cv2.contourArea(largest_contour) > 500:
                # Calculer le centre
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Dessiner le contour et le centre
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    
                    return cx, cy
        
        return None, None
    
    def draw_shape(self, start_x, start_y, end_x, end_y, shape_type):
        """Dessiner une forme géométrique"""
        if shape_type == 'line':
            cv2.line(self.canvas, (start_x, start_y), (end_x, end_y), 
                    self.current_color, self.brush_size)
        elif shape_type == 'rectangle':
            cv2.rectangle(self.canvas, (start_x, start_y), (end_x, end_y), 
                         self.current_color, self.brush_size)
        elif shape_type == 'circle':
            center_x = (start_x + end_x) // 2
            center_y = (start_y + end_y) // 2
            radius = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) // 2)
            cv2.circle(self.canvas, (center_x, center_y), radius, 
                      self.current_color, self.brush_size)
    
    def apply_eraser(self, x, y):
        """Appliquer la gomme"""
        eraser_size = self.brush_size * 2
        cv2.circle(self.canvas, (x, y), eraser_size, (0, 0, 0), -1)
    
    def check_zone_click(self, x, y):
        """Vérifier les clics sur les zones avec feedback sonore"""
        current_time = time.time()
        
        # Éviter les sélections trop rapides
        if current_time - self.last_selection < 0.3:
            return
        
        # Vérifier les couleurs
        for zone in self.color_zones:
            zone_x, zone_y, zone_w, zone_h = zone['rect']
            if zone_x <= x <= zone_x + zone_w and zone_y <= y <= zone_y + zone_h:
                self.current_color = zone['color']
                self.last_selection = current_time
                print(f"🎨 Couleur changée: {zone['name']}")
                return
        
        # Vérifier les outils
        for zone in self.tool_zones:
            zone_x, zone_y, zone_w, zone_h = zone['rect']
            if zone_x <= x <= zone_x + zone_w and zone_y <= y <= zone_y + zone_h:
                self.brush_size = zone['size']
                self.last_selection = current_time
                print(f"🔧 Taille changée: {zone['name']} ({zone['size']}px)")
                return
        
        # Vérifier les formes
        for zone in self.shape_zones:
            zone_x, zone_y, zone_w, zone_h = zone['rect']
            if zone_x <= x <= zone_x + zone_w and zone_y <= y <= zone_y + zone_h:
                self.current_shape = zone['type']
                self.last_selection = current_time
                print(f"🔶 Forme changée: {zone['name']}")
                return
    
    def process_frame(self, frame):
        """Traiter un frame de la webcam avec fonctionnalités avancées"""
        # Calcul du FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # Effet miroir
        frame = cv2.flip(frame, 1)
        
        # Redimensionner
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Détection selon le mode
        if self.detection_mode == "finger" and MEDIAPIPE_AVAILABLE:
            x, y = self.detect_finger(frame)
        else:
            x, y = self.detect_color_object(frame)
        
        if x is not None and y is not None:
            if self.drawing_mode:
                if self.eraser_mode:
                    # Mode gomme
                    self.apply_eraser(x, y)
                    cv2.putText(frame, "🧹 GOMME", (x - 40, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                elif self.current_shape == 'free':
                    # Dessin libre
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                                self.current_color, self.brush_size)
                    
                    self.prev_x, self.prev_y = x, y
                    
                    # Indicateur de dessin
                    cv2.putText(frame, "🎨 DESSIN", (x - 40, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Formes géométriques
                    if not self.is_drawing_shape:
                        # Début de forme
                        self.shape_start_x, self.shape_start_y = x, y
                        self.is_drawing_shape = True
                        self.temp_canvas = self.canvas.copy()
                        self.add_to_history()
                    else:
                        # Prévisualisation de la forme
                        temp_preview = self.temp_canvas.copy()
                        self.draw_shape(self.shape_start_x, self.shape_start_y, x, y, self.current_shape)
                        
                        # Afficher prévisualisation
                        shape_name = self.get_shape_name()
                        cv2.putText(frame, f"🔶 {shape_name.upper()}", (x - 50, y - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            else:
                # Mode sélection
                self.check_zone_click(x, y)
                self.prev_x, self.prev_y = None, None
                self.is_drawing_shape = False
                
                # Indicateur de sélection
                cv2.putText(frame, "🔍 SELECT", (x - 50, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Aucun objet détecté
            if self.is_drawing_shape and self.current_shape != 'free':
                # Finaliser la forme
                if self.shape_start_x is not None and self.shape_start_y is not None:
                    # La forme est terminée, pas besoin de coordonnées finales
                    pass
                self.is_drawing_shape = False
            
            self.prev_x, self.prev_y = None, None
        
        # Sauvegarde automatique
        if time.time() - self.last_auto_save > self.auto_save_interval:
            self.auto_save()
            self.last_auto_save = time.time()
        
        # Superposer le canvas
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        
        canvas_area = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        frame_area = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        combined = cv2.add(frame_area, canvas_area)
        
        # Dessiner l'interface
        self.draw_interface(combined)
        
        return combined
    
    def auto_save(self):
        """Sauvegarde automatique"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"auto_save_{timestamp}.png"
        
        if not os.path.exists("auto_saves"):
            os.makedirs("auto_saves")
        
        filepath = os.path.join("auto_saves", filename)
        cv2.imwrite(filepath, self.canvas)
        print(f"💾 Sauvegarde automatique: {filepath}")
    
    def reset_canvas(self):
        """Remettre à zéro le canvas"""
        self.add_to_history()  # Sauvegarder l'état actuel avant reset
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        print("🔄 Canvas remis à zéro!")
    
    def clear_canvas(self):
        """Effacer le canvas avec confirmation"""
        self.add_to_history()  # Sauvegarder avant effacement
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        print("🧹 Canvas effacé!")
    
    def save_painting(self):
        """Sauvegarder la peinture avec métadonnées"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thinker_paint_{timestamp}.png"
        
        # Créer les dossiers nécessaires
        os.makedirs("peintures", exist_ok=True)
        os.makedirs("metadata", exist_ok=True)
        
        # Sauvegarder l'image
        filepath = os.path.join("peintures", filename)
        cv2.imwrite(filepath, self.canvas)
        
        # Sauvegarder les métadonnées
        metadata = {
            "filename": filename,
            "timestamp": timestamp,
            "dimensions": {"width": self.width, "height": self.height},
            "settings": {
                "brush_size": self.brush_size,
                "current_color": self.current_color,
                "current_shape": self.current_shape,
                "detection_sensitivity": self.detection_sensitivity
            }
        }
        
        metadata_file = os.path.join("metadata", f"meta_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Peinture sauvegardée: {filepath}")
        print(f"📄 Métadonnées: {metadata_file}")
    
    def run(self):
        """Démarrer l'application améliorée"""
        print("🎨 Démarrage de Thinker Paint Enhanced")
        print("=" * 60)
        print("📹 Ouverture de la webcam...")
        
        # Créer les dossiers nécessaires
        os.makedirs("peintures", exist_ok=True)
        os.makedirs("auto_saves", exist_ok=True)
        os.makedirs("metadata", exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la webcam!")
            return
        
        # Configuration optimale de la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire la latence
        
        print("✅ Webcam ouverte!")
        print("🔴 Utilisez un objet ROUGE pour dessiner")
        print("🎯 Contrôles améliorés:")
        print("   - ESC/Q: Quitter")
        print("   - S: Sauvegarder")
        print("   - C: Effacer")
        print("   - Z: Annuler (Undo)")
        print("   - R: Reset complet")
        print("   - E: Mode gomme")
        print("   - ESPACE: Changer de mode (Dessin/Sélection)")
        print("   - H: Afficher/Masquer l'aide")
        print("   - T: Afficher/Masquer les stats")
        print("   - F: Basculer détection (Rouge/Doigt)")
        print("   - +/-: Ajuster la sensibilité de détection")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Impossible de lire le frame de la webcam")
                    break
                
                # Traiter le frame
                processed_frame = self.process_frame(frame)
                
                # Afficher
                cv2.imshow("🎨 Thinker Paint Enhanced - Détection de Couleur", processed_frame)
                
                # Gestion des touches améliorée
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC ou Q
                    print("⏹️ Arrêt demandé")
                    break
                elif key == ord('s') or key == ord('S'):  # Sauvegarder
                    self.save_painting()
                elif key == ord('c') or key == ord('C'):  # Effacer
                    self.clear_canvas()
                elif key == ord('z') or key == ord('Z'):  # Annuler
                    self.undo_last_action()
                elif key == ord('r') or key == ord('R'):  # Reset
                    self.reset_canvas()
                elif key == ord('e') or key == ord('E'):  # Mode gomme
                    self.eraser_mode = not self.eraser_mode
                    mode = "ACTIVÉ" if self.eraser_mode else "DÉSACTIVÉ"
                    print(f"🧹 Mode gomme {mode}")
                elif key == ord(' '):  # ESPACE - Changer de mode
                    self.drawing_mode = not self.drawing_mode
                    mode = "DESSIN" if self.drawing_mode else "SÉLECTION"
                    print(f"🔄 Mode changé: {mode}")
                elif key == ord('h') or key == ord('H'):  # Aide
                    self.show_help = not self.show_help
                    print(f"❓ Aide {'affichée' if self.show_help else 'masquée'}")
                elif key == ord('t') or key == ord('T'):  # Stats
                    self.show_stats = not self.show_stats
                    print(f"📊 Stats {'affichées' if self.show_stats else 'masquées'}")
                elif key == ord('+') or key == ord('='):  # Augmenter sensibilité
                    self.detection_sensitivity = min(100, self.detection_sensitivity + 5)
                    print(f"🔺 Sensibilité: {self.detection_sensitivity}")
                elif key == ord('-') or key == ord('_'):  # Diminuer sensibilité
                    self.detection_sensitivity = max(10, self.detection_sensitivity - 5)
                    print(f"🔻 Sensibilité: {self.detection_sensitivity}")
                elif key == ord('1'):  # Raccourcis couleurs
                    self.current_color = (0, 0, 255)  # Rouge
                    print("🔴 Rouge sélectionné")
                elif key == ord('2'):
                    self.current_color = (0, 255, 0)  # Vert
                    print("🟢 Vert sélectionné")
                elif key == ord('3'):
                    self.current_color = (255, 0, 0)  # Bleu
                    print("🔵 Bleu sélectionné")
                elif key == ord('f') or key == ord('F'):  # Basculer mode détection
                    if MEDIAPIPE_AVAILABLE:
                        self.detection_mode = "finger" if self.detection_mode == "red" else "red"
                        mode_name = "DOIGT" if self.detection_mode == "finger" else "OBJET ROUGE"
                        print(f"🔄 Mode détection: {mode_name}")
                    else:
                        print("⚠️ MediaPipe non disponible - Mode rouge seulement")
        
        except KeyboardInterrupt:
            print("⏹️ Arrêt demandé par l'utilisateur (Ctrl+C)")
        
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
        
        finally:
            # Sauvegarder les paramètres avant de quitter
            self.save_settings()
            
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Thinker Paint Enhanced fermé")
            print(f"📊 Statistiques finales:")
            print(f"   - FPS moyen: {self.current_fps:.1f}")
            print(f"   - Actions dans l'historique: {len(self.canvas_history)}")
            print(f"   - Sensibilité finale: {self.detection_sensitivity}")

def main():
    """Fonction principale"""
    print("🎨 THINKER PAINT ENHANCED")
    print("=" * 60)
    print("🚀 Version améliorée avec détection de couleur rouge")
    print("📅 Fonctionnalités:")
    print("   ✓ Détection de couleur rouge optimisée")
    print("   ✓ Interface utilisateur moderne")
    print("   ✓ Formes géométriques (ligne, rectangle, cercle)")
    print("   ✓ Mode gomme")
    print("   ✓ Historique et annulation (Undo)")
    print("   ✓ Sauvegarde automatique")
    print("   ✓ Métadonnées des créations")
    print("   ✓ Statistiques de performance")
    print("   ✓ Paramètres ajustables")
    print()
    
    try:
        app = EnhancedColorPaint()
        app.run()
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("💡 Installez les dépendances: pip install opencv-python numpy")
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()

