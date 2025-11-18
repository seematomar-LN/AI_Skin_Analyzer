# skin_scanner_minimalist.py
import io
from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import mediapipe as mp

# -------------------------
# UTILITIES
# -------------------------
def read_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def check_image_quality(bgr: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 70:
        return True, "Image is too blurry. Please use a clearer photo."
    mean_brightness = np.mean(gray)
    if mean_brightness < 50:
        return False, "Image is too dark. Please use better lighting."
    if mean_brightness > 240:
        return False, "Image is severely overexposed. Please reduce lighting."
    overexposed_pixels = np.sum(gray > 250) / gray.size
    if overexposed_pixels > 0.3:
        return False, "Image is overexposed. Please use softer lighting."
    return True, "OK"

# -------------------------
# MEDIAPIPE FACEMESH
# -------------------------
mp_face = mp.solutions.face_mesh

def get_landmarks(image_bgr: np.ndarray) -> Tuple[List[Tuple[int,int]], Tuple[int,int]]:
    h, w = image_bgr.shape[:2]
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return [], (h,w)
        lm = res.multi_face_landmarks[0].landmark
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        return pts, (h,w)

def draw_landmarks_on_face(image_bgr: np.ndarray, landmarks: List[Tuple[int,int]], concern: str) -> np.ndarray:
    """Draw subtle transparent highlight on face based on concern type."""
    img_with_highlight = image_bgr.copy().astype(np.float32)
    h, w = img_with_highlight.shape[:2]

    # Create highlight mask
    highlight_mask = np.zeros((h, w), dtype=np.float32)
    region_hulls = []

    # Define landmark groups for different concerns
    concern_landmarks = {
        'Dark Circles': {
            'left': [145, 153, 154, 155, 133, 130, 243, 112, 26, 22, 23, 24, 110, 25],
            'right': [374, 380, 381, 382, 263, 359, 463, 341, 256, 252, 253, 254, 339, 255]
        },
        'Pores': {
            'nose': [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 75, 97, 326, 305, 290, 289, 60, 20],
            'cheeks': [50, 101, 36, 205, 206, 280, 330, 266, 425, 426]
        },
        'Pigmentation': {
            'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
            'cheeks': [50, 187, 207, 216, 212, 202, 280, 411, 427, 436, 432, 422]
        },
        'Acne': {
            'tzone': [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 51, 281, 48, 278],
            'cheeks': [50, 101, 36, 205, 206, 207, 216, 280, 330, 266, 425, 426, 427, 436]
        },
        'Redness': {
            'central': [1, 2, 98, 327, 168, 6, 50, 101, 118, 205, 206, 280, 330, 425, 426]
        },
        'Lines': {
            'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 362, 382, 381, 380, 374, 373]
        },
        'Texture': {
            'face': [10, 338, 297, 332, 50, 280, 187, 411, 101, 330, 234, 454]
        },
        'Hydration': {
            'face': [10, 338, 297, 332, 50, 280, 187, 411, 101, 330, 234, 454, 152, 377]
        },
        'Uniformness': {
            'cheeks': [50, 187, 207, 216, 212, 202, 280, 411, 427, 436, 432, 422, 101, 330]
        },
        'Skin Tone': {
            'face': [10, 338, 297, 332, 50, 280, 187, 411, 101, 330, 234, 454, 152, 377]
        }
    }
    
    # Color scheme (BGR format) - softer colors
    concern_colors = {
        'Dark Circles': (180, 100, 220),  # Light Purple
        'Pores': (100, 200, 200),  # Light Cyan
        'Pigmentation': (100, 180, 255),  # Light Orange
        'Acne': (100, 100, 255),  # Light Red
        'Redness': (100, 120, 255),  # Soft Red
        'Lines': (200, 200, 200),  # Light Gray
        'Texture': (200, 180, 100),  # Light Blue
        'Hydration': (255, 220, 180),  # Very Light Blue
        'Uniformness': (150, 200, 255),  # Light Yellow
        'Skin Tone': (150, 200, 250)  # Light Golden
    }
    
    color = concern_colors.get(concern, (100, 255, 100))
    landmark_groups = concern_landmarks.get(concern, {'default': [10, 50, 280, 187, 411]})
    # Draw filled polygons on mask and keep hulls for dotted outlines
    for region_name, indices in landmark_groups.items():
        if len(indices) >= 3:
            points = []
            for idx in indices:
                if idx < len(landmarks):
                    points.append(landmarks[idx])

            if len(points) >= 3:
                points_array = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points_array)
                cv2.fillConvexPoly(highlight_mask, hull, 1.0)
                region_hulls.append(hull)
    
    # Apply strong Gaussian blur for soft glow effect
    kernel_size = max(51, int(0.1 * max(h, w)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    highlight_mask = cv2.GaussianBlur(highlight_mask, (kernel_size, kernel_size), 0)
    
    # NOTE: User requested no large shaded fill — only highlight markers.
    # So we skip the translucent colored fill and keep the original image as base.
    result = np.clip(img_with_highlight, 0, 255).astype(np.uint8)
    
    # --- Minimal single-color highlighting (no lines, no dots) ---
    # The user requested only a single-color highlight that follows the region shape.
    # Use a single neutral highlight color (BGR). Adjust this color to taste.
    HIGHLIGHT_BGR = (80, 150, 255)

    # Create colored highlight using the (already) blurred mask
    highlight = np.zeros_like(img_with_highlight)
    highlight[:, :, 0] = HIGHLIGHT_BGR[0] * highlight_mask  # B
    highlight[:, :, 1] = HIGHLIGHT_BGR[1] * highlight_mask  # G
    highlight[:, :, 2] = HIGHLIGHT_BGR[2] * highlight_mask  # R

    # Blend the highlight softly over the image; no outlines or dots
    alpha = 0.28
    result = cv2.addWeighted(result, 1.0, highlight.astype(np.uint8), alpha, 0)

    return result

# -------------------------
# MASKS
# -------------------------
def poly_mask(shape: Tuple[int,int], pts: List[Tuple[int,int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, np.array(pts, np.int32), 255)
    return mask

def hull_points(landmarks: List[Tuple[int,int]], indices: List[int]) -> List[Tuple[int,int]]:
    return [landmarks[i] for i in indices if i < len(landmarks)]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_UNDER = [145, 153, 154, 155, 133, 130, 243, 112, 26]
RIGHT_UNDER = [374, 380, 381, 382, 263, 359, 463, 341, 256]
LEFT_CHEEK = [50, 187, 207, 216, 212, 202, 204, 194, 135, 138, 215, 213, 192, 177, 137]
RIGHT_CHEEK = [280, 411, 427, 436, 432, 422, 424, 418, 364, 367, 435, 433, 416, 401, 366]
NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 75, 97, 2, 326, 305]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def build_region_masks(landmarks: List[Tuple[int,int]], shape: Tuple[int,int]) -> Dict[str, np.ndarray]:
    h, w = shape
    masks = {}
    if landmarks:
        hull = cv2.convexHull(np.array(landmarks, np.int32))
        face_mask = np.zeros((h,w), np.uint8)
        cv2.fillConvexPoly(face_mask, hull[:,0], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    else:
        face_mask = np.ones((h,w), np.uint8) * 255
    masks['face'] = face_mask
    masks['left_eye'] = poly_mask((h,w), hull_points(landmarks, LEFT_EYE))
    masks['right_eye'] = poly_mask((h,w), hull_points(landmarks, RIGHT_EYE))
    masks['left_under'] = poly_mask((h,w), hull_points(landmarks, LEFT_UNDER))
    masks['right_under'] = poly_mask((h,w), hull_points(landmarks, RIGHT_UNDER))
    masks['under_eyes'] = cv2.bitwise_or(masks['left_under'], masks['right_under'])
    masks['left_cheek'] = poly_mask((h,w), hull_points(landmarks, LEFT_CHEEK))
    masks['right_cheek'] = poly_mask((h,w), hull_points(landmarks, RIGHT_CHEEK))
    masks['cheeks'] = cv2.bitwise_or(masks['left_cheek'], masks['right_cheek'])
    masks['nose'] = poly_mask((h,w), hull_points(landmarks, NOSE))
    masks['forehead'] = poly_mask((h,w), hull_points(landmarks, FOREHEAD))
    masks['tzone'] = cv2.bitwise_or(masks['nose'], masks['forehead'])
    return masks

# -------------------------
# SKIN SEGMENTATION
# -------------------------
def enhanced_skin_mask(image_bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv1 = np.array([0, 15, 40], dtype=np.uint8)
    upper_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    lower_hsv2 = np.array([0, 10, 60], dtype=np.uint8)
    upper_hsv2 = np.array([20, 150, 255], dtype=np.uint8)
    mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    b, g, r = cv2.split(image_bgr)
    mask_rgb = ((r > g) & (g > b) & (r > 60) & (r - g > 15)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.bitwise_or(mask, mask_rgb)
    mask = cv2.bitwise_and(mask, face_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        min_size = face_mask.sum() * 0.01 / 255
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
    return mask

# -------------------------
# CONCERN DETECTION
# -------------------------
def enhanced_redness_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_excess = np.clip((r - g) + (r - b), 0, None) / (r + 1e-6)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_hue_mask = ((h <= 15) | (h >= 165)).astype(np.float32)
    hsv_redness = red_hue_mask * (s / 255.0) * (v / 255.0)
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel = lab[..., 1]
    a_normalized = np.clip((a_channel - 128) / 30.0, 0, None)
    combined_redness = (0.4 * red_excess * 100 + 0.3 * hsv_redness * 255 + 0.3 * a_normalized * 100)
    combined_redness[skin_mask == 0] = 0
    if np.any(skin_mask > 0):
        skin_values = combined_redness[skin_mask > 0]
        median_val = np.median(skin_values)
        p90 = np.percentile(skin_values, 90)
        score = float(np.clip((p90 - median_val) * 3.0, 0, 100))
    else:
        score = 0.0
    return score, combined_redness

def enhanced_pigmentation_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = lab[..., 0], lab[..., 1], lab[..., 2]
    if not np.any(skin_mask > 0):
        return 0.0, np.zeros_like(l_channel)
    skin_l = l_channel[skin_mask > 0]
    skin_a = a_channel[skin_mask > 0]
    skin_b = b_channel[skin_mask > 0]
    l_median = np.median(skin_l)
    a_median = np.median(skin_a)
    b_median = np.median(skin_b)
    l_dev = np.abs(l_channel - l_median)
    a_dev = np.abs(a_channel - a_median)
    b_dev = np.abs(b_channel - b_median)
    pigment_map = np.sqrt(0.5 * l_dev**2 + 0.2 * a_dev**2 + 0.3 * b_dev**2)
    pigment_map_filtered = cv2.bilateralFilter(pigment_map.astype(np.float32), 9, 75, 75)
    pigment_map_filtered[skin_mask == 0] = 0
    skin_pigment = pigment_map_filtered[skin_mask > 0]
    std_pigment = np.std(skin_pigment)
    p90 = np.percentile(skin_pigment, 90)
    high_pigment_ratio = np.sum(skin_pigment > p90 * 0.7) / len(skin_pigment)
    score = float(np.clip(std_pigment / 2.0 + high_pigment_ratio * 40, 0, 100))
    return score, pigment_map_filtered

def enhanced_acne_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel = lab[..., 1]
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    acne_hue = ((h_ch <= 20) | (h_ch >= 160)).astype(np.float32)
    color_score = acne_hue * (s_ch / 255.0) * (1.0 - v_ch / 255.0) * (a_channel - 128) / 50.0
    color_score = np.clip(color_score, 0, None)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    mean = cv2.blur(gray, (7, 7)).astype(np.float32)
    sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (7, 7))
    variance = np.clip(sqr_mean - mean ** 2, 0, None)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    acne_map = (0.5 * color_score * 50 + 0.3 * (variance / (np.max(variance) + 1e-6)) * 100 + 0.2 * blackhat.astype(np.float32))
    acne_map[skin_mask == 0] = 0
    if np.any(skin_mask > 0):
        skin_acne = acne_map[skin_mask > 0]
        threshold = np.percentile(skin_acne, 85)
        high_acne_ratio = np.sum(skin_acne > threshold) / len(skin_acne)
        avg_acne = np.mean(skin_acne)
        score = float(np.clip(avg_acne * 2.5 + high_acne_ratio * 30, 0, 100))
    else:
        score = 0.0
    return score, acne_map

def enhanced_pores_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    pore_maps = []
    for ksize in [5, 7, 9, 11]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        pore_maps.append(blackhat.astype(np.float32))
    pores = (0.4 * pore_maps[0] + 0.3 * pore_maps[1] + 0.2 * pore_maps[2] + 0.1 * pore_maps[3])
    mean_local = cv2.blur(gray.astype(np.float32), (9, 9))
    sqr_mean_local = cv2.blur(gray.astype(np.float32) ** 2, (9, 9))
    std_local = np.sqrt(np.clip(sqr_mean_local - mean_local ** 2, 0, None))
    pores = pores + 0.3 * std_local
    pores[skin_mask == 0] = 0
    if np.any(skin_mask > 0):
        skin_pores = pores[skin_mask > 0]
        median = np.median(skin_pores)
        high_pore_values = skin_pores[skin_pores > median]
        if len(high_pore_values) > 0:
            score = float(np.clip(np.mean(high_pore_values) / 4.0, 0, 100))
        else:
            score = 0.0
    else:
        score = 0.0
    return score, pores

def enhanced_dark_circles_map(face_bgr: np.ndarray, masks: Dict[str,np.ndarray]):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[..., 0]
    b_channel = lab[..., 2]
    under_eye = masks.get('under_eyes', np.zeros_like(l_channel, dtype=np.uint8))
    cheeks = masks.get('cheeks', np.zeros_like(l_channel, dtype=np.uint8))
    face = masks.get('face', np.ones_like(l_channel, dtype=np.uint8))
    if not np.any(under_eye > 0) or not np.any(cheeks > 0):
        return 0.0, np.zeros_like(l_channel)
    under_l = l_channel[under_eye > 0]
    cheek_l = l_channel[cheeks > 0]
    mean_under = np.median(under_l)
    mean_cheek = np.median(cheek_l)
    darkness_diff = mean_cheek - l_channel
    darkness_diff = np.clip(darkness_diff, 0, None)
    cheek_b = b_channel[cheeks > 0]
    b_diff = b_channel - np.median(cheek_b)
    dark_circles_map = darkness_diff + 0.3 * np.abs(b_diff)
    dark_circles_map[face == 0] = 0
    brightness_diff = mean_cheek - mean_under
    spread = np.std(under_l)
    score = float(np.clip(brightness_diff * 2.5 + spread * 0.5, 0, 100))
    return score, dark_circles_map

def texture_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap[skin_mask == 0] = 0
    score = float(np.clip(np.mean(lap[skin_mask>0]) / 3.0, 0, 100)) if np.any(skin_mask>0) else 0.0
    return score, lap

def lines_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    edges[skin_mask == 0] = 0
    score = float(np.clip(np.mean(edges[skin_mask>0]) / 2.0, 0, 100)) if np.any(skin_mask>0) else 0.0
    return score, edges

def hydration_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[...,0]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32)
    lap[skin_mask==0] = 0
    mean_l = np.mean(l[skin_mask>0]) if np.any(skin_mask>0) else np.mean(l)
    mean_tex = np.mean(lap[skin_mask>0]) if np.any(skin_mask>0) else np.mean(lap)
    score = float(np.clip((mean_l / 2.0) - (mean_tex / 5.0), 0, 100))
    return score, (255 - lap)

def skin_tone_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = lab[...,2]
    mean_b = np.mean(b[skin_mask>0]) if np.any(skin_mask>0) else np.mean(b)
    score = float(np.interp(mean_b, [95,155], [0,100]))
    tone_map = b.copy()
    tone_map[skin_mask==0] = 0
    return float(np.clip(score,0,100)), tone_map

def uniformness_map(face_bgr: np.ndarray, skin_mask: np.ndarray):
    pig_s, pig_map = enhanced_pigmentation_map(face_bgr, skin_mask)
    tex_s, tex_map = texture_map(face_bgr, skin_mask)
    score = 100.0 - np.clip((pig_s + tex_s) / 2.0, 0, 100)
    return float(score), pig_map

# -------------------------
# PRODUCT RECOMMENDATIONS DATABASE
# -------------------------
SKINCARE_RECOMMENDATIONS = {
    'Redness': {
        'high': {
            'key_ingredients': ['Centella Asiatica (Cica)', 'Niacinamide 5-10%', 'Azelaic Acid 10%', 'Allantoin', 'Bisabolol'],
            'avoid': ['Alcohol denat', 'Fragrance', 'Essential oils', 'Physical scrubs'],
            'products': {
                'Cleanser': ['La Roche-Posay Toleriane Hydrating Gentle Cleanser', 'Cetaphil Gentle Skin Cleanser'],
                'Treatment': ['The Ordinary Azelaic Acid Suspension 10%', 'Paula\'s Choice 10% Azelaic Acid Booster'],
                'Moisturizer': ['La Roche-Posay Cicaplast Baume B5', 'Etude House Soon Jung 2x Barrier Intensive Cream'],
                'Sunscreen': ['EltaMD UV Clear SPF 46', 'La Roche-Posay Anthelios Mineral SPF 50']
            }
        },
        'moderate': {
            'key_ingredients': ['Niacinamide 2-5%', 'Green Tea Extract', 'Licorice Root', 'Ceramides', 'Panthenol'],
            'avoid': ['Hot water', 'Harsh surfactants', 'Strong acids'],
            'products': {
                'Cleanser': ['CeraVe Hydrating Cleanser', 'Neutrogena Ultra Gentle Hydrating Cleanser'],
                'Serum': ['The Ordinary Niacinamide 10% + Zinc 1%', 'Good Molecules Niacinamide Serum'],
                'Moisturizer': ['CeraVe PM Facial Moisturizing Lotion'],
                'Sunscreen': ['Neutrogena Hydro Boost SPF 50']
            }
        },
        'low': {
            'key_ingredients': ['Niacinamide', 'Centella Asiatica', 'Hyaluronic Acid'],
            'avoid': ['Over-exfoliation', 'Unnecessary actives'],
            'products': {
                'Serum': ['COSRX Snail Mucin 96 Power Essence', 'Purito Centella Green Level Buffet Serum'],
                'Moisturizer': ['Neutrogena Hydro Boost Water Gel']
            }
        }
    },
    'Pigmentation': {
        'high': {
            'key_ingredients': ['Vitamin C 15-20%', 'Tranexamic Acid 2-5%', 'Alpha Arbutin 2%', 'Kojic Acid', 'Retinoids'],
            'avoid': ['Sun exposure', 'Picking at skin', 'Harsh scrubs'],
            'products': {
                'Vitamin C': ['Timeless 20% Vitamin C + E Ferulic Acid Serum', 'Mad Hippie Vitamin C Serum'],
                'Treatment': ['The Ordinary Alpha Arbutin 2% + HA', 'Naturium Tranexamic Acid 5%'],
                'Night Cream': ['The Ordinary Granactive Retinoid 2% Emulsion', 'CeraVe Resurfacing Retinol Serum'],
                'Sunscreen': ['EltaMD UV Clear SPF 46', 'Beauty of Joseon Relief Sun SPF 50+']
            }
        },
        'moderate': {
            'key_ingredients': ['Vitamin C 10-15%', 'Niacinamide', 'Licorice Extract', 'Tranexamic Acid'],
            'avoid': ['Sun without SPF', 'Inconsistent routine'],
            'products': {
                'Serum': ['The Ordinary Ascorbyl Glucoside 12%', 'Good Molecules Discoloration Correcting Serum'],
                'Treatment': ['The Inkey List Tranexamic Acid Serum'],
                'Exfoliant': ['The Ordinary Lactic Acid 10% + HA'],
                'Sunscreen': ['La Roche-Posay Anthelios SPF 60']
            }
        },
        'low': {
            'key_ingredients': ['Vitamin C', 'Niacinamide', 'Licorice Root', 'SPF'],
            'avoid': ['Skipping sunscreen', 'Inconsistent use'],
            'products': {
                'Serum': ['CeraVe Skin Renewing Vitamin C Serum'],
                'Sunscreen': ['CeraVe AM Facial Moisturizing Lotion SPF 30']
            }
        }
    },
    'Acne': {
        'high': {
            'key_ingredients': ['Benzoyl Peroxide 2.5-5%', 'Salicylic Acid 2%', 'Niacinamide', 'Adapalene 0.1%'],
            'avoid': ['Heavy oils', 'Comedogenic ingredients', 'Picking'],
            'products': {
                'Cleanser': ['CeraVe Acne Foaming Cream Cleanser', 'La Roche-Posay Effaclar Gel Cleanser'],
                'Treatment': ['Differin Gel (Adapalene 0.1%)', 'Paula\'s Choice 2% BHA Liquid Exfoliant'],
                'Moisturizer': ['CeraVe PM Facial Moisturizing Lotion', 'Cetaphil Pro Oil Absorbing Moisturizer'],
                'Sunscreen': ['EltaMD UV Clear SPF 46', 'La Roche-Posay Anthelios Clear Skin SPF 60']
            }
        },
        'moderate': {
            'key_ingredients': ['Salicylic Acid 0.5-2%', 'Niacinamide 5%', 'Benzoyl Peroxide 2.5%'],
            'avoid': ['Over-cleansing', 'Touching face', 'Thick creams'],
            'products': {
                'Cleanser': ['CeraVe Renewing SA Cleanser', 'Neutrogena Oil-Free Acne Wash'],
                'Treatment': ['The Ordinary Salicylic Acid 2% Solution'],
                'Serum': ['The Inkey List Niacinamide'],
                'Moisturizer': ['Neutrogena Hydro Boost Water Gel (Oil-Free)']
            }
        },
        'low': {
            'key_ingredients': ['Salicylic Acid', 'Niacinamide', 'Hyaluronic Acid'],
            'avoid': ['Over-treating', 'Harsh products'],
            'products': {
                'Cleanser': ['CeraVe Foaming Facial Cleanser'],
                'Treatment': ['The Ordinary Salicylic Acid 2% Masque'],
                'Moisturizer': ['CeraVe PM Lotion']
            }
        }
    },
    'Pores': {
        'high': {
            'key_ingredients': ['Salicylic Acid (BHA) 2%', 'Niacinamide 5-10%', 'Retinol', 'Clay'],
            'avoid': ['Skipping cleansing', 'Heavy silicones', 'Not removing makeup'],
            'products': {
                'Cleanser': ['Paula\'s Choice CLEAR Pore Normalizing Cleanser', 'CeraVe SA Cleanser'],
                'Exfoliant': ['Paula\'s Choice 2% BHA Liquid Exfoliant', 'The Ordinary Salicylic Acid 2%'],
                'Serum': ['The Ordinary Niacinamide 10% + Zinc 1%'],
                'Mask': ['Aztec Secret Indian Healing Clay', 'Origins Clear Improvement Charcoal Mask'],
                'Retinol': ['The Ordinary Retinol 0.5% in Squalane']
            }
        },
        'moderate': {
            'key_ingredients': ['Niacinamide', 'BHA 1-2%', 'AHA', 'Clay'],
            'avoid': ['Pore strips', 'Skipping double cleanse'],
            'products': {
                'Cleanser': ['Neutrogena Oil-Free Acne Wash'],
                'Treatment': ['The Inkey List Niacinamide', 'CosRX BHA Blackhead Power Liquid'],
                'Mask': ['Innisfree Super Volcanic Pore Clay Mask']
            }
        },
        'low': {
            'key_ingredients': ['Niacinamide', 'Hyaluronic Acid', 'Light BHA'],
            'avoid': ['Heavy makeup without removal'],
            'products': {
                'Serum': ['The Ordinary Niacinamide 10% + Zinc 1%'],
                'Cleanser': ['CeraVe Foaming Cleanser']
            }
        }
    },
    'Dark Circles': {
        'high': {
            'key_ingredients': ['Caffeine 5%', 'Vitamin K', 'Retinol', 'Peptides', 'Vitamin C'],
            'avoid': ['Rubbing eyes', 'Salty foods', 'Inadequate sleep'],
            'products': {
                'Eye Cream': ['The Ordinary Caffeine Solution 5% + EGCG', 'CeraVe Eye Repair Cream'],
                'Serum': ['The Inkey List Caffeine Eye Cream', 'Good Molecules Yerba Mate Wake Up Eye Gel'],
                'Treatment': ['RoC Retinol Correxion Eye Cream', 'Olay Eyes Ultimate Eye Cream']
            }
        },
        'moderate': {
            'key_ingredients': ['Caffeine', 'Peptides', 'Vitamin C', 'Hyaluronic Acid'],
            'avoid': ['Screen time before bed', 'Dehydration'],
            'products': {
                'Eye Cream': ['The Ordinary Caffeine Solution 5% + EGCG', 'Neutrogena Rapid Wrinkle Repair Eye Cream'],
                'Serum': ['The Inkey List Caffeine']
            }
        },
        'low': {
            'key_ingredients': ['Caffeine', 'Hyaluronic Acid', 'Peptides'],
            'avoid': ['Neglecting eye area'],
            'products': {
                'Eye Cream': ['CeraVe Eye Repair Cream', 'Neutrogena Hydro Boost Eye Gel']
            }
        }
    },
    'Texture': {
        'high': {
            'key_ingredients': ['Retinol/Retinoids', 'AHA (Glycolic/Lactic) 5-10%', 'BHA 2%', 'Niacinamide'],
            'avoid': ['Over-exfoliation', 'Physical scrubs', 'Skipping moisturizer'],
            'products': {
                'Exfoliant': ['Paula\'s Choice 8% AHA Gel', 'The Ordinary Lactic Acid 10% + HA'],
                'Retinol': ['The Ordinary Retinol 1% in Squalane', 'CeraVe Resurfacing Retinol Serum'],
                'Serum': ['The Ordinary Niacinamide 10% + Zinc 1%'],
                'Moisturizer': ['CeraVe Moisturizing Cream']
            }
        },
        'moderate': {
            'key_ingredients': ['AHA 5-8%', 'BHA', 'Niacinamide', 'Hyaluronic Acid'],
            'avoid': ['Daily harsh exfoliation'],
            'products': {
                'Exfoliant': ['The Ordinary Lactic Acid 5% + HA', 'Paula\'s Choice 2% BHA Liquid'],
                'Serum': ['The Ordinary Niacinamide'],
                'Moisturizer': ['Neutrogena Hydro Boost']
            }
        },
        'low': {
            'key_ingredients': ['Gentle AHA', 'Hyaluronic Acid', 'Ceramides'],
            'avoid': ['Harsh scrubs'],
            'products': {
                'Exfoliant': ['CosRX AHA/BHA Clarifying Treatment Toner'],
                'Moisturizer': ['CeraVe Moisturizing Cream']
            }
        }
    },
    'Lines': {
        'high': {
            'key_ingredients': ['Retinol/Tretinoin', 'Peptides', 'Vitamin C', 'Hyaluronic Acid', 'Niacinamide'],
            'avoid': ['Sun exposure without SPF', 'Smoking', 'Dehydration'],
            'products': {
                'Retinol': ['The Ordinary Retinol 1% in Squalane', 'CeraVe Skin Renewing Retinol Serum'],
                'Peptide Serum': ['The Ordinary Buffet', 'The Inkey List Peptide Moisturizer'],
                'Vitamin C': ['Timeless 20% Vitamin C + E + Ferulic Acid'],
                'Eye Cream': ['RoC Retinol Correxion Eye Cream'],
                'Sunscreen': ['EltaMD UV Clear SPF 46']
            }
        },
        'moderate': {
            'key_ingredients': ['Retinol 0.25-0.5%', 'Peptides', 'Niacinamide'],
            'avoid': ['Inconsistent SPF', 'Dehydration'],
            'products': {
                'Retinol': ['The Ordinary Retinol 0.5% in Squalane'],
                'Serum': ['The Ordinary Buffet'],
                'Moisturizer': ['Olay Regenerist Micro-Sculpting Cream']
            }
        },
        'low': {
            'key_ingredients': ['Retinol (preventive)', 'Peptides', 'SPF'],
            'avoid': ['Skipping SPF'],
            'products': {
                'Retinol': ['CeraVe Resurfacing Retinol Serum'],
                'Sunscreen': ['Any broad-spectrum SPF 30+']
            }
        }
    },
    'Hydration': {
        'high': {
            'key_ingredients': ['Hyaluronic Acid', 'Glycerin', 'Ceramides', 'Squalane', 'Panthenol'],
            'avoid': ['Hot water', 'Harsh cleansers', 'Over-exfoliation'],
            'products': {
                'Cleanser': ['CeraVe Hydrating Cleanser', 'La Roche-Posay Toleriane Hydrating Gentle Cleanser'],
                'Serum': ['The Ordinary Hyaluronic Acid 2% + B5', 'Neutrogena Hydro Boost Hydrating Serum'],
                'Moisturizer': ['CeraVe Moisturizing Cream', 'Neutrogena Hydro Boost Gel-Cream'],
                'Facial Oil': ['The Ordinary 100% Plant-Derived Squalane']
            }
        },
        'moderate': {
            'key_ingredients': ['Hyaluronic Acid', 'Glycerin', 'Ceramides'],
            'avoid': ['Hot showers', 'Skipping moisturizer'],
            'products': {
                'Serum': ['The Ordinary Hyaluronic Acid 2% + B5'],
                'Moisturizer': ['Neutrogena Hydro Boost Water Gel']
            }
        },
        'low': {
            'key_ingredients': ['Hyaluronic Acid', 'Glycerin'],
            'avoid': ['Over-drying products'],
            'products': {
                'Moisturizer': ['Neutrogena Hydro Boost', 'CeraVe PM Lotion']
            }
        }
    },
    'Uniformness': {
        'high': {
            'key_ingredients': ['Vitamin C', 'Niacinamide', 'AHA', 'Retinol', 'Tranexamic Acid'],
            'avoid': ['Sun without SPF 50+', 'Inconsistent routine'],
            'products': {
                'Morning': ['Vitamin C serum', 'Niacinamide', 'SPF 50+'],
                'Evening': ['AHA or Retinol', 'Azelaic Acid', 'Moisturizer']
            }
        },
        'moderate': {
            'key_ingredients': ['Vitamin C', 'Niacinamide', 'Gentle AHA'],
            'avoid': ['Skipping SPF'],
            'products': {
                'Serum': ['Vitamin C + Niacinamide products'],
                'Sunscreen': ['Broad-spectrum SPF 50+']
            }
        },
        'low': {
            'key_ingredients': ['Vitamin C', 'Niacinamide', 'SPF'],
            'avoid': ['Sun damage'],
            'products': {
                'Serum': ['Any Vitamin C or Niacinamide serum'],
                'Sunscreen': ['Daily SPF 30+']
            }
        }
    },
    'Skin Tone': {
        'high': {
            'key_ingredients': ['SPF', 'Vitamin C', 'Niacinamide'],
            'avoid': ['Sun damage'],
            'products': {
                'Serum': ['Vitamin C serum'],
                'Sunscreen': ['Broad-spectrum SPF 50+']
            }
        },
        'moderate': {
            'key_ingredients': ['SPF', 'Vitamin C'],
            'avoid': ['Unprotected sun'],
            'products': {
                'Sunscreen': ['SPF 30-50 daily']
            }
        },
        'low': {
            'key_ingredients': ['SPF'],
            'avoid': ['Unnecessary treatments'],
            'products': {
                'Sunscreen': ['Any SPF 30+']
            }
        }
    }
}

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config("Skin Scanner", "🔬", layout="wide")
st.title("🔬 AI Skin Scanner")
st.markdown("**Advanced skin analysis** • Upload a clear frontal selfie")

# Sidebar
with st.sidebar:
    st.header("📸 Photo Tips")
    st.markdown("""
    **For best results:**
    - Use natural daylight
    - Face camera directly
    - Remove makeup if possible
    - Ensure face is clearly visible
    - Avoid shadows on face
    """)

input_choice = st.radio("Input method:", ["📁 Upload", "📷 Camera"], horizontal=True)
img_bgr = None

if input_choice == "📁 Upload":
    f = st.file_uploader("Upload selfie (jpg/png)", type=["jpg","jpeg","png"])
    if f is not None:
        img_bgr = read_image(f.read())
else:
    c = st.camera_input("Take a selfie")
    if c is not None:
        img_bgr = read_image(c.read())

if img_bgr is not None:
    quality_ok, quality_msg = check_image_quality(img_bgr)
    if not quality_ok:
        st.error(f"⚠️ {quality_msg}")
        st.stop()
    
    with st.spinner("🔄 Analyzing your skin..."):
        original_bgr = img_bgr.copy()
        
        # Get landmarks
        landmarks, (H,W) = get_landmarks(img_bgr)
        if not landmarks:
            st.error("❌ No face detected. Please try a clearer frontal image.")
            st.stop()

        # Crop to face
        hull = cv2.convexHull(np.array(landmarks, np.int32))
        x,y,w,h = cv2.boundingRect(hull)
        margin = int(0.08 * max(w,h))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        
        face_bgr_display = original_bgr[y0:y1, x0:x1].copy()
        face_bgr = img_bgr[y0:y1, x0:x1].copy()
        lm_local = [(px - x0, py - y0) for px,py in landmarks]

        # Build masks
        masks = build_region_masks(lm_local, face_bgr.shape[:2])
        skin_mask = enhanced_skin_mask(face_bgr, masks['face'])
        face_skin_mask = cv2.bitwise_and(masks['face'], skin_mask)

        # Compute all metrics
        scores = {}
        s_red, _ = enhanced_redness_map(face_bgr, face_skin_mask)
        scores['Redness'] = s_red
        s_pig, _ = enhanced_pigmentation_map(face_bgr, face_skin_mask)
        scores['Pigmentation'] = s_pig
        s_acne, _ = enhanced_acne_map(face_bgr, face_skin_mask)
        scores['Acne'] = s_acne
        s_pores, _ = enhanced_pores_map(face_bgr, face_skin_mask)
        scores['Pores'] = s_pores
        s_dark, _ = enhanced_dark_circles_map(face_bgr, masks)
        scores['Dark Circles'] = s_dark
        s_skin, _ = skin_tone_map(face_bgr, face_skin_mask)
        scores['Skin Tone'] = s_skin
        s_tex, _ = texture_map(face_bgr, face_skin_mask)
        scores['Texture'] = s_tex
        s_lines, _ = lines_map(face_bgr, face_skin_mask)
        scores['Lines'] = s_lines
        s_hyd, _ = hydration_map(face_bgr, face_skin_mask)
        scores['Hydration'] = s_hyd
        s_unif, _ = uniformness_map(face_bgr, face_skin_mask)
        scores['Uniformness'] = s_unif

    # Top 2 concerns
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2 = sorted_scores[:2]

    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Analysis Result")
        
        # Draw landmarks on top concern
        face_with_dots = draw_landmarks_on_face(face_bgr_display, lm_local, top2[0][0])
        st.image(to_rgb(face_with_dots), use_container_width=True)

    with col2:
        st.subheader("🎯 Primary Concern")
        st.markdown(f"### {top2[0][0]}")
        st.caption(f"Score: {int(round(top2[0][1]))}/100")
        
        st.subheader("🎯 Secondary Concern")
        st.markdown(f"### {top2[1][0]}")
        st.caption(f"Score: {int(round(top2[1][1]))}/100")

    # All scores in clickable format - SORTED FROM LOW TO HIGH
    st.markdown("---")
    st.subheader("📊 All Concerns (Sorted by severity - Low to High)")
    
    # Sort scores from LOW to HIGH
    sorted_all_scores = sorted(scores.items(), key=lambda x: x[1])
    
    # Create columns for scores
    cols = st.columns(5)
    for i, (concern, score) in enumerate(sorted_all_scores):
        with cols[i % 5]:
            # Color code based on severity
            if score > 60:
                color = "#e74c3c"  # Red
            elif score > 30:
                color = "#f39c12"  # Orange
            else:
                color = "#27ae60"  # Green
            
            st.markdown(f"**{concern}**")
            st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold; color: {color};'>{int(round(score))}</div>", unsafe_allow_html=True)
            st.caption("out of 100")
    
    # Select concern to view highlight
    st.markdown("---")
    st.subheader("🔍 View Concern Details")
    selected_concern = st.selectbox("Select a concern to view highlighted area:", list(scores.keys()))
    
    if selected_concern:
        # Create side-by-side comparison
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Original Image**")
            st.image(to_rgb(face_bgr_display), use_container_width=True)
        
        with col_b:
            st.markdown(f"**{selected_concern} Highlighted**")
            face_highlighted = draw_landmarks_on_face(face_bgr_display, lm_local, selected_concern)
            st.image(to_rgb(face_highlighted), use_container_width=True)
            
            # Show severity
            concern_score = scores[selected_concern]
            if concern_score > 60:
                severity_text = "🔴 High Severity"
            elif concern_score > 30:
                severity_text = "🟡 Moderate Severity"
            else:
                severity_text = "🟢 Low Severity"
            
            st.markdown(f"### Score: {int(round(concern_score))}/100")
            st.markdown(f"### {severity_text}")

    # Recommendations ONLY for top 2 concerns
    st.markdown("---")
    st.subheader("💡 Personalized Skincare Recommendations")
    st.info("✨ Recommendations are provided only for your top 2 concerns")
    
    for idx, (concern, score) in enumerate(top2, 1):
        # Determine severity
        if score > 60:
            severity_level = 'high'
            severity_text = "High"
            severity_emoji = "🔴"
        elif score > 30:
            severity_level = 'moderate'
            severity_text = "Moderate"
            severity_emoji = "🟡"
        else:
            severity_level = 'low'
            severity_text = "Low"
            severity_emoji = "🟢"
        
        if concern in SKINCARE_RECOMMENDATIONS:
            concern_data = SKINCARE_RECOMMENDATIONS[concern][severity_level]
            
            with st.expander(f"**{idx}. {severity_emoji} {concern}** (Severity: {int(score)}/100 - {severity_text})", expanded=(idx==1)):
                # Key Ingredients
                st.markdown(f"### 🧪 Key Ingredients")
                for ingredient in concern_data['key_ingredients']:
                    st.markdown(f"• **{ingredient}**")
                
                # Ingredients to Avoid
                st.markdown(f"### ⛔ Avoid These")
                for avoid_item in concern_data['avoid']:
                    st.markdown(f"• {avoid_item}")
                
                # Product Recommendations
                st.markdown(f"### 🛍️ Recommended Products")
                for product_type, product_list in concern_data['products'].items():
                    st.markdown(f"**{product_type}:**")
                    for product in product_list:
                        st.markdown(f"• {product}")
                    st.write("")
                
                st.markdown("---")
