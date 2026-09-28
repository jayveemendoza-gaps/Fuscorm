"""
Banana Corm Fusarium Wilt Analysis Tool
Developed by the Plant Pathology Laboratory, Institute of Plant Breeding, UPLB
Co-funded by the Gates Foundation

This application analyzes banana corm images to detect and quantify fusarium wilt browning.
"""

# ==================== IMPORTS ====================
import cv2
import hashlib
import io
import json
import numpy as np
import streamlit as st
from colorspacious import cspace_convert
from datetime import datetime
from PIL import Image
from rembg import remove
from streamlit_drawable_canvas import st_canvas

# ==================== CONFIGURATION ====================
MAX_IMAGE_DIM = 1200  # Maximum image dimension for cloud performance

# ==================== UTILITY FUNCTIONS ====================

def cache_resource(func):
    """Wrapper for Streamlit caching with version compatibility"""
    if hasattr(st, "cache_resource"):
        return st.cache_resource(show_spinner=False)(func)
    return st.cache_data(show_spinner=False)(func)

def rerun_app():
    """Rerun Streamlit app with version compatibility"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("Streamlit rerun not available")

# ==================== BACKGROUND REMOVAL ====================

@cache_resource
def remove_background_and_filter_colors(image):
    """
    Remove background using AI and apply blue background for visualization.
    Falls back to color-based approach if AI removal fails.
    """
    img_array = np.array(image)
    
    try:
        # Prepare image for AI background removal
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Remove background using AI
        no_bg = remove(img_bytes.getvalue())
        no_bg_image = Image.open(io.BytesIO(no_bg)).convert('RGBA')
        no_bg_array = np.array(no_bg_image)
        
        # Create blue background for visibility
        blue_bg = np.array([0, 0, 255], dtype=np.uint8)
        background = np.full(img_array.shape, blue_bg)
        
        # Blend using alpha channel
        alpha = no_bg_array[:, :, 3:4] / 255.0
        blended_array = (img_array * alpha + background * (1 - alpha)).astype(np.uint8)
        
        return blended_array
        
    except Exception as e:
        st.warning(f"AI background removal failed: {e}. Using color-based approach.")
        # Fallback to color-based method
        refined_mask = create_hybrid_mask(img_array)
        no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
        filtered_image = filter_corm_colors(no_bg_array[:, :, :3], refined_mask)
        return filtered_image

def create_hybrid_mask(img_array):
    """Create mask to include corm tissue while excluding pot/soil"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    include_mask = (
        (hsv[:, :, 2] > 40) &  # Not too dark
        ~((hsv[:, :, 2] < 25) & (hsv[:, :, 1] < 30)) &  # Not pure black
        ~((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] < 50))  # Not dark brown pot
    )
    return include_mask

def refine_corm_mask(img_array, initial_mask):
    """Refine mask to better distinguish corm tissue from pot/soil"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    not_too_dark = hsv[:, :, 2] > 5
    not_pot = ~((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 20) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] < 40))
    refined = initial_mask & not_too_dark & not_pot
    
    return refined

def create_simple_mask(rgb_image):
    """Create simple foreground/background mask based on brightness"""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Background: very dark or very bright unsaturated areas
    dark_mask = hsv[:, :, 2] < 30
    very_bright_mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 30)
    background_mask = dark_mask | very_bright_mask
    foreground_mask = ~background_mask
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    return foreground_mask.astype(bool)

@cache_resource
def filter_corm_colors(rgb_image, mask):
    """
    Filter to keep only corm tissue colors while excluding pot/soil.
    Preserves: white, yellow, green, brown tissue and lesions.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Corm tissue color ranges (HSV)
    corm_ranges = {
        'white_tissue': [(0, 0, 140), (180, 45, 255)],
        'yellow_tissue': [(15, 30, 100), (45, 200, 255)],
        'light_green': [(40, 20, 80), (80, 120, 220)],
        'healthy_green': [(35, 40, 60), (75, 180, 200)],
        'light_brown': [(8, 25, 80), (25, 150, 200)],
        'medium_brown': [(5, 40, 50), (20, 180, 160)],
        'dark_necrotic': [(0, 0, 10), (30, 255, 85)],
        'beige_tan': [(12, 15, 90), (30, 80, 220)]
    }
    
    # Create combined color mask
    corm_mask = np.zeros(hsv.shape[:2], dtype=bool)
    for color_name, (lower, upper) in corm_ranges.items():
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        corm_mask |= (color_mask > 0)
    
    # Apply filters to exclude pot/soil
    brightness_filter = hsv[:, :, 2] > 5
    saturation_filter = hsv[:, :, 1] < 220
    not_black = ~((hsv[:, :, 2] < 5) & (hsv[:, :, 1] < 50))
    exclude_dark_brown = ~cv2.inRange(hsv, np.array([5, 80, 10]), np.array([25, 255, 40]))
    exclude_pure_black = ~cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 5]))
    
    # Combine all filters
    final_mask = (corm_mask & brightness_filter & saturation_filter & not_black & 
                  exclude_dark_brown.astype(bool) & exclude_pure_black.astype(bool) & mask)
    
    # Morphological cleanup
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_large = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_large)
    final_mask = final_mask.astype(bool)
    
    # Apply mask to image
    result = np.zeros_like(rgb_image)
    result[final_mask] = rgb_image[final_mask]
    return result


# ==================== CANVAS & SELECTION ====================

def _apply_zoom_viewport_to_mask(canvas_mask, original_height, original_width, zoom_viewport):
    """Convert a canvas-space boolean mask to original image space, accounting for zoom viewport."""
    if zoom_viewport:
        crop_w = zoom_viewport["crop_w"]
        crop_h = zoom_viewport["crop_h"]
        crop_x = zoom_viewport["crop_x"]
        crop_y = zoom_viewport["crop_y"]
        crop_mask = cv2.resize(
            canvas_mask.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        full_mask = np.zeros((original_height, original_width), dtype=bool)
        paste_h = min(crop_h, original_height - crop_y)
        paste_w = min(crop_w, original_width - crop_x)
        full_mask[crop_y:crop_y + paste_h, crop_x:crop_x + paste_w] = crop_mask[:paste_h, :paste_w]
        return full_mask
    else:
        if canvas_mask.shape != (original_height, original_width):
            return cv2.resize(
                canvas_mask.astype(np.uint8), (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        return canvas_mask


@st.fragment
def _zoom_pan_canvas_fragment(frag_key, full_img_pil, orig_w, orig_h, drawing_mode, shape_type, polygon_input_mode):
    """Fragment: zoom/pan controls + drawing canvas."""
    from PIL import ImageDraw as _IDraw
    _pan_step = 10
    zoom_viewport = None

    with st.expander("🔍 Zoom & Pan (for precise placement on large images)", expanded=False):
        zoom_level = st.slider(
            "Zoom level", 1.0, 4.0, 1.0, 0.5,
            key=f"{frag_key}_zoom",
            help="Zoom in for more precise selection."
        )
        if zoom_level > 1.0:
            st.caption(f"📍 Showing 1/{zoom_level:.0f}× of image area.")
            nav_crop_w = max(50, int(orig_w / zoom_level))
            nav_crop_h = max(50, int(orig_h / zoom_level))
            nav_max_cx = max(0, orig_w - nav_crop_w)
            nav_max_cy = max(0, orig_h - nav_crop_h)
            st.session_state.setdefault(f"{frag_key}_pan_x", 50)
            st.session_state.setdefault(f"{frag_key}_pan_y", 50)
            nav_pan_x = st.session_state[f"{frag_key}_pan_x"]
            nav_pan_y = st.session_state[f"{frag_key}_pan_y"]
            nav_cx = int(nav_max_cx * nav_pan_x / 100)
            nav_cy = int(nav_max_cy * nav_pan_y / 100)

            mm_max = 220
            mm_scale = mm_max / max(orig_w, orig_h)
            mm_w = max(80, int(orig_w * mm_scale))
            mm_h = max(80, int(orig_h * mm_scale))
            minimap_img = full_img_pil.resize((mm_w, mm_h), Image.Resampling.LANCZOS).copy()
            mm_draw = _IDraw.Draw(minimap_img)
            vp_l = int(nav_cx * mm_scale)
            vp_t = int(nav_cy * mm_scale)
            vp_r = int((nav_cx + nav_crop_w) * mm_scale)
            vp_b = int((nav_cy + nav_crop_h) * mm_scale)
            mm_draw.rectangle([vp_l, vp_t, vp_r, vp_b], outline=(255, 50, 50), width=2)

            nav_col1, nav_col2 = st.columns([1, 1])
            with nav_col1:
                st.caption("🗺️ Minimap (red = current view)")
                st.image(minimap_img, use_container_width=False)
            with nav_col2:
                st.caption("Pan with arrows:")
                _, _up_col, _ = st.columns([1, 1, 1])
                with _up_col:
                    if st.button("⬆️", key=f"{frag_key}_pan_up"):
                        st.session_state[f"{frag_key}_pan_y"] = max(0, nav_pan_y - _pan_step)
                _left_col, _, _right_col = st.columns([1, 1, 1])
                with _left_col:
                    if st.button("⬅️", key=f"{frag_key}_pan_left"):
                        st.session_state[f"{frag_key}_pan_x"] = max(0, nav_pan_x - _pan_step)
                with _right_col:
                    if st.button("➡️", key=f"{frag_key}_pan_right"):
                        st.session_state[f"{frag_key}_pan_x"] = min(100, nav_pan_x + _pan_step)
                _, _down_col, _ = st.columns([1, 1, 1])
                with _down_col:
                    if st.button("⬇️", key=f"{frag_key}_pan_down"):
                        st.session_state[f"{frag_key}_pan_y"] = min(100, nav_pan_y + _pan_step)
                st.caption(f"Position: {nav_pan_x}% → / {nav_pan_y}% ↓")

            pan_x_pct = st.session_state[f"{frag_key}_pan_x"]
            pan_y_pct = st.session_state[f"{frag_key}_pan_y"]
        else:
            pan_x_pct, pan_y_pct = 50, 50

    # Apply zoom viewport
    canvas_image = full_img_pil.copy()
    if zoom_level > 1.0:
        crop_w = max(50, int(orig_w / zoom_level))
        crop_h = max(50, int(orig_h / zoom_level))
        max_crop_x = max(0, orig_w - crop_w)
        max_crop_y = max(0, orig_h - crop_h)
        crop_x = int(max_crop_x * pan_x_pct / 100)
        crop_y = int(max_crop_y * pan_y_pct / 100)
        canvas_image = canvas_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        zoom_viewport = {"crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h}
        img_width, img_height = crop_w, crop_h
    else:
        img_width, img_height = orig_w, orig_h

    # Scale cropped/full image to canvas display size
    max_canvas_size = 500 if zoom_level > 1.0 else 350
    if max(img_width, img_height) > max_canvas_size:
        scale_factor = max_canvas_size / max(img_width, img_height)
        canvas_width = int(img_width * scale_factor)
        canvas_height = int(img_height * scale_factor)
        canvas_image = canvas_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    else:
        canvas_width, canvas_height = img_width, img_height
        scale_factor = 1.0

    min_size = 200
    if canvas_width < min_size or canvas_height < min_size:
        scale = min_size / min(canvas_width, canvas_height)
        canvas_width = int(canvas_width * scale)
        canvas_height = int(canvas_height * scale)
        canvas_image = canvas_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        scale_factor = canvas_width / img_width

    st.caption("💡 Draw your selection on the canvas, then click **Confirm Selection** below.")

    # Canvas key encodes viewport
    if zoom_viewport:
        _ck = f"{frag_key}_z{int(zoom_level*2)}_x{zoom_viewport['crop_x']}_y{zoom_viewport['crop_y']}"
    else:
        _ck = f"{frag_key}_z1"

    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=canvas_image,
            update_streamlit=False,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            key=_ck,
            display_toolbar=False,
        )
    except Exception as canvas_error:
        st.error(f"Canvas initialization failed: {canvas_error}")
        canvas_result = None

    # Confirm Selection
    confirm_key = f"{frag_key}_confirmed"
    if st.button("✅ Confirm Selection", key=f"{frag_key}_confirm_btn"):
        st.session_state[confirm_key] = True
        st.session_state[f"_cf_{frag_key}_result"] = canvas_result
        st.session_state[f"_cf_{frag_key}_sf"] = scale_factor
        st.session_state[f"_cf_{frag_key}_vp"] = zoom_viewport
        st.rerun(scope="app")

    # Persist latest canvas state
    if canvas_result is not None:
        st.session_state[f"_cf_{frag_key}_result"] = canvas_result
        st.session_state[f"_cf_{frag_key}_sf"] = scale_factor
        st.session_state[f"_cf_{frag_key}_vp"] = zoom_viewport


@st.fragment
def _zoom_pan_scale_fragment(canvas_background_pil, original_width, original_height):
    """Fragment: zoom/pan controls + scale calibration canvas."""
    from PIL import ImageDraw as _IDraw
    _pan_step = 10
    scale_zoom_viewport = None
    sc_counter = st.session_state.get('scale_canvas_clear_counter', 0)

    with st.expander("🔍 Zoom & Pan (to precisely place the scale line)", expanded=False):
        scale_zoom = st.slider(
            "Zoom level", 1.0, 4.0, 1.0, 0.5,
            key="scale_canvas_zoom",
            help="Zoom in to draw a more precise scale line."
        )
        if scale_zoom > 1.0:
            st.caption(f"📍 Showing 1/{scale_zoom:.0f}× of image.")
            sc_nav_crop_w = max(50, int(original_width / scale_zoom))
            sc_nav_crop_h = max(50, int(original_height / scale_zoom))
            sc_nav_max_cx = max(0, original_width - sc_nav_crop_w)
            sc_nav_max_cy = max(0, original_height - sc_nav_crop_h)
            st.session_state.setdefault("scale_canvas_pan_x", 50)
            st.session_state.setdefault("scale_canvas_pan_y", 50)
            sc_nav_px = st.session_state["scale_canvas_pan_x"]
            sc_nav_py = st.session_state["scale_canvas_pan_y"]
            sc_nav_cx = int(sc_nav_max_cx * sc_nav_px / 100)
            sc_nav_cy = int(sc_nav_max_cy * sc_nav_py / 100)

            sc_mm_max = 220
            sc_mm_scale = sc_mm_max / max(original_width, original_height)
            sc_mm_w = max(80, int(original_width * sc_mm_scale))
            sc_mm_h = max(80, int(original_height * sc_mm_scale))
            sc_minimap_img = canvas_background_pil.resize((sc_mm_w, sc_mm_h), Image.Resampling.LANCZOS).copy()
            sc_mm_draw = _IDraw.Draw(sc_minimap_img)
            sc_vp_l = int(sc_nav_cx * sc_mm_scale)
            sc_vp_t = int(sc_nav_cy * sc_mm_scale)
            sc_vp_r = int((sc_nav_cx + sc_nav_crop_w) * sc_mm_scale)
            sc_vp_b = int((sc_nav_cy + sc_nav_crop_h) * sc_mm_scale)
            sc_mm_draw.rectangle([sc_vp_l, sc_vp_t, sc_vp_r, sc_vp_b], outline=(255, 50, 50), width=2)

            sc_nav_col1, sc_nav_col2 = st.columns([1, 1])
            with sc_nav_col1:
                st.caption("🗺️ Minimap (red = current view)")
                st.image(sc_minimap_img, use_container_width=False)
            with sc_nav_col2:
                st.caption("Pan with arrows:")
                _, _sc_up_col, _ = st.columns([1, 1, 1])
                with _sc_up_col:
                    if st.button("⬆️", key="scale_canvas_pan_up"):
                        st.session_state["scale_canvas_pan_y"] = max(0, sc_nav_py - _pan_step)
                _sc_left_col, _, _sc_right_col = st.columns([1, 1, 1])
                with _sc_left_col:
                    if st.button("⬅️", key="scale_canvas_pan_left"):
                        st.session_state["scale_canvas_pan_x"] = max(0, sc_nav_px - _pan_step)
                with _sc_right_col:
                    if st.button("➡️", key="scale_canvas_pan_right"):
                        st.session_state["scale_canvas_pan_x"] = min(100, sc_nav_px + _pan_step)
                _, _sc_down_col, _ = st.columns([1, 1, 1])
                with _sc_down_col:
                    if st.button("⬇️", key="scale_canvas_pan_down"):
                        st.session_state["scale_canvas_pan_y"] = min(100, sc_nav_py + _pan_step)
                st.caption(f"Position: {sc_nav_px}% → / {sc_nav_py}% ↓")

            sc_pan_x = st.session_state["scale_canvas_pan_x"]
            sc_pan_y = st.session_state["scale_canvas_pan_y"]
        else:
            sc_pan_x, sc_pan_y = 50, 50

    # Apply zoom viewport
    scale_bg = canvas_background_pil.copy()
    if scale_zoom > 1.0:
        sc_crop_w = max(50, int(original_width / scale_zoom))
        sc_crop_h = max(50, int(original_height / scale_zoom))
        sc_max_cx = max(0, original_width - sc_crop_w)
        sc_max_cy = max(0, original_height - sc_crop_h)
        sc_crop_x = int(sc_max_cx * sc_pan_x / 100)
        sc_crop_y = int(sc_max_cy * sc_pan_y / 100)
        scale_bg = scale_bg.crop((sc_crop_x, sc_crop_y, sc_crop_x + sc_crop_w, sc_crop_y + sc_crop_h))
        scale_zoom_viewport = {"crop_x": sc_crop_x, "crop_y": sc_crop_y,
                               "crop_w": sc_crop_w, "crop_h": sc_crop_h}
        view_w, view_h = sc_crop_w, sc_crop_h
    else:
        view_w, view_h = original_width, original_height

    max_canvas_height = 450 if scale_zoom > 1.0 else 400
    max_canvas_width = 650 if scale_zoom > 1.0 else 600
    sc_h_ratio = max_canvas_height / view_h
    sc_w_ratio = max_canvas_width / view_w
    sc_sf = min(sc_h_ratio, sc_w_ratio, 1.0)
    if scale_zoom > 1.0:
        sc_sf = min(sc_h_ratio, sc_w_ratio)
    canvas_height = int(view_h * sc_sf)
    canvas_width = int(view_w * sc_sf)
    canvas_background_resized = scale_bg.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    # Canvas key encodes viewport
    if scale_zoom_viewport:
        _svp = scale_zoom_viewport
        _sc_ck = f"scale_canvas_{sc_counter}_z{int(scale_zoom*2)}_x{_svp['crop_x']}_y{_svp['crop_y']}"
    else:
        _sc_ck = f"scale_canvas_{sc_counter}_z1"

    st.caption("💡 Draw a line on the ruler, then click **Calculate Scale** below.")
    scale_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=canvas_background_resized,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="line",
        key=_sc_ck,
        display_toolbar=False,
    )
    
    if scale_canvas and hasattr(scale_canvas, 'json_data') and scale_canvas.json_data:
        objects = scale_canvas.json_data.get("objects", [])
        if objects:
            obj = objects[-1]
            if obj and obj.get("type") == "line":
                try:
                    x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                    x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                    px_len = np.sqrt((x2-x1)**2 + (y2-y1)**2) / sc_sf
                    if px_len > 0:
                        st.session_state['scale_line_px'] = px_len
                        st.caption(f"📐 Line: {px_len:.1f} image px")
                except Exception:
                    pass
    
    st.session_state['_cf_scale_canvas'] = scale_canvas
    st.session_state['_cf_scale_sf'] = sc_sf
    st.session_state['_cf_scale_vp'] = scale_zoom_viewport

    if st.button("📐 Calculate Scale", key="scale_calc_btn"):
        st.rerun(scope="app")


def extract_shape_mask(canvas_result, scale_factor, shape_type, original_shape, zoom_viewport=None):
    """Extract mask from shape drawn on canvas."""
    original_height, original_width = original_shape[:2]

    def _offset(val_x, val_y):
        if zoom_viewport:
            return val_x + zoom_viewport["crop_x"], val_y + zoom_viewport["crop_y"]
        return val_x, val_y

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        shape_obj = canvas_result.json_data["objects"][-1]
        mask = np.zeros((original_height, original_width), dtype=bool)
        
        if shape_type == "Rectangle" and shape_obj["type"] == "rect":
            left = int(shape_obj["left"] / scale_factor)
            top = int(shape_obj["top"] / scale_factor)
            width = int(shape_obj["width"] / scale_factor)
            height = int(shape_obj["height"] / scale_factor)
            left, top = _offset(left, top)
            
            left = max(0, min(left, original_width - 1))
            top = max(0, min(top, original_height - 1))
            right = max(left + 1, min(left + width, original_width))
            bottom = max(top + 1, min(top + height, original_height))
            
            mask[top:bottom, left:right] = True
            st.info(f"✅ Rectangle: ({left}, {top}) to ({right}, {bottom})")
            return mask
        
        elif shape_type == "Polygon":
            if canvas_result.image_data is not None:
                alpha_data = canvas_result.image_data[:, :, 3]
                if np.any(alpha_data > 0):
                    drawn_mask = alpha_data > 0
                    drawn_mask = _apply_zoom_viewport_to_mask(drawn_mask, original_height, original_width, zoom_viewport)
                    area = np.sum(drawn_mask)
                    if area > 10:
                        st.info(f"✅ Polygon - Area: {area} pixels")
                        return drawn_mask
            
            st.warning("⚠️ Polygon not detected.")
            return None
    
    if canvas_result.image_data is not None:
        alpha_data = canvas_result.image_data[:, :, 3]
        if np.any(alpha_data > 0):
            line_mask = alpha_data > 0
            line_mask = _apply_zoom_viewport_to_mask(line_mask, original_height, original_width, zoom_viewport)
            area = np.sum(line_mask)
            st.info(f"✅ Selection - Area: {area} pixels")
            return line_mask
    
    return None


def analyze_shape_region(image, shape_mask, ignore_blue=True):
    """Analyze browning in the selected shape region."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        st.error(f"Invalid image format. Expected RGB image with shape (H, W, 3).")
        return None
    
    if shape_mask is None:
        st.warning("No shape selected.")
        return None
    
    if ignore_blue:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        blue_lower = np.array([100, 150, 200])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
        non_blue_mask = blue_mask == 0
    else:
        non_blue_mask = np.ones(image.shape[:2], dtype=bool)
    
    analysis_mask = shape_mask & non_blue_mask
    
    if not np.any(analysis_mask):
        st.warning("No valid corm pixels found.")
        return None
    
    selected_pixels = image[analysis_mask]
    
    if selected_pixels.ndim == 1:
        selected_pixels = selected_pixels.reshape(-1, 3)
    elif selected_pixels.shape[-1] != 3:
        st.error(f"Invalid pixel data shape: {selected_pixels.shape}.")
        return None
    
    # Calculate browning
    percent_browning, browning_pixels, total_corm_pixels, browning_breakdown = calculate_percent_browning(selected_pixels)
    
    # Convert to LAB
    lab_pixels = rgb_to_lab(selected_pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]
    
    mm_per_px = st.session_state.get('mm_per_px', None)
    scale_info = {
        'mm_per_px': mm_per_px,
        'has_scale': mm_per_px is not None
    }
    
    if mm_per_px:
        total_area_mm2 = total_corm_pixels * (mm_per_px ** 2)
        browning_area_mm2 = browning_pixels * (mm_per_px ** 2)
        scaled_breakdown = {}
        for key, value in browning_breakdown.items():
            if isinstance(value, (int, np.integer)):
                scaled_breakdown[key + '_area_mm2'] = value * (mm_per_px ** 2)
            scaled_breakdown[key] = value
        if 'fusarium' in browning_breakdown:
            scaled_breakdown['fusarium_area_mm2'] = browning_breakdown['fusarium'] * (mm_per_px ** 2)
    else:
        total_area_mm2 = total_corm_pixels
        browning_area_mm2 = browning_pixels
        scaled_breakdown = browning_breakdown

    return {
        'analysis_mask': analysis_mask,
        'percent_browning': percent_browning,
        'browning_pixels': browning_pixels,
        'total_corm_pixels': total_corm_pixels,
        'browning_breakdown': scaled_breakdown,
        'scale_info': scale_info,
        'total_area_mm2': total_area_mm2,
        'browning_area_mm2': browning_area_mm2
    }


def calculate_percent_browning(rgb_pixels):
    """Calculate browning percentage based on multiple lesion types."""
    if rgb_pixels.ndim == 1:
        rgb_pixels = rgb_pixels.reshape(-1, 3)
    elif rgb_pixels.shape[-1] != 3:
        raise ValueError(f"Invalid pixel data shape: {rgb_pixels.shape}.")
    
    hsv_pixels = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    
    # Define lesion color ranges
    fusarium_red_mask = cv2.inRange(hsv_pixels, np.array([0, 60, 20]), np.array([10, 255, 120]))
    fusarium_purple_mask = cv2.inRange(hsv_pixels, np.array([140, 50, 20]), np.array([180, 255, 120]))
    fusarium_mask = fusarium_red_mask | fusarium_purple_mask
    
    dark_necrotic_mask = cv2.inRange(hsv_pixels, np.array([0, 0, 0]), np.array([180, 255, 80]))
    dark_brown_mask = cv2.inRange(hsv_pixels, np.array([0, 40, 30]), np.array([35, 255, 140]))
    normal_brown_mask = cv2.inRange(hsv_pixels, np.array([8, 50, 50]), np.array([30, 255, 180]))
    yellowish_brown_mask = cv2.inRange(hsv_pixels, np.array([25, 80, 60]), np.array([35, 200, 160]))
    
    # Create exclusion masks
    H = hsv_pixels[:, :, 0]
    S = hsv_pixels[:, :, 1]
    V = hsv_pixels[:, :, 2]
    
    white_mask = (S < 30) & (V > 230)
    very_light_mask = (S < 20) & (V > 200)
    shadow_mask = (V < 15) & (S < 20)
    healthy_corm_mask = (S < 35) & (V > 170) & (H < 35)
    
    # Apply exclusions
    fusarium_mask = fusarium_mask & ~white_mask.astype(np.uint8) * 255
    dark_necrotic_mask = dark_necrotic_mask & ~very_light_mask.astype(np.uint8) * 255
    dark_brown_mask = dark_brown_mask & ~very_light_mask.astype(np.uint8) * 255
    normal_brown_mask = normal_brown_mask & ~very_light_mask.astype(np.uint8) * 255
    yellowish_brown_mask = yellowish_brown_mask & ~(white_mask | very_light_mask | healthy_corm_mask | shadow_mask).astype(np.uint8) * 255
    
    # Combine all lesion types
    combined_browning_mask = (fusarium_mask > 0) | (dark_necrotic_mask > 0) | (dark_brown_mask > 0) | (normal_brown_mask > 0) | (yellowish_brown_mask > 0)
    
    # Calculate counts
    fusarium_pixels = np.sum(fusarium_mask > 0)
    dark_necrotic_pixels = np.sum(dark_necrotic_mask > 0)
    dark_brown_pixels = np.sum(dark_brown_mask > 0)
    normal_brown_pixels = np.sum(normal_brown_mask > 0)
    yellowish_brown_pixels = np.sum(yellowish_brown_mask > 0)
    total_browning_pixels = np.sum(combined_browning_mask)
    
    total_pixels = len(rgb_pixels)
    percent_browning = (total_browning_pixels / total_pixels) * 100
    
    browning_breakdown = {
        'fusarium': fusarium_pixels,
        'dark_necrotic': dark_necrotic_pixels,
        'dark_brown': dark_brown_pixels,
        'normal_brown': normal_brown_pixels,
        'yellowish_brown': yellowish_brown_pixels,
        'total_browning': total_browning_pixels,
    }
    
    return percent_browning, total_browning_pixels, total_pixels, browning_breakdown


def rgb_to_lab(rgb_array):
    """Convert RGB to CIELAB color space."""
    rgb_normalized = rgb_array / 255.0
    lab = cspace_convert(rgb_normalized, "sRGB1", "CIELab")
    return lab


def main():
    st.set_page_config(
        page_title="Banana Corm Browning Analyzer",
        page_icon="🍌",
        layout="wide"
    )
    
    st.title("🍌 Banana Corm Browning Analyzer")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='margin: 0; color: #1f77b4;'>📋 Workflow: Upload → Select Area → Analyze</h4>
    <p style='margin: 5px 0 0 0; color: #666;'>Accurate browning detection with multiple lesion types</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'mm_per_px' not in st.session_state:
        st.session_state.mm_per_px = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose corm image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of banana corm cross-section"
        )
    
    if uploaded_file is not None:
        try:
            # Load image
            original_image = Image.open(uploaded_file)
            
            if original_image.mode == 'RGBA':
                background = Image.new('RGB', original_image.size, (255, 255, 255))
                background.paste(original_image, mask=original_image.split()[3])
                original_image = background
            elif original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            if max(original_image.size) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(original_image.size)
                new_size = (int(original_image.size[0]*scale), int(original_image.size[1]*scale))
                original_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                st.info(f"Image resized to {new_size}.")

            # Step 1: Scale Calibration
            st.markdown("---")
            st.subheader("📏 Step 1: Scale Calibration")
            
            image_np = np.array(original_image)
            canvas_background = Image.fromarray(image_np.astype('uint8'))
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'scale_canvas_clear_counter' not in st.session_state:
                    st.session_state.scale_canvas_clear_counter = 0
                original_height, original_width = image_np.shape[:2]
                _zoom_pan_scale_fragment(canvas_background, original_width, original_height)

            with col2:
                st.markdown("**Instructions:**")
                st.markdown("1. Draw a line over a known measurement")
                st.markdown("2. Enter the real-world length")
                st.markdown("3. Scale will be calculated automatically")
                
                if st.button("🧹 Clear Scale Line", type="secondary"):
                    st.session_state.scale_canvas_clear_counter += 1
                    st.session_state.pop('scale_line_px', None)
                    st.rerun()
                
                scale_length_mm = st.number_input(
                    "Real length of drawn line (mm):",
                    min_value=0.1,
                    value=10.0,
                    step=1.0
                )
                
                scale_px = st.session_state.get('scale_line_px')
                if scale_px:
                    st.info(f"📐 Line length: {scale_px:.1f} px")
                
                if scale_px and scale_length_mm > 0:
                    st.session_state.mm_per_px = scale_length_mm / scale_px
                    st.success(f"📏 Scale: {st.session_state.mm_per_px:.4f} mm/pixel")
                elif st.session_state.get('mm_per_px'):
                    st.success(f"📏 Scale active: {st.session_state.mm_per_px:.4f} mm/pixel")
                else:
                    st.info("👆 Draw a line on the ruler/scale")
                
                if st.button("⏭️ Skip Scale"):
                    st.session_state.mm_per_px = None
                    st.warning("Scale skipped. Measurements in pixels only.")

            # Step 2: Prepare Image
            st.markdown("---")
            st.subheader("🔧 Step 2: Prepare Image")

            if st.button("📋 Prepare Image", type="primary"):
                img_array = np.array(original_image)
                
                if img_array.ndim == 3 and img_array.shape[2] == 4:
                    rgb = img_array[:, :, :3]
                    alpha = img_array[:, :, 3:4] / 255.0
                    white_bg = np.ones_like(rgb) * 255
                    img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                
                st.session_state.processed_image = img_array
                st.success("✅ Image prepared!")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original_image, caption="📷 Original", use_container_width=True)
            with col2:
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, caption="🎨 Processed", use_container_width=True)

            # Step 3: Selection
            if st.session_state.processed_image is not None:
                st.markdown("---")
                st.subheader("🎯 Step 3: Select Analysis Area")
                
                if st.session_state.get('mm_per_px'):
                    st.success(f"📏 Scale: {st.session_state.mm_per_px:.3f} mm/pixel")
                else:
                    st.info("📏 No scale — measurements in pixels")

                canvas_result, scale_factor, shape_type, zoom_viewport = create_selection_canvas(
                    st.session_state.processed_image,
                    canvas_key="corm_selection"
                )
                
                if canvas_result and (
                    (canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) > 0) or
                    (canvas_result.image_data and np.any(canvas_result.image_data[:, :, 3] > 0))
                ):
                    st.markdown("---")
                    selection_mask = extract_shape_mask(
                        canvas_result,
                        scale_factor,
                        shape_type,
                        st.session_state.processed_image.shape,
                        zoom_viewport=zoom_viewport
                    )
                    
                    if selection_mask is not None and np.any(selection_mask):
                        if st.button("🔬 Analyze Area", type="primary"):
                            with st.spinner("🔬 Analyzing..."):
                                analysis_results = analyze_shape_region(
                                    st.session_state.processed_image,
                                    selection_mask,
                                    ignore_blue=True
                                )
                            if analysis_results:
                                st.success("✅ Analysis complete!")
                                st.session_state.analysis_results = analysis_results
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        results = st.session_state.analysis_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📐 Total Area", f"{results['total_corm_pixels']:,} px")
        with col2:
            st.metric("🔴 Lesion Area", f"{results['browning_pixels']:,} px")
        with col3:
            st.metric("📊 Browning %", f"{results['percent_browning']:.1f}%")
        
        with st.expander("📋 Detailed Breakdown"):
            breakdown = results['browning_breakdown']
            st.write(f"Total Browning Pixels: {breakdown.get('total_browning', 0):,}")
            st.write(f"Fusarium: {breakdown.get('fusarium', 0):,}")
            st.write(f"Dark Necrotic: {breakdown.get('dark_necrotic', 0):,}")
            st.write(f"Dark Brown: {breakdown.get('dark_brown', 0):,}")
            st.write(f"Normal Brown: {breakdown.get('normal_brown', 0):,}")
            st.write(f"Yellowish Brown: {breakdown.get('yellowish_brown', 0):,}")


def create_selection_canvas(image, canvas_key="canvas"):
    """Create interactive canvas for selecting corm area."""
    try:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] in [3, 4]:
                canvas_image = Image.fromarray(image.astype('uint8'))
            else:
                st.error(f"Unexpected image shape: {image.shape}")
                return None, None, None, None
        else:
            canvas_image = image

        if canvas_image is None:
            st.error("Canvas image is None")
            return None, None, None, None

        orig_img_width, orig_img_height = canvas_image.size

        # Selection tools UI
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            shape_type = st.selectbox(
                "🎯 Selection Shape:",
                ["Polygon", "Freeform"],
                help="Choose shape type for area selection"
            )
            
            polygon_input_mode = None
            if shape_type == "Polygon":
                st.info("💡 Click near your first point to auto-close the polygon")
                radio_key = f"{canvas_key}_polygon_input_mode"
                if radio_key not in st.session_state:
                    st.session_state[radio_key] = "Point mode (click to add points)"
                polygon_input_mode = st.radio(
                    "Polygon input method:",
                    ["Point mode (click to add points)", "Freehand (drag to draw)"],
                    index=0,
                    key=radio_key
                )
        
        with col2:
            canvas_mode = st.selectbox(
                "⚙️ Mode:",
                ["Draw New", "Edit/Transform"],
                help="Draw new shapes or edit existing ones"
            )
        
        with col3:
            if st.button("🗑️ Clear", help="Clear all shapes"):
                if 'canvas_clear_counter' not in st.session_state:
                    st.session_state.canvas_clear_counter = 0
                st.session_state.canvas_clear_counter += 1
                st.session_state.pop(f"_cf_{canvas_key}_result", None)
                canvas_key = f"{canvas_key}_{st.session_state.canvas_clear_counter}"

        # Compute drawing mode
        mode_map = {
            ("Polygon", "Draw New"): "polygon",
            ("Freeform", "Draw New"): "freedraw",
            ("Polygon", "Edit/Transform"): "transform",
            ("Freeform", "Edit/Transform"): "transform",
        }
        drawing_mode = mode_map.get((shape_type, canvas_mode), "polygon")
        if shape_type == "Polygon" and canvas_mode == "Draw New" and polygon_input_mode:
            if "Freehand" in polygon_input_mode:
                drawing_mode = "freedraw"

        # Fragment for zoom/pan + canvas
        _zoom_pan_canvas_fragment(
            canvas_key, canvas_image, orig_img_width, orig_img_height,
            drawing_mode, shape_type, polygon_input_mode
        )

        # Read results from fragment
        canvas_result = st.session_state.get(f"_cf_{canvas_key}_result")
        scale_factor = st.session_state.get(f"_cf_{canvas_key}_sf", 1.0)
        zoom_viewport = st.session_state.get(f"_cf_{canvas_key}_vp")

        return canvas_result, scale_factor, shape_type, zoom_viewport

    except Exception as e:
        st.error(f"Canvas error: {e}")
        return None, None, None, None


if __name__ == "__main__":
    main()
