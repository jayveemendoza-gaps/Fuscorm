import streamlit as st
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from colorspacious import cspace_convert
from streamlit_drawable_canvas import st_canvas
from rembg import remove
import io
import json
import hashlib

# --- Streamlit Cloud Optimizations ---
MAX_IMAGE_DIM = 1200  # Limit max image dimension for upload

# Use Streamlit caching for heavy deterministic functions
def cache_resource(func):
    # Use st.cache_resource if available, else fallback to st.cache_data
    if hasattr(st, "cache_resource"):
        return st.cache_resource(show_spinner=False)(func)
    return st.cache_data(show_spinner=False)(func)

def rerun_app():
    """Compatible rerun function for different Streamlit versions"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # Fallback for very old versions
        raise RuntimeError("Streamlit rerun not available")

@cache_resource
def remove_background_and_filter_colors(image):
    """
    Remove background and keep only corm colors
    Balanced approach - removes pot/soil while preserving corm tissue
    Cloud-optimized with timeout and fallbacks
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    try:
        # Convert to bytes for rembg with timeout protection
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Remove background using AI with cloud-friendly error handling
        try:
            no_bg = remove(img_bytes.getvalue())
            no_bg_image = Image.open(io.BytesIO(no_bg)).convert('RGBA')
            no_bg_array = np.array(no_bg_image)
        except Exception as e:
            st.warning(f"AI background removal failed ({str(e)}). Using color-based approach...")
            # Fallback to color-based approach
            alpha_mask = create_hybrid_mask(img_array)
            no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
            no_bg_array[:, :, 3] = (alpha_mask * 255).astype(np.uint8)
        
        # Create mask from alpha channel with more permissive threshold
        alpha_mask = no_bg_array[:, :, 3] > 30  # Lower threshold to keep more tissue
        
        # If too much was removed (less than 5% remains), use color-based approach
        if np.sum(alpha_mask) < (alpha_mask.size * 0.05):
            st.warning("AI background removal was too aggressive. Using hybrid approach...")
            # Create a more permissive mask based on color analysis
            alpha_mask = create_hybrid_mask(img_array)
            no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
        
        # Additional refinement: exclude obvious non-corm areas
        refined_mask = refine_corm_mask(img_array, alpha_mask)
        
    except Exception as e:
        st.warning(f"AI background removal failed: {e}. Using color-based approach.")
        # Fallback to color-based segmentation
        refined_mask = create_hybrid_mask(img_array)
        no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
    
    # Apply the refined color filtering
    filtered_image = filter_corm_colors(no_bg_array[:, :, :3], refined_mask)
    
    return filtered_image

def create_hybrid_mask(img_array):
    """
    Create a mask that includes corm tissue while excluding pot/soil
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Include areas that are likely corm tissue
    include_mask = (
        # Not too dark
        (hsv[:, :, 2] > 40) &
        # Not pure black
        ~((hsv[:, :, 2] < 25) & (hsv[:, :, 1] < 30)) &
        # Not extremely dark brown (pot material)
        ~((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25) & 
          (hsv[:, :, 1] > 100) & (hsv[:, :, 2] < 50))
    )
    
    return include_mask

def refine_corm_mask(img_array, initial_mask):
    """
    Refine the mask to better distinguish corm tissue from pot/soil
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Exclude very dark areas that are likely pot/soil
    not_too_dark = hsv[:, :, 2] > 35
    
    # Exclude areas that are clearly pot material (very dark brown)
    not_pot = ~((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 20) & 
                (hsv[:, :, 1] > 120) & (hsv[:, :, 2] < 60))
    
    # Combine with initial mask
    refined = initial_mask & not_too_dark & not_pot
    
    return refined

def create_simple_mask(rgb_image):
    """
    Create a simple mask to separate foreground from background
    Based on the assumption that background is usually darker or very different from corm colors
    """
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Create mask for likely background (very dark or very bright)
    dark_mask = hsv[:, :, 2] < 30  # Very dark pixels
    very_bright_mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 30)  # Very bright, unsaturated
    
    # Background is likely dark or very bright unsaturated areas
    background_mask = dark_mask | very_bright_mask
    
    # Foreground is everything else
    foreground_mask = ~background_mask
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    return foreground_mask.astype(bool)

@cache_resource
def filter_corm_colors(rgb_image, mask):
    """
    Filter image to keep only corm colors: green, white, yellow, brown
    Balanced approach to preserve corm tissue while excluding pot/soil
    """
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Primary corm tissue colors - more inclusive but targeted
    corm_ranges = {
        'white_tissue': [(0, 0, 140), (180, 45, 255)],     # White/cream corm tissue
        'yellow_tissue': [(15, 30, 100), (45, 200, 255)],  # Yellow corm areas
        'light_green': [(40, 20, 80), (80, 120, 220)],     # Light green corm tissue
        'healthy_green': [(35, 40, 60), (75, 180, 200)],   # Healthy green areas
        'light_brown': [(8, 25, 80), (25, 150, 200)],      # Light brown lesions
        'medium_brown': [(5, 40, 50), (20, 180, 160)],     # Medium brown lesions
        'beige_tan': [(12, 15, 90), (30, 80, 220)]         # Beige/tan tissue
    }
    
    # Create mask for corm colors
    corm_mask = np.zeros(hsv.shape[:2], dtype=bool)
    for color_name, (lower, upper) in corm_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        color_mask = cv2.inRange(hsv, lower, upper)
        corm_mask |= (color_mask > 0)
    
    # Exclude very dark areas (pot, soil, deep shadows)
    brightness_threshold = hsv[:, :, 2] > 45  # Exclude very dark pixels
    
    # Exclude extremely saturated colors (not natural tissue)
    saturation_filter = hsv[:, :, 1] < 220  # Allow most natural colors
    
    # Exclude pure black and very dark brown (pot/soil)
    not_black = ~((hsv[:, :, 2] < 35) & (hsv[:, :, 1] < 50))
    
    # Specific exclusions for pot materials
    exclude_dark_brown = ~cv2.inRange(hsv, np.array([5, 80, 10]), np.array([25, 255, 60]))
    exclude_pure_black = ~cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
    
    # Combine all filters
    final_mask = (corm_mask & 
                  brightness_threshold & 
                  saturation_filter & 
                  not_black & 
                  exclude_dark_brown.astype(bool) & 
                  exclude_pure_black.astype(bool) & 
                  mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    final_mask = final_mask.astype(bool)
    
    # Create output image
    result = np.zeros_like(rgb_image)
    result[final_mask] = rgb_image[final_mask]
    
    return result

def create_selection_canvas(image, canvas_key="canvas"):
    """
    Create an interactive canvas for selecting corm area using shapes with editing capabilities
    """
    try:
        # Convert numpy array to PIL Image for canvas
        if isinstance(image, np.ndarray):
            # Ensure image has proper shape and values
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                canvas_image = Image.fromarray(image.astype('uint8'))
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA image
                canvas_image = Image.fromarray(image.astype('uint8'))
            else:
                st.error(f"Unexpected image shape: {image.shape}")
                return None, None, None
        else:
            canvas_image = image
        
        # Validate canvas_image
        if canvas_image is None:
            st.error("Canvas image is None")
            return None, None, None
        
        # Get image dimensions
        img_width, img_height = canvas_image.size
        
        # Scale down for canvas if too large - more aggressive for cloud stability
        max_canvas_size = 400  # Reduced from 600 for better cloud performance
        if max(img_width, img_height) > max_canvas_size:
            scale_factor = max_canvas_size / max(img_width, img_height)
            canvas_width = int(img_width * scale_factor)
            canvas_height = int(img_height * scale_factor)
            canvas_image = canvas_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        else:
            canvas_width, canvas_height = img_width, img_height
            scale_factor = 1.0
        
        # Ensure minimum canvas size for usability
        min_size = 200
        if canvas_width < min_size or canvas_height < min_size:
            scale = min_size / min(canvas_width, canvas_height)
            canvas_width = int(canvas_width * scale)
            canvas_height = int(canvas_height * scale)
            canvas_image = canvas_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # Streamlined selection tools
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            shape_type = st.selectbox(
                "🎯 Selection Shape:",
                ["Rectangle", "Circle", "Polygon", "Freeform"],
                help="Choose shape type for area selection"
            )
            # If the user chose Polygon, offer a small input-mode chooser so they know
            # we can use either point-click polygon mode or a freehand draw (freedraw).
            polygon_input_mode = None
            if shape_type == "Polygon":
                # Set a default in session state so the radio is always initialized
                radio_key = f"{canvas_key}_polygon_input_mode"
                if radio_key not in st.session_state:
                    st.session_state[radio_key] = "Point mode (click to add points)"
                polygon_input_mode = st.radio(
                    "Polygon input method:",
                    ["Point mode (click to add points)", "Freehand (drag to draw)"],
                    index=0,
                    key=radio_key,
                    help="Point mode: Click multiple points to create polygon vertices, then close the shape. Freehand: Draw continuously by dragging."
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
                canvas_key = f"{canvas_key}_{st.session_state.canvas_clear_counter}"
        
        # Set drawing mode based on selection
        mode_map = {
            ("Rectangle", "Draw New"): "rect",
            ("Circle", "Draw New"): "circle", 
            # Polygon defaults to point-click polygon mode; user can choose Freehand via the radio.
            ("Polygon", "Draw New"): "polygon",
            ("Freeform", "Draw New"): "freedraw",
            ("Rectangle", "Edit/Transform"): "transform",
            ("Circle", "Edit/Transform"): "transform",
            ("Polygon", "Edit/Transform"): "transform",
            ("Freeform", "Edit/Transform"): "transform"
        }
        
        drawing_mode = mode_map.get((shape_type, canvas_mode), "rect")

        # If the user selected Polygon, respect their polygon_input_mode choice
        if shape_type == "Polygon" and canvas_mode == "Draw New":
            if polygon_input_mode is not None:
                # If user chose Point mode, use polygon; if they chose Freehand, use freedraw
                if "Point mode" in polygon_input_mode:
                    drawing_mode = "polygon"
                elif "Freehand" in polygon_input_mode:
                    drawing_mode = "freedraw"
                # else fall back to mode_map default (polygon)
        
        # Simple instructions
        instruction_map = {
            "rect": "📦 Click and drag to draw rectangle",
            "circle": "⭕ Click and drag to draw circle", 
            "polygon": "🔷 Click points to create polygon vertices. Right-click or press ESC to close the polygon",
            "freedraw": "🖊️ Draw any shape by dragging your mouse",
            "transform": "✏️ Click and drag shapes to resize/move"
        }
        
        st.info(instruction_map.get(drawing_mode, "Select area for analysis"))
        # Extra clarification for polygon point mode (users often click and see dots)
        if shape_type == "Polygon" and polygon_input_mode is not None and "Point mode" in polygon_input_mode:
            st.info("🔷 **Polygon Point Mode Instructions:**\n"
                   "1. **Left-click** points around your analysis area (minimum 3 points)\n" 
                   "2. **Right-click** or **press ESC** to close the polygon (avoid double-click as it deletes points)\n"
                   "3. The polygon will auto-close and analysis will start automatically\n"
                   "4. You'll see a red filled area when the polygon is complete")
            st.warning("⚠️ **Important**: Double-clicking will DELETE the last point, not close the polygon. Use right-click or ESC instead!")
        
        # Create canvas with cloud-optimized settings
        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.2)",
                stroke_width=3,  # Thicker stroke for better visibility
                stroke_color="#FF0000",
                background_image=canvas_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key=canvas_key,
                display_toolbar=False,  # Hide toolbar for cleaner interface
            )
        except Exception as canvas_error:
            st.error(f"Canvas initialization failed: {canvas_error}")
            st.warning("Falling back to alternative selection method...")
            return create_alternative_selection(image, canvas_key + "_fallback")
        
        # Check for polygon completion and provide feedback
        if canvas_result is not None and shape_type == "Polygon":
            # Check if we have a completed polygon
            polygon_completed = False
            point_count = 0
            
            if canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) > 0:
                for obj in canvas_result.json_data["objects"]:
                    if obj.get("type") in ["polygon", "path"]:
                        # Check if polygon has enough points and is closed
                        if "points" in obj:
                            point_count = len(obj["points"])
                            polygon_completed = point_count >= 3
                        elif "path" in obj and obj["path"]:
                            # Path-based polygon is usually completed
                            polygon_completed = True
                            try:
                                point_count = len(parse_svg_path(obj["path"], scale_factor))
                            except Exception:
                                point_count = 0
            
            # Provide real-time feedback
            if point_count > 0:
                if polygon_completed:
                    st.success(f"✅ Polygon completed with {point_count} points! Ready for analysis.")
                    # Add a session state flag for auto-analysis
                    if f"{canvas_key}_polygon_completed" not in st.session_state:
                        st.session_state[f"{canvas_key}_polygon_completed"] = True
                        rerun_app()  # Trigger rerun to show analysis button
                else:
                    st.info(f"🔷 Polygon in progress: {point_count} points. Need at least 3 points. Right-click or press ESC to complete.")
        
        return canvas_result, scale_factor, shape_type
        
    except Exception as e:
        st.error(f"Canvas error: {e}")
        return create_alternative_selection(image, canvas_key + "_fallback")

def create_alternative_selection(image, canvas_key="alt_canvas"):
    """
    Alternative selection method using coordinate inputs
    """
    st.subheader("Alternative Selection Method")
    st.info("Draw a rectangular region by specifying coordinates")
    
    # Get image dimensions
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        display_image = Image.fromarray(image.astype('uint8'))
    else:
        width, height = image.size
        display_image = image
    
    # Display image with coordinates
    st.image(display_image, caption=f"Image size: {width} x {height} pixels", width=400)
    
    # Input coordinates
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("X1 (left)", min_value=0, max_value=width-1, value=0, key=f"{canvas_key}_x1")
        y1 = st.number_input("Y1 (top)", min_value=0, max_value=height-1, value=0, key=f"{canvas_key}_y1")
    
    with col2:
        x2 = st.number_input("X2 (right)", min_value=0, max_value=width-1, value=width//2, key=f"{canvas_key}_x2")
        y2 = st.number_input("Y2 (bottom)", min_value=0, max_value=height-1, value=height//2, key=f"{canvas_key}_y2")
    
    # Create a fake canvas result with rectangular selection
    class AlternativeCanvasResult:
        def __init__(self, x1, y1, x2, y2, width, height):
            # Create a simple rectangular mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[int(y1):int(y2), int(x1):int(x2)] = 255
            
            # Create image_data in the format expected by the rest of the code
            self.image_data = np.zeros((height, width, 4), dtype=np.uint8)
            self.image_data[:, :, 3] = mask  # Alpha channel
            
        @property
        def json_data(self):
            return None
    
    # Ensure coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    canvas_result = AlternativeCanvasResult(x1, y1, x2, y2, width, height)
    
    # Show preview of selection
    if x2 > x1 and y2 > y1:
        st.success(f"Selected region: ({x1}, {y1}) to ({x2}, {y2}) - Area: {(x2-x1) * (y2-y1)} pixels")
        
        # Create preview image with selection overlay
        preview_img = np.array(display_image)
        if len(preview_img.shape) == 3:
            # Add red overlay to selected area
            preview_img[int(y1):int(y2), int(x1):int(x2), 0] = np.minimum(
                preview_img[int(y1):int(y2), int(x1):int(x2), 0] + 50, 255
            )
        
        st.image(preview_img, caption="Selection Preview (red overlay)", width=400)
    
    return canvas_result, 1.0, "Rectangle"  # No scaling for alternative method, always rectangle

def extract_shape_mask(canvas_result, scale_factor, shape_type, original_shape):
    """
    Extract mask from shape drawn on canvas - supports Rectangle, Circle, and Polygon
    """
    # First check if we have json_data with objects (rectangles and circles)
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        # Get the last drawn object
        shape_obj = canvas_result.json_data["objects"][-1]
        
        # Get original image dimensions
        original_height, original_width = original_shape[:2]
        
        # Create mask based on shape type
        mask = np.zeros((original_height, original_width), dtype=bool)
        
        if shape_type == "Rectangle" and shape_obj["type"] == "rect":
            # Extract rectangle coordinates
            left = int(shape_obj["left"] / scale_factor)
            top = int(shape_obj["top"] / scale_factor)
            width = int(shape_obj["width"] / scale_factor)
            height = int(shape_obj["height"] / scale_factor)
            
            # Ensure coordinates are within bounds
            left = max(0, min(left, original_width - 1))
            top = max(0, min(top, original_height - 1))
            right = max(left + 1, min(left + width, original_width))
            bottom = max(top + 1, min(top + height, original_height))
            
            # Create rectangular mask
            mask[top:bottom, left:right] = True
            
            st.info(f"✅ Rectangle selected: ({left}, {top}) to ({right}, {bottom}) - Area: {(right-left) * (bottom-top)} pixels")
            return mask
            
        elif shape_type == "Circle" and shape_obj["type"] == "circle":
            # Extract circle parameters
            center_x = int(shape_obj["left"] / scale_factor) + int(shape_obj["radius"] / scale_factor)
            center_y = int(shape_obj["top"] / scale_factor) + int(shape_obj["radius"] / scale_factor)
            radius = int(shape_obj["radius"] / scale_factor)
            
            # Ensure center is within bounds
            center_x = max(0, min(center_x, original_width - 1))
            center_y = max(0, min(center_y, original_height - 1))
            
            # Create circular mask
            y, x = np.ogrid[:original_height, :original_width]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask = circle_mask
            
            area = np.pi * radius**2
            st.info(f"✅ Circle selected: Center ({center_x}, {center_y}), Radius {radius} - Area: {area:.0f} pixels")
            return mask
        
        elif shape_type == "Polygon":
            # For polygons, try multiple detection methods
            polygon_detected = False
            
            # Method 1: Alpha channel method (works for both point-mode and freehand polygons)
            if canvas_result.image_data is not None:
                alpha_data = canvas_result.image_data[:, :, 3]
                if np.any(alpha_data > 0):
                    # Create mask from alpha channel
                    drawn_mask = alpha_data > 0
                    
                    # Resize mask to match original dimensions if needed
                    if drawn_mask.shape != (original_height, original_width):
                        drawn_mask = cv2.resize(
                            drawn_mask.astype(np.uint8), 
                            (original_width, original_height), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    area = np.sum(drawn_mask)
                    st.info(f"✅ Polygon selected (alpha method) - Area: {area} pixels")
                    return drawn_mask
            
            # Method 2: JSON parsing as fallback (for point-mode polygons)
            if "type" in shape_obj and shape_obj["type"] in ["polygon", "path"]:
                try:
                    points = []
                    # Handle different polygon formats
                    if "path" in shape_obj and shape_obj["path"]:
                        # Parse path data for complex polygons
                        points = parse_svg_path(shape_obj["path"], scale_factor)
                    elif "points" in shape_obj:
                        # Direct points array
                        points = [(int(p["x"] / scale_factor), int(p["y"] / scale_factor)) for p in shape_obj["points"]]
                    
                    if len(points) >= 3:
                        # Ensure polygon is closed by adding first point at end if needed
                        if points[0] != points[-1]:
                            points.append(points[0])
                        
                        # Create polygon mask using PIL
                        from PIL import Image, ImageDraw
                        mask_img = Image.new('L', (original_width, original_height), 0)
                        draw = ImageDraw.Draw(mask_img)
                        
                        # Ensure points are within bounds
                        bounded_points = []
                        for x, y in points:
                            x = max(0, min(x, original_width - 1))
                            y = max(0, min(y, original_height - 1))
                            bounded_points.append((x, y))
                        
                        # Draw filled polygon
                        draw.polygon(bounded_points, fill=255)
                        
                        # Convert to numpy boolean mask
                        mask = np.array(mask_img) > 0
                        
                        area = np.sum(mask)
                        st.info(f"✅ Polygon selected (JSON method): {len(points)-1} points - Area: {area} pixels")
                        return mask
                        
                except Exception as e:
                    st.warning(f"Could not extract polygon from JSON data: {e}")
            
            # If both methods failed, provide helpful guidance
            st.warning("⚠️ Polygon not detected. For Point mode: Click at least 3 points, then double-click the last point or press ESC to close the polygon. For Freehand mode: Draw a closed shape.")
            return None
        
        elif shape_type == "Freeform":
            # For freeform, use alpha channel method and fill the enclosed area
            if canvas_result.image_data is not None:
                alpha_data = canvas_result.image_data[:, :, 3]
                if np.any(alpha_data > 0):
                    # Create mask from alpha channel (the drawn line)
                    line_mask = alpha_data > 0
                    
                    # Resize mask to match original dimensions if needed
                    if line_mask.shape != (original_height, original_width):
                        line_mask = cv2.resize(
                            line_mask.astype(np.uint8), 
                            (original_width, original_height), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # Fill the enclosed area using flood fill
                    filled_mask = fill_enclosed_area(line_mask)
                    
                    area = np.sum(filled_mask)
                    st.info(f"✅ Freeform area filled - Enclosed area: {area} pixels")
                    return filled_mask
            
            # If no alpha data
            st.warning("No freeform selection detected. Please draw a freeform shape on the image.")
            return None
    
    # If json_data method failed or for freeform polygons, try alpha channel method
    if canvas_result.image_data is not None:
        alpha_data = canvas_result.image_data[:, :, 3]
        if np.any(alpha_data > 0):
            # Get original image dimensions
            original_height, original_width = original_shape[:2]
            
            # Create mask from alpha channel (the drawn line)
            line_mask = alpha_data > 0
            
            # Resize mask to match original dimensions if needed
            if line_mask.shape != (original_height, original_width):
                line_mask = cv2.resize(
                    line_mask.astype(np.uint8), 
                    (original_width, original_height), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # For freeform shapes, fill the enclosed area
            if shape_type == "Freeform":
                filled_mask = fill_enclosed_area(line_mask)
                area = np.sum(filled_mask)
                st.info(f"✅ Freeform area filled - Enclosed area: {area} pixels")
                return filled_mask
            else:
                # For other shapes, use the line mask as-is
                area = np.sum(line_mask)
                st.info(f"✅ Selection detected - Area: {area} pixels")
                return line_mask
    
    # No valid selection found
    return None

def display_analysis_results(analysis_results, context="main"):
    """
    Display analysis results in a clean, organized format
    """
    # Key metrics at the top
    col1, col2, col3 = st.columns(3)
    
    # Get scale info and areas
    scale_info = analysis_results.get('scale_info', {})
    has_scale = scale_info.get('has_scale', False)
    total_area = analysis_results.get('total_area_mm2', analysis_results['total_corm_pixels'])
    lesion_area = analysis_results.get('browning_area_mm2', analysis_results['browning_pixels'])
    
    with col1:
        # Total analyzed area
        if has_scale:
            if total_area >= 100:  # Show in cm² if large
                area_cm2 = total_area / 100
                area_text = f"{area_cm2:.1f} cm²"
            else:
                area_text = f"{total_area:.1f} mm²"
        else:
            area_text = f"{total_area:,} pixels"
        
        st.metric(
            "📐 Total Area", 
            area_text,
            help="Total area analyzed"
        )
    
    with col2:
        # Lesion area
        if has_scale:
            if lesion_area >= 100:  # Show in cm² if large
                lesion_cm2 = lesion_area / 100
                lesion_text = f"{lesion_cm2:.1f} cm²"
            else:
                lesion_text = f"{lesion_area:.1f} mm²"
        else:
            lesion_text = f"{lesion_area:,} pixels"
        
        st.metric(
            "� Lesion Area", 
            lesion_text,
            help="Total area of detected lesions"
        )
    
    with col3:
        st.metric(
            "📊 Browning %", 
            f"{analysis_results['percent_browning']:.1f}%",
            help="Percentage of area showing browning/lesions"
        )
    

    
    # Detailed breakdown
    with st.expander("📋 Detailed Breakdown", expanded=True):
        breakdown = analysis_results['browning_breakdown']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Lesion Types:**")
            
            # Create colored indicators for each type with pixel counts
            dark_count = np.sum(breakdown['dark_brown_mask'])
            normal_count = np.sum(breakdown['normal_brown_mask'])
            yellow_count = np.sum(breakdown['yellowish_brown_mask'])
            
            # Use pre-calculated scaled values from analysis results
            scale_info = analysis_results.get('scale_info', {})
            has_scale = scale_info.get('has_scale', False)
            
            # Helper function to format area using pre-calculated values
            def format_area_from_breakdown(area_key, pixel_count):
                if has_scale and area_key in breakdown:
                    area_mm2 = breakdown[area_key]
                    if area_mm2 >= 10:  # Show in cm² if reasonably large
                        area_cm2 = area_mm2 / 100
                        return f"({pixel_count:,} px, {area_cm2:.2f} cm²)"
                    else:
                        return f"({pixel_count:,} px, {area_mm2:.1f} mm²)"
                return f"({pixel_count:,} pixels)"
            
            # Show reddish brown lesions (previously called fusarium)
            reddish_pixels = breakdown.get('fusarium', 0)
            if reddish_pixels > 0:
                reddish_area_text = format_area_from_breakdown('fusarium_area_mm2', reddish_pixels)
                reddish_percent = (reddish_pixels / analysis_results['total_corm_pixels']) * 100 if analysis_results['total_corm_pixels'] > 0 else 0
                st.markdown(f"� **Reddish Brown (Critical):** {reddish_percent:.1f}% {reddish_area_text}")
            
            st.markdown(f"� **Dark Brown (Severe):** {breakdown['dark_brown_percent']:.1f}% {format_area_from_breakdown('dark_brown_area_mm2', dark_count)}")
            st.markdown(f"🟠 **Normal Brown (Moderate):** {breakdown['normal_brown_percent']:.1f}% {format_area_from_breakdown('normal_brown_area_mm2', normal_count)}")
            st.markdown(f"🟡 **Yellowish Brown (Early):** {breakdown['yellowish_brown_percent']:.1f}% {format_area_from_breakdown('yellowish_brown_area_mm2', yellow_count)}")
                
            # Scale information
            if has_scale:
                mm_per_px = scale_info.get('mm_per_px', 0)
                st.markdown(f"📏 **Scale:** {mm_per_px:.3f} mm/pixel")
            else:
                st.markdown("📏 **Scale:** Not calibrated (pixel measurements only)")
        
        with col2:
            # Add legend for visualization
            st.markdown("**Visualization Legend:**")
            st.markdown("🟢 Green: Analysis area")
            st.markdown("🔴 Bright Red: Reddish brown lesions (critical)")
            st.markdown("🟤 Dark Red: Dark brown lesions (severe)")
            st.markdown("🟠 Red-Orange: Normal brown lesions (moderate)")
            st.markdown("🟡 Orange: Yellowish brown lesions (early)")
    
    # Visualization
    st.subheader("📸 Visual Analysis")
    
    # Create overlay visualization
    overlay_image = create_analysis_overlay(analysis_results)
    if overlay_image is not None:
        st.image(overlay_image, caption="Browning Detection Overlay", use_column_width=True)
    
    # Export option
    if st.button("📥 Export Results", key=f"export_results_{context}"):
        with st.spinner("📊 Generating CSV data..."):
            export_data = create_export_data(analysis_results)
        st.download_button(
            label="Download CSV",
            data=export_data,
            file_name=f"browning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_csv_{context}"
        )

def create_analysis_overlay(analysis_results):
    """Create a clean overlay visualization"""
    try:
        # Get the processed image from session state
        if 'processed_image' not in st.session_state:
            return None
            
        base_image = st.session_state.processed_image.copy()
        overlay = np.zeros_like(base_image)
        
        # Green overlay for analyzed area
        overlay[analysis_results['analysis_mask']] = [0, 255, 0]
        
        # Create lesion overlay by re-detecting lesions in the analyzed region
        # This ensures we get all lesion types correctly visualized
        analysis_mask = analysis_results['analysis_mask']
        if np.any(analysis_mask):
            # Get pixels in the analyzed region
            selected_pixels = st.session_state.processed_image[analysis_mask]
            
            # Re-run lesion detection to get proper masks
            hsv_pixels = cv2.cvtColor(selected_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
            
            # Use same detection logic as in calculate_percent_browning
            # 1. Fusarium lesions (reddish/purplish)
            fusarium_red_lower = np.array([0, 60, 20])
            fusarium_red_upper = np.array([10, 255, 120])
            fusarium_purple_lower = np.array([140, 50, 20])
            fusarium_purple_upper = np.array([180, 255, 120])
            
            fusarium_red_mask_1d = cv2.inRange(hsv_pixels, fusarium_red_lower, fusarium_red_upper)
            fusarium_purple_mask_1d = cv2.inRange(hsv_pixels, fusarium_purple_lower, fusarium_purple_upper)
            fusarium_mask_1d = fusarium_red_mask_1d | fusarium_purple_mask_1d
            
            # 2. Dark brown lesions (severe browning)
            dark_brown_lower = np.array([5, 80, 15])
            dark_brown_upper = np.array([20, 255, 110])
            dark_brown_mask_1d = cv2.inRange(hsv_pixels, dark_brown_lower, dark_brown_upper)
            
            # 3. Normal brown lesions (moderate browning)
            normal_brown_lower = np.array([8, 70, 60])
            normal_brown_upper = np.array([25, 255, 170])
            normal_brown_mask_1d = cv2.inRange(hsv_pixels, normal_brown_lower, normal_brown_upper)
            
            # 4. Yellowish brown lesions (early browning)
            yellowish_brown_lower = np.array([20, 50, 80])
            yellowish_brown_upper = np.array([35, 200, 220])
            yellowish_brown_mask_1d = cv2.inRange(hsv_pixels, yellowish_brown_lower, yellowish_brown_upper)
            
            # Apply exclusions (same as in calculate_percent_browning)
            H = hsv_pixels[:, :, 0]
            S = hsv_pixels[:, :, 1] 
            V = hsv_pixels[:, :, 2]
            
            white_mask = (S < 30) & (V > 230)
            very_light_mask = (S < 20) & (V > 200)
            shadow_mask = (V < 15) & (S < 20)
            healthy_corm_mask = (S < 40) & (V > 180) & (H < 30)
            
            # Apply exclusions
            fusarium_exclusion = white_mask | very_light_mask | healthy_corm_mask
            dark_brown_exclusion = white_mask | very_light_mask | healthy_corm_mask
            normal_brown_exclusion = white_mask | very_light_mask | shadow_mask | healthy_corm_mask
            yellowish_brown_exclusion = white_mask | very_light_mask | shadow_mask | healthy_corm_mask
            
            fusarium_mask_1d = fusarium_mask_1d & ~fusarium_exclusion.astype(np.uint8) * 255
            dark_brown_mask_1d = dark_brown_mask_1d & ~dark_brown_exclusion.astype(np.uint8) * 255
            normal_brown_mask_1d = normal_brown_mask_1d & ~normal_brown_exclusion.astype(np.uint8) * 255
            yellowish_brown_mask_1d = yellowish_brown_mask_1d & ~yellowish_brown_exclusion.astype(np.uint8) * 255
            
            # Convert 1D masks back to 2D coordinates
            analysis_coords = np.where(analysis_mask)
            
            # Apply different colors for different lesion types
            if len(analysis_coords[0]) == len(fusarium_mask_1d):
                # Fusarium lesions - Bright red (most severe)
                fusarium_indices = np.where(fusarium_mask_1d > 0)[0]
                if len(fusarium_indices) > 0:
                    fusarium_coords_y = analysis_coords[0][fusarium_indices]
                    fusarium_coords_x = analysis_coords[1][fusarium_indices]
                    overlay[fusarium_coords_y, fusarium_coords_x] = [255, 0, 0]  # Bright red for fusarium
                
                # Dark brown lesions - Dark red
                dark_brown_indices = np.where(dark_brown_mask_1d > 0)[0]
                if len(dark_brown_indices) > 0:
                    dark_coords_y = analysis_coords[0][dark_brown_indices]
                    dark_coords_x = analysis_coords[1][dark_brown_indices]
                    overlay[dark_coords_y, dark_coords_x] = [180, 0, 0]  # Dark red
                
                # Normal brown lesions - Red-orange
                normal_brown_indices = np.where(normal_brown_mask_1d > 0)[0]
                if len(normal_brown_indices) > 0:
                    normal_coords_y = analysis_coords[0][normal_brown_indices]
                    normal_coords_x = analysis_coords[1][normal_brown_indices]
                    overlay[normal_coords_y, normal_coords_x] = [255, 50, 0]  # Red-orange
                
                # Yellowish brown lesions - Orange
                yellow_brown_indices = np.where(yellowish_brown_mask_1d > 0)[0]
                if len(yellow_brown_indices) > 0:
                    yellow_coords_y = analysis_coords[0][yellow_brown_indices]
                    yellow_coords_x = analysis_coords[1][yellow_brown_indices]
                    overlay[yellow_coords_y, yellow_coords_x] = [255, 165, 0]  # Orange
        
        # Blend with original
        alpha = 0.3
        result = cv2.addWeighted(base_image, 1-alpha, overlay, alpha, 0)
        return result
        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def create_export_data(analysis_results):
    """Create CSV export data with browning analysis results and scaled measurements"""
    
    # Create summary data
    breakdown = analysis_results['browning_breakdown']
    scale_info = analysis_results.get('scale_info', {})
    has_scale = scale_info.get('has_scale', False)
    
    data = [
        ["Metric", "Value", "Unit"],
        ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""],
        ["Total Browning Percentage", f"{analysis_results['percent_browning']:.2f}", "%"],
    ]
    
    # Add scale information
    if has_scale:
        mm_per_px = scale_info.get('mm_per_px', 0)
        data.extend([
            ["Scale Calibration", f"{mm_per_px:.4f}", "mm/pixel"],
            ["Total Analyzed Area", f"{analysis_results.get('total_area_mm2', 0):.2f}", "mm²"],
            ["Total Lesion Area", f"{analysis_results.get('browning_area_mm2', 0):.2f}", "mm²"],
        ])
        
        # Add area measurements for each lesion type
        if 'fusarium_area_mm2' in breakdown:
            data.append(["Reddish Brown Area", f"{breakdown['fusarium_area_mm2']:.2f}", "mm²"])
        if 'dark_brown_area_mm2' in breakdown:
            data.append(["Dark Brown Area", f"{breakdown['dark_brown_area_mm2']:.2f}", "mm²"])
        if 'normal_brown_area_mm2' in breakdown:
            data.append(["Normal Brown Area", f"{breakdown['normal_brown_area_mm2']:.2f}", "mm²"])
        if 'yellowish_brown_area_mm2' in breakdown:
            data.append(["Yellowish Brown Area", f"{breakdown['yellowish_brown_area_mm2']:.2f}", "mm²"])
    else:
        data.extend([
            ["Scale Calibration", "Not set", "pixel measurements only"],
            ["Total Analyzed Pixels", analysis_results['total_corm_pixels'], "pixels"],
            ["Total Lesion Pixels", breakdown['total_browning'], "pixels"],
        ])
    
    # Add pixel counts (always included)
    data.extend([
        ["Reddish Brown Pixels", breakdown['fusarium'], "pixels"],
        ["Dark Brown Pixels", breakdown['dark_brown'], "pixels"],
        ["Normal Brown Pixels", breakdown['normal_brown'], "pixels"],
        ["Yellowish Brown Pixels", breakdown['yellowish_brown'], "pixels"],
        ["Light Brown Pixels", breakdown['light_brown'], "pixels"],
    ])
    

    
    # Convert to CSV
    output = io.StringIO()
    for row in data:
        output.write(",".join(map(str, row)) + "\n")
    
    return output.getvalue()

def create_batch_export():
    """Create comprehensive CSV export for all batch results"""
    if not st.session_state.batch_results:
        return "No batch results to export"
    
    output = io.StringIO()
    
    # Write header
    header = [
        "Image_Name", "Analysis_Date", "Total_Browning_%", 
        "Scale_mm_per_pixel", "Total_Area_mm2", "Total_Area_pixels",
        "Lesion_Area_mm2", "Lesion_Area_pixels",
        "Reddish_Brown_mm2", "Reddish_Brown_pixels",
        "Dark_Brown_mm2", "Dark_Brown_pixels", 
        "Normal_Brown_mm2", "Normal_Brown_pixels",
        "Yellowish_Brown_mm2", "Yellowish_Brown_pixels",
        "Light_Brown_mm2", "Light_Brown_pixels"
    ]
    output.write(",".join(header) + "\n")
    
    # Write data for each image
    for result in st.session_state.batch_results:
        analysis = result['analysis_results']
        breakdown = analysis['browning_breakdown']
        scale_info = analysis.get('scale_info', {})
        
        row = [
            result['image_name'],
            result['timestamp'],
            f"{analysis['percent_browning']:.2f}",
            f"{scale_info.get('mm_per_px', 0):.4f}",
            f"{analysis.get('total_area_mm2', 0):.2f}",
            analysis['total_corm_pixels'],
            f"{analysis.get('browning_area_mm2', 0):.2f}",
            breakdown['total_browning'],
            f"{breakdown.get('fusarium_area_mm2', 0):.2f}",
            breakdown['fusarium'],
            f"{breakdown.get('dark_brown_area_mm2', 0):.2f}",
            breakdown['dark_brown'],
            f"{breakdown.get('normal_brown_area_mm2', 0):.2f}",
            breakdown['normal_brown'],
            f"{breakdown.get('yellowish_brown_area_mm2', 0):.2f}",
            breakdown['yellowish_brown'],
            f"{breakdown.get('light_brown_area_mm2', 0):.2f}",
            breakdown['light_brown']
        ]
        output.write(",".join(map(str, row)) + "\n")
    
    return output.getvalue()

def add_to_batch_results(analysis_results, image_name):
    """Add analysis results to batch processing storage with cloud limits"""
    if st.session_state.processing_mode == "Batch Processing":
        MAX_BATCH_SIZE = 20  # Cloud deployment limit
        
        # Check batch size limit
        if len(st.session_state.batch_results) >= MAX_BATCH_SIZE:
            st.warning(f"⚠️ Batch limit reached ({MAX_BATCH_SIZE} images). Export current results and clear to continue.")
            return
        
        # Create lightweight batch entry (remove large arrays to save memory)
        batch_analysis = analysis_results.copy()
        # Remove large image arrays to save memory
        for key in ['original', 'lab', 'L', 'a', 'b']:
            if key in batch_analysis:
                del batch_analysis[key]
        
        batch_entry = {
            'image_name': image_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_results': batch_analysis,
            'batch_index': len(st.session_state.batch_results) + 1
        }
        st.session_state.batch_results.append(batch_entry)
        st.success(f"✅ Added '{image_name}' to batch results ({len(st.session_state.batch_results)}/{MAX_BATCH_SIZE} total)")
        
        # Auto-save batch data after adding new result
        save_batch_data()

def save_batch_data():
    """Save batch results to persistent storage (JSON file)"""
    try:
        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
            # Create a data directory if it doesn't exist
            import os
            data_dir = "batch_data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Save with timestamp for crash recovery
            save_path = os.path.join(data_dir, "batch_session.json")
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'batch_results': st.session_state.batch_results,
                'processing_mode': getattr(st.session_state, 'processing_mode', 'Batch Processing'),
                'version': '1.0'
            }
            
            with open(save_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
                
    except Exception as e:
        # Silent fail - don't interrupt user workflow
        pass

def load_batch_data():
    """Load batch results from persistent storage"""
    try:
        import os
        save_path = os.path.join("batch_data", "batch_session.json")
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                backup_data = json.load(f)
            
            # Check if data is recent (within last 7 days to avoid old stale data)
            from datetime import datetime, timedelta
            save_time = datetime.fromisoformat(backup_data.get('timestamp', ''))
            if datetime.now() - save_time < timedelta(days=7):
                return backup_data.get('batch_results', [])
    except Exception as e:
        # Silent fail - return empty if can't load
        pass
    return []

def clear_saved_batch_data():
    """Clear the saved batch data file"""
    try:
        import os
        save_path = os.path.join("batch_data", "batch_session.json")
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        pass

def delete_batch_result(index):
    """Delete a specific batch result by index"""
    try:
        if 0 <= index < len(st.session_state.batch_results):
            deleted_name = st.session_state.batch_results[index]['image_name']
            del st.session_state.batch_results[index]
            save_batch_data()  # Auto-save after deletion
            return deleted_name
    except Exception:
        pass
    return None

def parse_svg_path(path_string, scale_factor):
    """
    Simple SVG path parser for basic polygon paths
    """
    points = []
    try:
        # Handle both string and list inputs
        if isinstance(path_string, list):
            # If it's already a list, convert to string
            path_string = ' '.join(str(item) for item in path_string)
        elif not isinstance(path_string, str):
            # If it's neither string nor list, convert to string
            path_string = str(path_string)
            
        # Basic parser for simple move-to and line-to commands
        commands = path_string.replace(',', ' ').split()
        i = 0
        while i < len(commands):
            if commands[i] in ['M', 'L']:  # Move to or Line to
                if i + 2 < len(commands):
                    x = float(commands[i + 1]) / scale_factor
                    y = float(commands[i + 2]) / scale_factor
                    points.append((int(x), int(y)))
                    i += 3
                else:
                    break
            else:
                i += 1
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        # If parsing fails, return empty list
        pass
    
    return points

def fill_enclosed_area(line_mask):
    """
    Fill the enclosed area within a freehand drawn shape.
    Uses flood fill from the corners to identify the background, 
    then inverts to get the enclosed area.
    """
    try:
        # Convert boolean mask to uint8
        mask_uint8 = line_mask.astype(np.uint8) * 255
        
        # Create a copy for flood fill (flood fill modifies the image)
        filled = mask_uint8.copy()
        
        # Get image dimensions
        h, w = filled.shape
        
        # Create a slightly larger image to ensure corners are always background
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:h+1, 1:w+1] = filled
        
        # Flood fill from all corners to mark background
        # This marks all areas connected to the edges as background (255)
        cv2.floodFill(padded, None, (0, 0), 255)
        cv2.floodFill(padded, None, (w+1, 0), 255)
        cv2.floodFill(padded, None, (0, h+1), 255)
        cv2.floodFill(padded, None, (w+1, h+1), 255)
        
        # Extract the original size
        filled = padded[1:h+1, 1:w+1]
        
        # The enclosed area is everything that is NOT background (not 255) and not the original line
        # Invert: background becomes 0, enclosed area becomes 255
        enclosed_area = (filled == 0).astype(np.uint8) * 255
        
        # Add the original line back to the enclosed area
        final_mask = np.logical_or(enclosed_area > 0, line_mask)
        
        return final_mask.astype(bool)
        
    except Exception as e:
        # If filling fails, fall back to the original line mask
        st.warning(f"Could not fill enclosed area: {e}. Using drawn line only.")
        return line_mask

def calculate_percent_browning(rgb_pixels):
    """
    Calculate percent browning based on multiple brown lesion types:
    - Dark brown lesions (severe browning)
    - Normal brown lesions (moderate browning) 
    - Light brown lesions (mild browning)
    - Yellowish brown lesions (early browning)
    """
    # Convert to HSV for better color classification
    hsv_pixels = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    
    # Define color ranges for different lesion types in HSV
    # Enhanced to detect fusarium-specific colors (red/purple) and browning
    
    # 1. Fusarium lesions (reddish/purplish) - characteristic of fusarium wilt
    fusarium_red_lower = np.array([0, 60, 20])    # Deep red to red
    fusarium_red_upper = np.array([10, 255, 120])
    fusarium_purple_lower = np.array([140, 50, 20])  # Purple to magenta  
    fusarium_purple_upper = np.array([180, 255, 120])
    
    # 2. Dark brown lesions (severe browning) - moderately restrictive
    dark_brown_lower = np.array([5, 80, 15])     # Moderate saturation, detect darker lesions
    dark_brown_upper = np.array([20, 255, 110])  # Reasonable brightness range
    
    # 3. Normal brown lesions (moderate browning) - balanced detection
    normal_brown_lower = np.array([8, 70, 60])   # Lower saturation threshold for actual brown lesions
    normal_brown_upper = np.array([25, 255, 170]) # Broader range to catch lesions
    
    # 4. Light brown lesions (mild browning) - DISABLED to avoid healthy corm confusion
    # Setting impossible range to effectively disable light brown detection
    light_brown_lower = np.array([0, 255, 255])  # Impossible values - effectively disabled
    light_brown_upper = np.array([0, 255, 255])  # This will detect nothing
    
    # 5. Yellowish brown lesions (early browning) - more inclusive for actual lesions
    yellowish_brown_lower = np.array([20, 50, 80])   # Lower saturation to catch more lesions
    yellowish_brown_upper = np.array([35, 200, 220]) # Broader range for yellowish browning
    
    # Create masks for each lesion type
    fusarium_red_mask = cv2.inRange(hsv_pixels, fusarium_red_lower, fusarium_red_upper)
    fusarium_purple_mask = cv2.inRange(hsv_pixels, fusarium_purple_lower, fusarium_purple_upper)
    fusarium_mask = fusarium_red_mask | fusarium_purple_mask  # Combine red and purple fusarium lesions
    
    dark_brown_mask = cv2.inRange(hsv_pixels, dark_brown_lower, dark_brown_upper)
    normal_brown_mask = cv2.inRange(hsv_pixels, normal_brown_lower, normal_brown_upper)
    light_brown_mask = cv2.inRange(hsv_pixels, light_brown_lower, light_brown_upper)
    yellowish_brown_mask = cv2.inRange(hsv_pixels, yellowish_brown_lower, yellowish_brown_upper)
    
    # Create white/very light pixel exclusion mask
    # White pixels typically have very low saturation (< 30) and high value (> 200)
    H = hsv_pixels[:, :, 0]
    S = hsv_pixels[:, :, 1] 
    V = hsv_pixels[:, :, 2]
    
    # Exclude white and very light pixels (low saturation, high brightness)
    white_mask = (S < 30) & (V > 230)  # Only exclude very white pixels
    very_light_mask = (S < 20) & (V > 200)  # More selective exclusion for light colors
    
    # Additional exclusions - more selective to preserve actual lesions
    # Exclude very dark pixels that are clearly shadows (very low brightness AND low saturation)
    # BUT preserve dark brown lesions which have higher saturation even if dark
    shadow_mask = (V < 15) & (S < 20)  # Only exclude very dark AND unsaturated pixels (true shadows)
    
    # Exclude healthy corm colors - be more specific about what constitutes healthy tissue
    # Target the specific appearance of healthy white/cream corm tissue
    healthy_corm_mask = (S < 40) & (V > 180) & (H < 30)  # Light, low-saturation, warm-toned areas
    
    # Combine exclusions - removed grey_mask to allow more brown detection
    exclusion_mask = white_mask | very_light_mask | shadow_mask | healthy_corm_mask
    
    # Apply exclusion to lesion masks, but be selective for different lesion types
    # Fusarium and dark brown lesions should be preserved even if they appear dark
    fusarium_exclusion = white_mask | very_light_mask | healthy_corm_mask  # NO shadow mask for fusarium
    dark_brown_exclusion = white_mask | very_light_mask | healthy_corm_mask  # NO shadow mask for dark brown
    normal_brown_exclusion = exclusion_mask
    light_brown_exclusion = exclusion_mask
    yellowish_brown_exclusion = exclusion_mask
    
    # Apply different exclusions for each type
    fusarium_mask = fusarium_mask & ~fusarium_exclusion.astype(np.uint8) * 255
    dark_brown_mask = dark_brown_mask & ~dark_brown_exclusion.astype(np.uint8) * 255
    normal_brown_mask = normal_brown_mask & ~normal_brown_exclusion.astype(np.uint8) * 255
    light_brown_mask = light_brown_mask & ~light_brown_exclusion.astype(np.uint8) * 255
    yellowish_brown_mask = yellowish_brown_mask & ~yellowish_brown_exclusion.astype(np.uint8) * 255
    
    # Combine all lesion types (after exclusion) - include fusarium lesions
    combined_browning_mask = (fusarium_mask > 0) | (dark_brown_mask > 0) | (normal_brown_mask > 0) | (light_brown_mask > 0) | (yellowish_brown_mask > 0)
    
    # Calculate individual counts
    fusarium_pixels = np.sum(fusarium_mask > 0)
    dark_brown_pixels = np.sum(dark_brown_mask > 0)
    normal_brown_pixels = np.sum(normal_brown_mask > 0)
    light_brown_pixels = np.sum(light_brown_mask > 0)
    yellowish_brown_pixels = np.sum(yellowish_brown_mask > 0)
    total_browning_pixels = np.sum(combined_browning_mask)
    
    # Calculate percentages
    total_pixels = len(rgb_pixels)
    fusarium_percent = (fusarium_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    percent_browning = (total_browning_pixels / total_pixels) * 100
    
    # Calculate individual percentages
    dark_brown_percent = (dark_brown_pixels / total_pixels) * 100
    normal_brown_percent = (normal_brown_pixels / total_pixels) * 100
    light_brown_percent = (light_brown_pixels / total_pixels) * 100
    yellowish_brown_percent = (yellowish_brown_pixels / total_pixels) * 100
    
    # Return comprehensive results including fusarium lesions
    browning_breakdown = {
        'fusarium': fusarium_pixels,
        'dark_brown': dark_brown_pixels,
        'normal_brown': normal_brown_pixels, 
        'light_brown': light_brown_pixels,
        'yellowish_brown': yellowish_brown_pixels,
        'total_browning': total_browning_pixels,
        'combined_mask': combined_browning_mask,
        'fusarium_mask': fusarium_mask > 0,
        'dark_brown_mask': dark_brown_mask > 0,
        'normal_brown_mask': normal_brown_mask > 0,
        'light_brown_mask': light_brown_mask > 0,
        'yellowish_brown_mask': yellowish_brown_mask > 0,
        'fusarium_percent': fusarium_percent,
        'dark_brown_percent': dark_brown_percent,
        'normal_brown_percent': normal_brown_percent,
        'light_brown_percent': light_brown_percent,
        'yellowish_brown_percent': yellowish_brown_percent,
        'excluded_pixels': np.sum(exclusion_mask)
    }
    
    return percent_browning, total_browning_pixels, total_pixels, browning_breakdown

def analyze_shape_region(image, shape_mask, ignore_black=True):
    """
    Analyze browning in the selected shape region, ignoring black background
    """
    if shape_mask is None:
        st.warning("No shape selected. Please draw a rectangle or circle on the image.")
        return None
    
    # Create mask for non-black pixels
    if ignore_black:
        # More robust black pixel detection using multiple criteria
        black_threshold = 30  # RGB threshold
        # Method 1: Check if any RGB channel is above threshold
        non_black_mask_1 = np.any(image > black_threshold, axis=2)
        # Method 2: Check if the sum of RGB values is above a threshold (brightness check)
        brightness = np.sum(image, axis=2)
        non_black_mask_2 = brightness > (black_threshold * 2)  # More lenient brightness check
        # Combine both methods (pixel is non-black if either condition is true)
        non_black_mask = non_black_mask_1 | non_black_mask_2
    else:
        non_black_mask = np.ones(image.shape[:2], dtype=bool)
    
    # Combine shape mask with non-black mask
    analysis_mask = shape_mask & non_black_mask
    
    if not np.any(analysis_mask):
        st.warning("No valid pixels found in selected shape region.")
        return None
    
    # Extract pixels for analysis
    selected_pixels = image[analysis_mask]
    
    # Calculate percent browning (multiple brown lesion types)
    percent_browning, browning_pixels, total_corm_pixels, browning_breakdown = calculate_percent_browning(selected_pixels)
    
    # Convert RGB to LAB
    lab_pixels = rgb_to_lab(selected_pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]
    
    # Calculate color metrics using new CDI-based approach
    cdi, hue_deg = calculate_color_dynamics_index(L, a, b)
    excess_red = calculate_excess_red(selected_pixels.reshape(-1, 1, 3)).flatten()
    excess_brown = calculate_excess_brown(selected_pixels.reshape(-1, 1, 3)).flatten()
    delta_e = calculate_delta_e(L, a, b)
    max_delta_e = np.max(delta_e) if len(delta_e) > 0 else 1.0
    fl_ci = calculate_fusarium_lesion_color_index(cdi, delta_e, max_delta_e)
    hue, chroma = calculate_hue_and_chroma(a, b)
    
    # Create full-size arrays for visualization
    full_cdi = np.zeros(image.shape[:2])
    full_er = np.zeros(image.shape[:2])
    full_eb = np.zeros(image.shape[:2])
    full_delta_e = np.zeros(image.shape[:2])
    full_fl_ci = np.zeros(image.shape[:2])
    full_hue = np.zeros(image.shape[:2])
    full_chroma = np.zeros(image.shape[:2])
    
    full_cdi[analysis_mask] = cdi
    full_er[analysis_mask] = excess_red
    full_eb[analysis_mask] = excess_brown
    full_delta_e[analysis_mask] = delta_e
    full_fl_ci[analysis_mask] = fl_ci
    full_hue[analysis_mask] = hue
    full_chroma[analysis_mask] = chroma
    
    # Calculate scaled measurements if scale is available
    mm_per_px = st.session_state.get('mm_per_px', None)
    scale_info = {
        'mm_per_px': mm_per_px,
        'has_scale': mm_per_px is not None
    }
    
    # Calculate areas in real units if scale is available
    if mm_per_px:
        # Convert pixel counts to areas
        total_area_mm2 = total_corm_pixels * (mm_per_px ** 2)
        browning_area_mm2 = browning_pixels * (mm_per_px ** 2)
        
        # Convert breakdown pixel counts to areas
        scaled_breakdown = {}
        for key, value in browning_breakdown.items():
            if isinstance(value, (int, np.integer)) and key != 'fusarium':  # fusarium is already a count
                scaled_breakdown[key + '_area_mm2'] = value * (mm_per_px ** 2)
            elif key.endswith('_mask') and hasattr(value, 'sum'):
                area_key = key.replace('_mask', '_area_mm2')
                scaled_breakdown[area_key] = np.sum(value) * (mm_per_px ** 2)
            scaled_breakdown[key] = value  # Keep original values too
        
        # Add fusarium lesion area
        if 'fusarium' in browning_breakdown:
            scaled_breakdown['fusarium_area_mm2'] = browning_breakdown['fusarium'] * (mm_per_px ** 2)
        
    else:
        # No scale - use pixel measurements
        total_area_mm2 = total_corm_pixels  # Will be labeled as pixels
        browning_area_mm2 = browning_pixels  # Will be labeled as pixels
        scaled_breakdown = browning_breakdown

    return {
        'selected_pixels': selected_pixels,
        'analysis_mask': analysis_mask,
        'shape_mask': shape_mask,
        'L': L,
        'a': a,
        'b': b,
        'color_dynamics_index': cdi,
        'excess_red': excess_red,
        'excess_brown': excess_brown,
        'delta_e': delta_e,
        'fl_ci': fl_ci,
        'hue': hue,
        'chroma': chroma,
        'full_color_dynamics_index': full_cdi,
        'full_excess_red': full_er,
        'full_excess_brown': full_eb,
        'full_delta_e': full_delta_e,
        'full_fl_ci': full_fl_ci,
        'full_hue': full_hue,
        'full_chroma': full_chroma,
        'percent_browning': percent_browning,
        'browning_pixels': browning_pixels,
        'total_corm_pixels': total_corm_pixels,
        'browning_breakdown': scaled_breakdown,
        'scale_info': scale_info,
        'total_area_mm2': total_area_mm2,
        'browning_area_mm2': browning_area_mm2
    }

def extract_selected_region(image, canvas_result, scale_factor):
    """
    Extract the region selected on the canvas
    """
    if canvas_result.image_data is None:
        return None, None
    
    # Get the drawn mask from canvas
    drawn_mask = canvas_result.image_data[:, :, 3] > 0  # Alpha channel
    
    # Scale mask back to original image size if needed
    if scale_factor != 1.0:
        original_height, original_width = image.shape[:2]
        drawn_mask = cv2.resize(
            drawn_mask.astype(np.uint8), 
            (original_width, original_height), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    
    return drawn_mask, canvas_result.image_data

def analyze_selected_region(image, selection_mask, ignore_black=True):
    """
    Analyze browning only in the selected region, ignoring black background
    """
    if selection_mask is None:
        st.warning("No region selected. Please draw on the image to select the corm area.")
        return None
    
    # Create mask for non-black pixels
    if ignore_black:
        # More robust black pixel detection using multiple criteria
        black_threshold = 30  # RGB threshold
        # Method 1: Check if any RGB channel is above threshold
        non_black_mask_1 = np.any(image > black_threshold, axis=2)
        # Method 2: Check if the sum of RGB values is above a threshold (brightness check)
        brightness = np.sum(image, axis=2)
        non_black_mask_2 = brightness > (black_threshold * 2)  # More lenient brightness check
        # Combine both methods (pixel is non-black if either condition is true)
        non_black_mask = non_black_mask_1 | non_black_mask_2
    else:
        non_black_mask = np.ones(image.shape[:2], dtype=bool)
    
    # Combine selection mask with non-black mask
    analysis_mask = selection_mask & non_black_mask
    
    if not np.any(analysis_mask):
        st.warning("No valid pixels found in selected region.")
        return None
    
    # Extract pixels for analysis
    selected_pixels = image[analysis_mask]
    
    # Convert RGB to LAB
    lab_pixels = rgb_to_lab(selected_pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]
    
    # Calculate color metrics using CDI-based approach
    cdi, hue_deg = calculate_color_dynamics_index(L, a, b)
    excess_red = calculate_excess_red(selected_pixels.reshape(-1, 1, 3)).flatten()
    excess_brown = calculate_excess_brown(selected_pixels.reshape(-1, 1, 3)).flatten()
    delta_e = calculate_delta_e(L, a, b)
    max_delta_e = np.max(delta_e) if len(delta_e) > 0 else 1.0
    fl_ci = calculate_fusarium_lesion_color_index(cdi, delta_e, max_delta_e)
    hue, chroma = calculate_hue_and_chroma(a, b)
    
    # Create full-size arrays for visualization
    full_cdi = np.zeros(image.shape[:2])
    full_er = np.zeros(image.shape[:2])
    full_eb = np.zeros(image.shape[:2])
    full_hue = np.zeros(image.shape[:2])
    full_chroma = np.zeros(image.shape[:2])
    
    full_cdi[analysis_mask] = cdi
    full_er[analysis_mask] = excess_red
    full_eb[analysis_mask] = excess_brown
    full_hue[analysis_mask] = hue
    full_chroma[analysis_mask] = chroma

    return {
        'analysis_mask': analysis_mask,
        'mask': selection_mask,
        'L': L,
        'a': a,
        'b': b,
        'color_dynamics_index': cdi,
        'excess_red': excess_red,
        'excess_brown': excess_brown,
        'delta_e': delta_e,
        'fl_ci': fl_ci,
        'hue': hue,
        'chroma': chroma,
        'full_color_dynamics_index': full_cdi,
        'full_excess_red': full_er,
        'full_excess_brown': full_eb,
        'full_hue': full_hue,
        'full_chroma': full_chroma,
        'total_corm_pixels': np.sum(analysis_mask),
        'browning_pixels': np.sum(full_cdi[analysis_mask] > 0.5),  # Using CDI threshold
        'percent_browning': (np.sum(full_cdi[analysis_mask] > 0.5) / np.sum(analysis_mask)) * 100
    }

def rgb_to_lab(rgb_array):
    """
    Convert RGB to CIELAB color space
    """
    # Normalize RGB values to 0-1 range
    rgb_normalized = rgb_array / 255.0
    
    # Convert to LAB using colorspacious
    lab = cspace_convert(rgb_normalized, "sRGB1", "CIELab")
    return lab

def calculate_color_dynamics_index(L, a, b):
    """
    Calculate Color Dynamics Index (CDI) based on hue angle
    CDI = 0 (green), 0.5 (yellow/blue), 1 (red)
    Based on Scalisi et al., 2022
    """
    # Calculate hue angle in degrees
    hue_rad = np.arctan2(b, a)
    hue_deg = np.degrees(hue_rad)
    
    # Normalize to 0-360 range
    hue_deg = np.where(hue_deg < 0, hue_deg + 360, hue_deg)
    
    # Map hue to CDI (0-1 scale)
    # Green ~120°→0, Yellow ~90°→0.5, Red ~0°→1
    # Adjust mapping for fusarium-relevant colors
    cdi = np.zeros_like(hue_deg)
    
    # Green region (90-150°) → CDI near 0
    green_mask = (hue_deg >= 90) & (hue_deg <= 150)
    cdi[green_mask] = (150 - hue_deg[green_mask]) / 150  # 0 at 150°, increases toward 90°
    
    # Yellow region (60-90°) → CDI around 0.5
    yellow_mask = (hue_deg >= 60) & (hue_deg < 90)
    cdi[yellow_mask] = 0.3 + 0.2 * (90 - hue_deg[yellow_mask]) / 30  # 0.3-0.5 range
    
    # Red-Orange region (0-60°) → CDI toward 1
    red_mask = (hue_deg >= 0) & (hue_deg < 60)
    cdi[red_mask] = 0.5 + 0.5 * (60 - hue_deg[red_mask]) / 60  # 0.5-1.0 range
    
    # Purple-Red region (300-360°) → CDI toward 1
    purple_mask = hue_deg >= 300
    cdi[purple_mask] = 0.7 + 0.3 * (hue_deg[purple_mask] - 300) / 60  # 0.7-1.0 range
    
    return cdi, hue_deg

def calculate_excess_red(rgb_pixels):
    """
    Calculate excess red index: ExR = 1.4 * R - G
    Higher values indicate more red coloration
    """
    R = rgb_pixels[:, :, 0] / 255.0
    G = rgb_pixels[:, :, 1] / 255.0
    excess_red = 1.4 * R - G
    return excess_red

def calculate_excess_brown(rgb_pixels):
    """
    Calculate excess brown index: ExBr = 1.6 * R - G - B
    Higher values indicate more brown coloration
    """
    R = rgb_pixels[:, :, 0] / 255.0
    G = rgb_pixels[:, :, 1] / 255.0
    B = rgb_pixels[:, :, 2] / 255.0
    excess_brown = 1.6 * R - G - B
    return excess_brown

def calculate_delta_e(L, a, b, healthy_L=85, healthy_a=-5, healthy_b=20):
    """
    Calculate color difference (ΔE) from healthy tissue in CIELAB space
    Default healthy values represent typical healthy banana corm tissue
    """
    delta_L = L - healthy_L
    delta_a = a - healthy_a  
    delta_b = b - healthy_b
    delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
    return delta_e

def calculate_fusarium_lesion_color_index(cdi, delta_e, max_delta_e, w1=0.7, w2=0.3):
    """
    Calculate Fusarium Lesion Color Index (FL-CI)
    Combines color progression (CDI) and lesion severity (ΔE)
    """
    if max_delta_e > 0:
        normalized_delta_e = np.minimum(delta_e / max_delta_e, 1.0)
    else:
        normalized_delta_e = np.zeros_like(delta_e)
    
    fl_ci = w1 * cdi + w2 * normalized_delta_e
    return fl_ci

def calculate_hue_and_chroma(a, b):
    """
    Calculate hue angle and chroma from CIELAB a* and b* values
    """
    # Hue angle in degrees
    hue = np.arctan2(b, a) * 180 / np.pi
    # Convert negative angles to positive
    hue = np.where(hue < 0, hue + 360, hue)
    
    # Chroma
    chroma = np.sqrt(a**2 + b**2)
    
    return hue, chroma

@cache_resource
def process_image(image):
    """
    Process the uploaded image and calculate browning metrics
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to LAB
    lab_array = rgb_to_lab(img_array)
    L = lab_array[:, :, 0]
    a = lab_array[:, :, 1]
    b = lab_array[:, :, 2]
    
    # Calculate color metrics using CDI approach
    cdi, hue_deg = calculate_color_dynamics_index(L, a, b)
    
    # Calculate excess measurements
    excess_red = calculate_excess_red(img_array)
    excess_brown = calculate_excess_brown(img_array)
    
    # Calculate other metrics
    delta_e = calculate_delta_e(L, a, b)
    max_delta_e = np.max(delta_e) if len(delta_e) > 0 else 1.0
    fl_ci = calculate_fusarium_lesion_color_index(cdi, delta_e, max_delta_e)
    hue, chroma = calculate_hue_and_chroma(a, b)
    
    return {
        'original': img_array,
        'lab': lab_array,
        'L': L,
        'a': a,
        'b': b,
        'color_dynamics_index': cdi,
        'excess_red': excess_red,
        'excess_brown': excess_brown,
        'delta_e': delta_e,
        'fl_ci': fl_ci,
        'hue': hue,
        'chroma': chroma
    }

def create_lesion_mask(cdi, threshold=0.5):
    """
    Create a mask for lesion areas based on CDI threshold
    CDI > 0.5 indicates progression toward red/brown lesions
    """
    return cdi > threshold

def calculate_lesion_percentage(lesion_mask):
    """
    Calculate the percentage of the image that shows lesions based on CDI
    """
    total_pixels = lesion_mask.size
    lesion_pixels = np.sum(lesion_mask)
    percent_lesions = (lesion_pixels / total_pixels) * 100
    return percent_lesions

def main():
    st.set_page_config(
        page_title="Banana Corm Browning Analyzer",
        page_icon="🍌",
        layout="wide"
    )
    # Page configuration and title
    st.title("🍌 Banana Corm Browning Analyzer")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='margin: 0; color: #1f77b4;'>📋 Workflow: Upload → Select Area → Analyze</h4>
    <p style='margin: 5px 0 0 0; color: #666;'>Accurate browning detection with multiple lesion types</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data recovery option (ask user instead of auto-recovering)
    if hasattr(st.session_state, 'show_recovery_option') and st.session_state.show_recovery_option and not st.session_state.data_recovery_shown:
        recovered_count = len(st.session_state.potential_recovery_data)
        
        st.info(f"� **Previous Session Found!** Found {recovered_count} batch result(s) from your last session.")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("🔄 Restore Data", type="primary", help="Load previous session results"):
                st.session_state.batch_results = st.session_state.potential_recovery_data.copy()
                st.session_state.data_recovery_shown = True
                st.session_state.show_recovery_option = False
                st.success(f"✅ Restored {recovered_count} previous results!")
                rerun_app()
        
        with col2:
            if st.button("🆕 Start Fresh", type="secondary", help="Ignore previous data and start new"):
                st.session_state.batch_results = []
                st.session_state.data_recovery_shown = True  
                st.session_state.show_recovery_option = False
                clear_saved_batch_data()  # Remove the saved file since user chose fresh start
                st.success("✅ Starting with fresh batch!")
                rerun_app()
                
        with col3:
            if st.button("❌ Dismiss", help="Hide this message, keep current empty state"):
                st.session_state.data_recovery_shown = True
                st.session_state.show_recovery_option = False
                rerun_app()
    
    # Initialize session state with memory optimization
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'canvas_result' not in st.session_state:
        st.session_state.canvas_result = None
    
    # Initialize batch processing session state with limits for cloud stability
    if 'batch_results' not in st.session_state:
        # Check for previous session data but don't auto-load
        potential_recovery_data = load_batch_data()
        st.session_state.batch_results = []  # Always start empty
        
        # Track recovery state
        if 'data_recovery_shown' not in st.session_state:
            st.session_state.data_recovery_shown = False
            if potential_recovery_data:
                st.session_state.potential_recovery_data = potential_recovery_data
                st.session_state.show_recovery_option = True  # Show user choice
            else:
                st.session_state.show_recovery_option = False
    if 'current_batch_index' not in st.session_state:
        st.session_state.current_batch_index = 0
    if 'processing_mode' not in st.session_state:
        st.session_state.processing_mode = "Single Image"
    
    # Cloud deployment: Limit batch size to prevent memory issues
    MAX_BATCH_SIZE = 20  # Limit to 20 images per batch for cloud stability
    
    # Cloud performance optimization: Enhanced memory management
    if 'cleanup_counter' not in st.session_state:
        st.session_state.cleanup_counter = 0
    st.session_state.cleanup_counter += 1
    
    # More aggressive cleanup for cloud stability
    if st.session_state.cleanup_counter % 5 == 0:
        import gc
        import os
        
        # Clear old analysis results to free memory
        if hasattr(st.session_state, 'analysis_results') and st.session_state.cleanup_counter % 20 == 0:
            # Keep only essential data for display, clear large arrays
            if 'analysis_results' in st.session_state and st.session_state.analysis_results:
                # Remove large image arrays from old results
                for key in ['original', 'lab', 'L', 'a', 'b']:
                    if key in st.session_state.analysis_results:
                        del st.session_state.analysis_results[key]
        
        # Aggressive garbage collection
        gc.collect()
    
    # Sidebar - Streamlined
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Processing Mode Selection
        st.subheader("📊 Processing Mode")
        processing_mode = st.radio(
            "Choose workflow:",
            ["Single Image", "Batch Processing"],
            index=0 if st.session_state.processing_mode == "Single Image" else 1,
            help="Single: Analyze one image at a time\nBatch: Analyze multiple images and combine results"
        )
        
        # Clean up when switching processing modes for memory efficiency
        if st.session_state.processing_mode != processing_mode:
            # Clear old analysis data when switching modes
            if 'analysis_results' in st.session_state:
                del st.session_state.analysis_results
            if 'processed_image' in st.session_state:
                st.session_state.processed_image = None
            if 'canvas_result' in st.session_state:
                st.session_state.canvas_result = None
            import gc
            gc.collect()
            
        st.session_state.processing_mode = processing_mode
        
        # File Upload Section
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose corm image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of banana corm cross-section"
        )
        
        # Batch Processing Status
        if processing_mode == "Batch Processing":
            st.markdown("---")
            st.subheader("📊 Batch Status")
            
            total_analyzed = len(st.session_state.batch_results)
            if total_analyzed > 0:
                st.success(f"✅ {total_analyzed} images analyzed")
                
                # Show batch results summary
                if st.button("📋 View Combined Results"):
                    st.session_state.show_batch_summary = True
                
                # Export all results
                if st.button("📥 Export All Results"):
                    batch_csv = create_batch_export()
                    st.download_button(
                        label="Download Batch Results CSV",
                        data=batch_csv,
                        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Clear batch results
                if st.button("🗑️ Clear All Results", type="secondary"):
                    st.session_state.batch_results = []
                    st.session_state.current_batch_index = 0
                    clear_saved_batch_data()  # Remove saved file when user clears
                    rerun_app()
            else:
                st.info("Upload and analyze images to build your batch results")
        
        # Quick help in sidebar
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            **Selection Tools:**
            - Rectangle: Regular shapes
            - Circle: Round corms
            - Polygon: Irregular shapes
            
            **Best Results:**
            - Good lighting
            - Clear corm boundaries
            - Minimal shadows
            """)
        
        # Clear & Start Over section
        st.markdown("---")
        st.subheader("🔄 Start Over")
        if st.button("🗑️ Clear & Start Over", type="secondary", help="Clear all data and start with a new image"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
    

    if uploaded_file is not None:
        try:
            # Load and resize the original image if too large
            original_image = Image.open(uploaded_file)
            if max(original_image.size) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(original_image.size)
                new_size = (int(original_image.size[0]*scale), int(original_image.size[1]*scale))
                original_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                st.info(f"Image resized to {new_size} for performance.")

            # Step 1: Scale Calibration
            st.markdown("---")
            st.subheader("📏 Step 1: Scale Calibration")
            st.markdown("Draw a line on a known measurement (ruler/scale) in your image to set the scale for measurements.")
            
            # Initialize scale session state
            if 'mm_per_px' not in st.session_state:
                st.session_state.mm_per_px = None
            
            # Convert PIL to numpy for canvas
            image_np = np.array(original_image)
            
            # Ensure image is in RGB format for cloud compatibility
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # Convert RGBA to RGB
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Convert back to PIL for canvas (cloud-compatible format)
            canvas_background = Image.fromarray(image_rgb.astype('uint8'), 'RGB')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Scale calibration canvas
                try:
                    # Initialize clear counter if not exists
                    if 'scale_canvas_clear_counter' not in st.session_state:
                        st.session_state.scale_canvas_clear_counter = 0
                    
                    scale_canvas = st_canvas(
                        fill_color="rgba(0,0,0,0)",
                        stroke_width=3,
                        stroke_color="rgba(255,0,0,1)",
                        background_image=canvas_background,
                        update_streamlit=True,
                        height=min(image_rgb.shape[0], 400),  # Limit height for better display
                        width=min(image_rgb.shape[1], 600),   # Limit width for better display
                        drawing_mode="line",
                        key=f"scale_calibration_canvas_{st.session_state.scale_canvas_clear_counter}",
                        display_toolbar=False,
                    )
                except Exception as e:
                    st.error(f"Canvas error: {e}")
                    st.info(f"Image format: {original_image.format}, Mode: {original_image.mode}, Size: {original_image.size}")
                    st.info("💡 **Workaround:** If canvas appears white, try:")
                    st.markdown("- Upload a different image format (JPG instead of PNG)")
                    st.markdown("- Refresh the page and try again")
                    st.markdown("- Use a smaller image size")
                    scale_canvas = None
            
            with col2:
                st.markdown("**Instructions:**")
                st.markdown("1. Draw a line over a known measurement in your image")
                st.markdown("2. Enter the real-world length below")
                st.markdown("3. The scale will be calculated automatically")
                
                # Clear scale canvas button
                if st.button("🧹 Clear Scale Line", help="Remove all drawn lines from scale canvas", type="secondary"):
                    # Force canvas refresh by incrementing a counter
                    if 'scale_canvas_clear_counter' not in st.session_state:
                        st.session_state.scale_canvas_clear_counter = 0
                    st.session_state.scale_canvas_clear_counter += 1
                    rerun_app()
                
                # Input for real measurement
                scale_length_mm = st.number_input(
                    "Real length of drawn line (mm):", 
                    min_value=0.1, 
                    value=10.0, 
                    step=1.0,
                    help="Enter the actual measurement of the line you drew"
                )
                
                # Calculate scale from drawn line
                scale_px = None
                if scale_canvas and hasattr(scale_canvas, 'json_data') and scale_canvas.json_data:
                    objects = scale_canvas.json_data.get("objects", [])
                    if objects and isinstance(objects, list):
                        obj = objects[-1]  # Get the last drawn line
                        if obj and isinstance(obj, dict) and obj.get("type") == "line":
                            try:
                                x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                                x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                                scale_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                                if scale_px > 0:
                                    st.success(f"✅ Line drawn: {scale_px:.1f} pixels")
                                else:
                                    scale_px = None
                            except Exception as e:
                                st.error(f"Error calculating line length: {e}")
                                scale_px = None
                
                # Calculate and store scale
                if scale_px and scale_length_mm > 0:
                    st.session_state.mm_per_px = scale_length_mm / scale_px
                    st.success(f"� Scale set: {st.session_state.mm_per_px:.3f} mm/pixel")
                    st.info(f"1 pixel = {st.session_state.mm_per_px:.3f} mm")
                elif scale_px:
                    st.warning("⚠️ Please enter a valid measurement length")
                else:
                    st.info("👆 Draw a line on the ruler/scale in the image")
                
                # Skip scale option
                if st.button("⏭️ Skip Scale Calibration"):
                    st.session_state.mm_per_px = None
                    st.warning("Scale calibration skipped. Measurements will be in pixels only.")

            # Processing options in sidebar
            st.markdown("---")
            st.subheader("�🔧 Step 2: Image Processing")
            processing_option = st.selectbox(
                "Processing method:",
                ["Skip Processing", "Color Filtering", "AI Background Removal"],
                help="Choose how to prepare the image"
            )

            if st.button("📋 Prepare Image", type="primary"):
                progress_bar = st.progress(0, text="Starting image processing...")
                try:
                    progress_bar.progress(20, text="Loading image data...")
                    img_array = np.array(original_image)
                    
                    if processing_option == "Skip Processing":
                        progress_bar.progress(80, text="Finalizing...")
                        processed_image = img_array
                    elif processing_option == "Color Filtering":
                        progress_bar.progress(40, text="Applying color filters...")
                        alpha_mask = np.ones(img_array.shape[:2], dtype=bool)
                        processed_image = filter_corm_colors(img_array, alpha_mask)
                        progress_bar.progress(80, text="Finalizing...")
                    else:  # AI Background Removal
                        progress_bar.progress(30, text="Removing background (this may take a moment)...")
                        processed_image = remove_background_and_filter_colors(original_image)
                        progress_bar.progress(70, text="Processing results...")
                    
                    progress_bar.progress(85, text="Optimizing for cloud performance...")
                    # Ensure processed image is not too large for memory
                    if processed_image.size > 2000000:  # ~2MP limit
                        st.warning("Large image detected. Optimizing for cloud performance...")
                        height, width = processed_image.shape[:2]
                        scale = np.sqrt(2000000 / processed_image.size)
                        new_height, new_width = int(height * scale), int(width * scale)
                        processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    progress_bar.progress(100, text="Complete!")
                    st.session_state.processed_image = processed_image
                    st.success("✅ Image prepared!")
                    progress_bar.empty()  # Remove progress bar when done
                except Exception as e:
                    progress_bar.empty()  # Clean up progress bar on error
                    st.error(f"Image processing failed: {e}")
                    st.info("Try using 'Skip Processing' option for large or complex images.")

            # Display images
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original_image, caption="📷 Original Image", use_column_width=True)
            with col2:
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, caption="🎨 Processed Image", use_column_width=True)
                else:
                    st.info("Process the image to prepare for analysis")

            # Step 3: Interactive Selection (only show if image is processed)
            if st.session_state.processed_image is not None:
                st.markdown("---")
                st.subheader("🎯 Step 3: Select Analysis Area")
                
                # Show current scale status
                if st.session_state.mm_per_px:
                    st.info(f"📏 Scale: {st.session_state.mm_per_px:.3f} mm/pixel - Measurements will include real-world dimensions")
                else:
                    st.info("📏 No scale set - Measurements will be in pixels only")
                try:
                    # Debug: Check if processed_image exists and is valid
                    if st.session_state.processed_image is None:
                        st.error("Processed image is None. Please reprocess the image.")
                        return
                    
                    # Debug: Show image info
                    if hasattr(st.session_state.processed_image, 'shape'):
                        st.info(f"Debug: Image shape: {st.session_state.processed_image.shape}")
                    
                    canvas_result, scale_factor, shape_type = create_selection_canvas(
                        st.session_state.processed_image, 
                        canvas_key="corm_selection"
                    )
                    
                    if canvas_result is None:
                        st.error("Canvas creation failed. Image may be corrupted.")
                        return
                        
                except Exception as e:
                    st.error(f"Canvas error: {e}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
                    return

                # Auto-analysis for completed polygons
                auto_analyze = False
                polygon_completed = st.session_state.get("corm_selection_polygon_completed", False)
                
                if polygon_completed and shape_type == "Polygon":
                    st.success("🎯 Polygon completed! Auto-analyzing...")
                    auto_analyze = True
                    # Reset the completion flag
                    st.session_state["corm_selection_polygon_completed"] = False
                
                # Quick Analyze button: available immediately after drawing (helps point-mode polygons)
                if canvas_result is not None:
                    st.info("💡 **Tip**: If polygon isn't detected, try clicking the radio button above (Point mode ↔ Freehand) to refresh the canvas data, then analyze.")
                    
                    # Auto-analyze or manual button
                    analyze_triggered = auto_analyze or st.button("🔬 Analyze Area", key="analyze_area_quick")
                    
                    if analyze_triggered:
                        with st.spinner("🔬 Attempting analysis..."):
                            try:
                                selection_mask = extract_shape_mask(
                                    canvas_result,
                                    scale_factor,
                                    shape_type,
                                    st.session_state.processed_image.shape
                                )
                                if selection_mask is None or not np.any(selection_mask):
                                    st.warning("⚠️ No valid polygon selection found. **Instructions:**")
                                    st.markdown("""
                                    - **Point mode**: Click at least 3 points on the image, then **double-click the last point** or **press ESC** to close the polygon
                                    - **Freehand mode**: Draw a closed shape by dragging the mouse
                                    - Make sure the polygon is **completely closed** before analyzing
                                    - **Try clicking the radio button above (Point mode ↔ Freehand)** to refresh canvas data
                                    """)
                                else:
                                    with st.spinner("🔍 Analyzing selected region..."):
                                        analysis_results = analyze_shape_region(
                                            st.session_state.processed_image,
                                            selection_mask,
                                            ignore_black=True
                                        )
                                    if analysis_results is not None:
                                        st.success("✅ Analysis complete!")
                                        st.session_state.analysis_results = analysis_results
                                        # Add to batch results if in batch mode
                                        if uploaded_file:
                                            add_to_batch_results(analysis_results, uploaded_file.name)
                                    else:
                                        st.warning("⚠️ Analysis failed. Please try selecting a different area.")
                            except Exception as e:
                                st.error(f"Error during quick analysis: {e}")

                # Step 4: Analysis (only show if selection exists)
                has_json_objects = (canvas_result.json_data is not None and 
                                  len(canvas_result.json_data.get("objects", [])) > 0)
                has_alpha_data = (canvas_result.image_data is not None and 
                                np.any(canvas_result.image_data[:, :, 3] > 0))
                if has_json_objects or has_alpha_data:
                    st.markdown("---")
                    try:
                        selection_mask = extract_shape_mask(
                            canvas_result, 
                            scale_factor, 
                            shape_type,
                            st.session_state.processed_image.shape
                        )
                        if selection_mask is None:
                            st.warning("⚠️ Could not extract selection. Please try drawing a different shape.")
                            if shape_type == "Polygon" and canvas_result.image_data is not None:
                                st.info("🔄 Trying alternative polygon extraction...")
                                alpha_data = canvas_result.image_data[:, :, 3]
                                if np.any(alpha_data > 0):
                                    selection_mask = alpha_data > 0
                                    if selection_mask.shape != st.session_state.processed_image.shape[:2]:
                                        original_height, original_width = st.session_state.processed_image.shape[:2]
                                        selection_mask = cv2.resize(
                                            selection_mask.astype(np.uint8), 
                                            (original_width, original_height), 
                                            interpolation=cv2.INTER_NEAREST
                                        ).astype(bool)
                                    st.success("✅ Alternative polygon extraction successful!")
                                else:
                                    selection_mask = None
                        elif not np.any(selection_mask):
                            st.warning("⚠️ Empty selection detected. Please draw a larger area.")
                            selection_mask = None
                    except Exception as e:
                        st.error(f"Error extracting selection: {e}")
                        selection_mask = None
                    if selection_mask is not None and np.any(selection_mask):
                        st.markdown("### 🔬 Analysis (automatic)")
                        canvas_sig = None
                        try:
                            if canvas_result is not None and canvas_result.json_data is not None:
                                canvas_sig = json.dumps(canvas_result.json_data, sort_keys=True)
                            elif canvas_result is not None and canvas_result.image_data is not None:
                                alpha_sum = int(np.sum(canvas_result.image_data[:, :, 3] > 0))
                                canvas_sig = f"alpha_sum:{alpha_sum}"
                        except Exception:
                            canvas_sig = None
                        canvas_hash = hashlib.md5(canvas_sig.encode()).hexdigest() if canvas_sig is not None else None
                        if 'last_canvas_hash' not in st.session_state:
                            st.session_state.last_canvas_hash = None
                        if canvas_hash is not None and canvas_hash != st.session_state.last_canvas_hash:
                            with st.spinner("🔬 Running automatic analysis..."):
                                try:
                                    analysis_results = analyze_shape_region(
                                        st.session_state.processed_image,
                                        selection_mask,
                                        ignore_black=True
                                    )
                                    if analysis_results is not None:
                                        st.success("✅ Analysis complete!")
                                        st.session_state.analysis_results = analysis_results
                                        # Add to batch results if in batch mode
                                        if uploaded_file:
                                            add_to_batch_results(analysis_results, uploaded_file.name)
                                    else:
                                        st.warning("⚠️ Analysis failed. Please try selecting a different area.")
                                except Exception as e:
                                    st.error(f"Automatic analysis error: {e}")
                            st.session_state.last_canvas_hash = canvas_hash
                    # Results will be displayed at the end if available
                else:
                    st.info("🎯 Draw a shape on the image above to select the analysis area.")
                    st.markdown("### 🔧 Alternative Analysis")
                    if st.button("🔬 Analyze Entire Image", type="secondary"):
                        img_array = st.session_state.processed_image
                        height, width = img_array.shape[:2]
                        full_mask = np.ones((height, width), dtype=bool)
                        black_threshold = 30
                        non_black_mask = np.any(img_array > black_threshold, axis=2)
                        full_mask = full_mask & non_black_mask
                        if np.any(full_mask):
                            with st.spinner("🔬 Analyzing entire image..."):
                                analysis_results = analyze_shape_region(
                                    st.session_state.processed_image, 
                                    full_mask, 
                                    ignore_black=True
                                )
                            if analysis_results is not None:
                                st.success("✅ Full image analysis complete!")
                                st.session_state.analysis_results = analysis_results
                                # Add to batch results if in batch mode
                                if uploaded_file:
                                    add_to_batch_results(analysis_results, uploaded_file.name)
                            else:
                                st.warning("⚠️ Analysis failed.")
                        else:
                            st.warning("⚠️ No valid pixels found in image.")
                # Alternative: Show analysis button even if no objects (for canvas fallback)
                if canvas_result.image_data is not None:
                    alpha_data = canvas_result.image_data[:, :, 3]
                    if np.any(alpha_data > 0):
                        st.markdown("---")
                        st.info("🔍 Freeform selection detected")
                        if st.button("� Analyze Freeform Selection", type="primary"):
                            drawn_mask = alpha_data > 0
                            if drawn_mask.shape != st.session_state.processed_image.shape[:2]:
                                original_height, original_width = st.session_state.processed_image.shape[:2]
                                drawn_mask = cv2.resize(
                                    drawn_mask.astype(np.uint8), 
                                    (original_width, original_height), 
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                            if np.any(drawn_mask):
                                with st.spinner("🔬 Analyzing freeform selection..."):
                                    analysis_results = analyze_shape_region(
                                        st.session_state.processed_image, 
                                        drawn_mask, 
                                        ignore_black=True
                                    )
                                if analysis_results is not None:
                                    st.success("✅ Analysis complete!")
                                    st.session_state.analysis_results = analysis_results
                                    # Add to batch results if in batch mode
                                    if uploaded_file:
                                        add_to_batch_results(analysis_results, uploaded_file.name)
                                else:
                                    st.warning("⚠️ Analysis failed. Please try selecting a different area.")
                    else:
                        st.info("🎯 Please draw a shape on the image above to select the analysis area.")
                if 'analysis_results' in st.session_state:
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    display_analysis_results(st.session_state.analysis_results, context="unified_results")
                
                # Show batch processing summary if enabled
                if processing_mode == "Batch Processing" and st.session_state.batch_results:
                    st.markdown("---")
                    st.markdown("## 📊 Batch Processing Summary")
                    
                    # Summary statistics
                    total_images = len(st.session_state.batch_results)
                    avg_browning = np.mean([r['analysis_results']['percent_browning'] for r in st.session_state.batch_results])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", total_images)
                    with col2:
                        st.metric("Average Browning", f"{avg_browning:.1f}%")
                    with col3:
                        total_area = sum([r['analysis_results'].get('total_area_mm2', 0) for r in st.session_state.batch_results])
                        st.metric("Total Area Analyzed", f"{total_area:.1f} mm²")
                    
                    # Batch results table
                    st.subheader("📋 Individual Results")
                    
                    # Create summary table
                    # Interactive batch results table with delete functionality
                    st.markdown("### 📊 Batch Results")
                    
                    for i, result in enumerate(st.session_state.batch_results):
                        analysis = result['analysis_results']
                        
                        # Create columns for data and delete button
                        col1, col2 = st.columns([6, 1])
                        
                        with col1:
                            # Display result info in an expandable format
                            with st.expander(f"🖼️ **{result['image_name']}** - {analysis['percent_browning']:.1f}% browning", expanded=False):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Total Area", f"{analysis.get('total_area_mm2', 0):.1f} mm²")
                                with col_b:
                                    st.metric("Lesion Area", f"{analysis.get('browning_area_mm2', 0):.1f} mm²")
                                with col_c:
                                    st.metric("Analyzed", result['timestamp'][:16])
                        
                        with col2:
                            # Delete button for this specific result
                            if st.button("🗑️", key=f"delete_{i}", help=f"Delete {result['image_name']}", type="secondary"):
                                # Confirmation dialog using session state
                                st.session_state[f'confirm_delete_{i}'] = True
                        
                        # Handle confirmation
                        if st.session_state.get(f'confirm_delete_{i}', False):
                            st.warning(f"⚠️ Delete **{result['image_name']}**? This cannot be undone.")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Yes, Delete", key=f"confirm_yes_{i}", type="primary"):
                                    # Remove the item and save data
                                    del st.session_state.batch_results[i]
                                    st.session_state[f'confirm_delete_{i}'] = False
                                    save_batch_data()  # Auto-save after deletion
                                    st.success(f"✅ Deleted {result['image_name']}")
                                    rerun_app()
                            with col_no:
                                if st.button("❌ Cancel", key=f"confirm_no_{i}"):
                                    st.session_state[f'confirm_delete_{i}'] = False
                                    rerun_app()
                        
                        st.markdown("---")  # Separator between results
                    
                    # Export button
                    if st.button("📥 Export Batch Results", type="primary"):
                        with st.spinner("📊 Generating CSV export..."):
                            batch_csv = create_batch_export()
                        st.download_button(
                            label="Download Batch CSV",
                            data=batch_csv,
                            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.info("📁 Please upload a banana corm image to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 30px;'>
        <p style='margin: 0; color: #666; font-size: 14px;'>
            <strong>🧬 Developed by the Plant Pathology Laboratory, Institute of Plant Breeding, UPLB</strong><br>
            🌍 Co-funded by the Gates Foundation<br>
            📧 For inquiries contact: <a href="mailto:jsmendoza5@up.edu.ph">jsmendoza5@up.edu.ph</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"App crashed: {e}")
