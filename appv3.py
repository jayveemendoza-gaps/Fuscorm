import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
from colorspacious import cspace_convert
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_drawable_canvas import st_canvas
from rembg import remove
from sklearn.cluster import KMeans
import io
import base64
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

@cache_resource
def remove_background_and_filter_colors(image):
    """
    Remove background and keep only corm colors (green, white, yellow, brown)
    Less aggressive approach to preserve corm tissue
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Try background removal, but with fallback options
    try:
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Remove background
        no_bg = remove(img_bytes.getvalue())
        no_bg_image = Image.open(io.BytesIO(no_bg)).convert('RGBA')
        no_bg_array = np.array(no_bg_image)
        
        # Create mask from alpha channel - be more permissive
        alpha_mask = no_bg_array[:, :, 3] > 50  # Lower threshold
        
        # If too much was removed, use a simpler approach
        if np.sum(alpha_mask) < (alpha_mask.size * 0.1):  # If less than 10% remains
            st.warning("Background removal was too aggressive. Using color-based segmentation instead.")
            # Fallback: use color-based background detection
            alpha_mask = create_simple_mask(img_array)
            no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
        
    except Exception as e:
        st.warning(f"Background removal failed: {e}. Using color-based segmentation.")
        # Fallback: use simple color-based mask
        alpha_mask = create_simple_mask(img_array)
        no_bg_array = np.concatenate([img_array, np.ones((*img_array.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
    
    # Filter colors to keep only corm-relevant colors
    filtered_image = filter_corm_colors(no_bg_array[:, :, :3], alpha_mask)
    
    return filtered_image

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
    More inclusive approach to preserve all corm tissue
    """
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # More inclusive color ranges to capture all corm tissue variations
    color_ranges = {
        'white_cream': [(0, 0, 150), (180, 40, 255)],      # White to cream colors
        'yellow_light': [(10, 20, 120), (40, 255, 255)],   # Yellow range (expanded)
        'green_all': [(30, 20, 40), (90, 255, 255)],       # All green variations
        'brown_orange': [(5, 30, 30), (25, 255, 220)],     # Brown/orange range
        'tan_beige': [(15, 10, 100), (35, 80, 200)],       # Tan/beige colors
        'light_colors': [(0, 0, 80), (180, 50, 255)],      # Very light colors
        'warm_tones': [(0, 20, 50), (30, 255, 255)]        # Warm tones (red-yellow)
    }
    
    # Create combined mask for all corm colors
    combined_mask = np.zeros(hsv.shape[:2], dtype=bool)
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        color_mask = cv2.inRange(hsv, lower, upper)
        combined_mask |= (color_mask > 0)
    
    # Additional approach: keep pixels that are not too dark or too saturated
    # (to catch edge cases)
    brightness_mask = hsv[:, :, 2] > 30  # Not too dark
    low_saturation_mask = hsv[:, :, 1] < 200  # Not too saturated (catches whites/greys)
    additional_mask = brightness_mask & low_saturation_mask
    
    # Combine all masks
    final_color_mask = combined_mask | additional_mask
    
    # Combine with alpha mask (object presence) - be more permissive
    final_mask = final_color_mask & mask
    
    # Apply morphological operations to fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    final_mask = final_mask.astype(bool)
    
    # Create output image with black background
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
            canvas_image = Image.fromarray(image.astype('uint8'))
        else:
            canvas_image = image
        
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
            "polygon": "🔷 Click points to create polygon vertices, then double-click or press ESC to close the shape",
            "freedraw": "🖊️ Draw any shape by dragging your mouse",
            "transform": "✏️ Click and drag shapes to resize/move"
        }
        
        st.info(instruction_map.get(drawing_mode, "Select area for analysis"))
        # Extra clarification for polygon point mode (users often click and see dots)
        if shape_type == "Polygon" and polygon_input_mode is not None and "Point mode" in polygon_input_mode:
            st.info("🔷 **Polygon Point Mode:** Click on the image to add polygon vertices. Click at least 3 points to define your shape. Double-click the last point or press ESC to close/finish the polygon. The area inside your polygon will be selected for analysis.")
        
        # Create canvas with cloud-optimized settings
        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.2)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=canvas_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key=canvas_key,
                display_toolbar=False,  # Hide toolbar for cleaner interface
                point_display_radius=3,  # Smaller points for better performance
            )
        except Exception as canvas_error:
            st.error(f"Canvas initialization failed: {canvas_error}")
            st.warning("Falling back to alternative selection method...")
            return create_alternative_selection(image, canvas_key + "_fallback")
        
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
                    if "path" in shape_obj:
                        # Parse path data for complex polygons
                        points = parse_svg_path(shape_obj["path"], scale_factor)
                    elif "points" in shape_obj:
                        # Direct points array
                        points = [(int(p["x"] / scale_factor), int(p["y"] / scale_factor)) for p in shape_obj["points"]]
                    
                    if len(points) >= 3:
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
                        st.info(f"✅ Polygon selected (JSON method): {len(points)} points - Area: {area} pixels")
                        return mask
                        
                except Exception as e:
                    st.warning(f"Could not extract polygon from JSON data: {e}")
            
            # If both methods failed, provide helpful guidance
            st.warning("⚠️ Polygon not detected. For Point mode: Click at least 3 points, then double-click the last point or press ESC to close the polygon. For Freehand mode: Draw a closed shape.")
            return None
        
        elif shape_type == "Freeform":
            # For freeform, use alpha channel method (freedraw mode works reliably)
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
                    st.info(f"✅ Freeform selected - Area: {area} pixels")
                    return drawn_mask
            
            # If no alpha data
            st.warning("No freeform selection detected. Please draw a freeform shape on the image.")
            return None
    
    # If json_data method failed or for freeform polygons, try alpha channel method
    if canvas_result.image_data is not None:
        alpha_data = canvas_result.image_data[:, :, 3]
        if np.any(alpha_data > 0):
            # Get original image dimensions
            original_height, original_width = original_shape[:2]
            
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
            st.info(f"✅ Freeform selection detected - Area: {area} pixels")
            return drawn_mask
    
    # No valid selection found
    return None

def display_analysis_results(analysis_results, context="main"):
    """
    Display analysis results in a clean, organized format
    """
    # Key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔬 Total Browning", 
            f"{analysis_results['percent_browning']:.1f}%",
            help="Overall percentage of browning detected"
        )
    
    with col2:
        st.metric(
            "📊 CDI Index", 
            f"{np.mean(analysis_results['color_dynamics_index']):.2f}",
            help="Color Dynamics Index: 0=green, 0.5=yellow, 1=red (Scalisi et al., 2022)"
        )
    
    with col3:
        st.metric(
            "🧮 Analyzed Pixels", 
            f"{analysis_results['total_corm_pixels']:,}",
            help="Number of pixels analyzed"
        )
    
    with col4:
        breakdown = analysis_results['browning_breakdown']
        fusarium_lesions = breakdown['fusarium']
        st.metric(
            "� Fusarium Lesions", 
            f"{fusarium_lesions:,}",
            help="Number of fusarium-characteristic lesion pixels (red/purple)"
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
            
            st.markdown(f"� **Dark Brown (Severe):** {breakdown['dark_brown_percent']:.1f}% ({dark_count:,} pixels)")
            st.markdown(f"� **Normal Brown (Moderate):** {breakdown['normal_brown_percent']:.1f}% ({normal_count:,} pixels)")
            st.markdown(f"🟡 **Yellowish Brown (Early):** {breakdown['yellowish_brown_percent']:.1f}% ({yellow_count:,} pixels)")
            
            # Show exclusions if any
            total_excluded = breakdown.get('excluded_pixels', 0)
            if total_excluded > 0:
                st.write(f"⚪ Excluded (shadows/highlights): {total_excluded:,} pixels")
        
        with col2:
            st.markdown("**Color Analysis (CDI-based):**")
            if len(analysis_results['L']) > 0:
                st.write(f"💡 Avg Lightness: {np.mean(analysis_results['L']):.1f}")
                st.write(f"🔴 Avg Red-Green: {np.mean(analysis_results['a']):.1f}")
                st.write(f"🟡 Avg Blue-Yellow: {np.mean(analysis_results['b']):.1f}")
                st.write(f"🌈 Avg CDI: {np.mean(analysis_results['color_dynamics_index']):.2f}")
                st.write(f"🔺 Avg ΔE: {np.mean(analysis_results['delta_e']):.1f}")
                st.write(f"🔴 Avg Excess Red: {np.mean(analysis_results['excess_red']):.2f}")
                st.write(f"🟤 Avg Excess Brown: {np.mean(analysis_results['excess_brown']):.2f}")
                st.write(f"🧬 Avg FL-CI: {np.mean(analysis_results['fl_ci']):.2f}")
            
            # Add legend for visualization
            st.markdown("**Visualization Legend:**")
            st.markdown("🟢 Green: Analysis area")
            st.markdown("🔴 Bright Red: Fusarium lesions (critical)")
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
    """Create CSV export data with CDI-based metrics"""
    from datetime import datetime
    import io
    
    # Create summary data
    breakdown = analysis_results['browning_breakdown']
    
    data = [
        ["Metric", "Value"],
        ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Total Browning Percentage", f"{analysis_results['percent_browning']:.2f}%"],
        ["Total Analyzed Pixels", analysis_results['total_corm_pixels']],
        ["Total Lesion Pixels", breakdown['total_browning']],
        ["Fusarium Lesion Pixels", breakdown['fusarium']],
        ["Dark Brown Pixels", breakdown['dark_brown']],
        ["Normal Brown Pixels", breakdown['normal_brown']],
        ["Yellowish Brown Pixels", breakdown['yellowish_brown']],
        ["Light Brown Pixels", breakdown['light_brown']],
        ["Average CDI (Color Dynamics Index)", f"{np.mean(analysis_results['color_dynamics_index']):.3f}"],
        ["Average FL-CI (Fusarium Lesion Color Index)", f"{np.mean(analysis_results['fl_ci']):.3f}"],
        ["Average Delta E", f"{np.mean(analysis_results['delta_e']):.2f}"],
        ["Average Excess Red", f"{np.mean(analysis_results['excess_red']):.3f}"],
        ["Average Excess Brown", f"{np.mean(analysis_results['excess_brown']):.3f}"],
        ["Average Lightness (L*)", f"{np.mean(analysis_results['L']):.2f}"],
        ["Average Red-Green (a*)", f"{np.mean(analysis_results['a']):.2f}"],
        ["Average Blue-Yellow (b*)", f"{np.mean(analysis_results['b']):.2f}"],
    ]
    
    # Convert to CSV
    output = io.StringIO()
    for row in data:
        output.write(",".join(map(str, row)) + "\n")
    
    return output.getvalue()

# Add the remaining imports at the top if not present
from datetime import datetime

def parse_svg_path(path_string, scale_factor):
    """
    Simple SVG path parser for basic polygon paths
    """
    points = []
    try:
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
    except:
        # If parsing fails, return empty list
        pass
    
    return points

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
        'browning_breakdown': browning_breakdown
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
    
    # Initialize session state with memory optimization
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'canvas_result' not in st.session_state:
        st.session_state.canvas_result = None
    
    # Cloud performance optimization: Clean up large objects periodically
    if 'cleanup_counter' not in st.session_state:
        st.session_state.cleanup_counter = 0
    st.session_state.cleanup_counter += 1
    
    # Clean up every 10 interactions to prevent memory buildup
    if st.session_state.cleanup_counter % 10 == 0:
        import gc
        gc.collect()
    
    # Sidebar - Streamlined
    with st.sidebar:
        st.header("🔧 Controls")
        
        # File Upload Section
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose corm image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of banana corm cross-section"
        )
        
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

            # Processing options in sidebar
            st.subheader("🔧 Image Processing")
            processing_option = st.selectbox(
                "Processing method:",
                ["Skip Processing", "Color Filtering", "AI Background Removal"],
                help="Choose how to prepare the image"
            )

            if st.button("📋 Prepare Image", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        img_array = np.array(original_image)
                        if processing_option == "Skip Processing":
                            processed_image = img_array
                        elif processing_option == "Color Filtering":
                            alpha_mask = np.ones(img_array.shape[:2], dtype=bool)
                            processed_image = filter_corm_colors(img_array, alpha_mask)
                        else:  # AI Background Removal
                            processed_image = remove_background_and_filter_colors(original_image)
                        
                        # Ensure processed image is not too large for memory
                        if processed_image.size > 2000000:  # ~2MP limit
                            st.warning("Large image detected. Optimizing for cloud performance...")
                            height, width = processed_image.shape[:2]
                            scale = np.sqrt(2000000 / processed_image.size)
                            new_height, new_width = int(height * scale), int(width * scale)
                            processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        st.session_state.processed_image = processed_image
                        st.success("✅ Image prepared!")
                    except Exception as e:
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
                st.subheader("🎯 Select Analysis Area")
                try:
                    canvas_result, scale_factor, shape_type = create_selection_canvas(
                        st.session_state.processed_image, 
                        canvas_key="corm_selection"
                    )
                except Exception as e:
                    st.error(f"Canvas error: {e}")
                    return

                # Quick Analyze button: available immediately after drawing (helps point-mode polygons)
                if canvas_result is not None:
                    st.info("💡 **Tip**: If polygon isn't detected, try clicking the radio button above (Point mode ↔ Freehand) to refresh the canvas data, then analyze.")
                    if st.button("🔬 Analyze Area", key="analyze_area_quick"):
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
                                    analysis_results = analyze_shape_region(
                                        st.session_state.processed_image,
                                        selection_mask,
                                        ignore_black=True
                                    )
                                    if analysis_results is not None:
                                        st.success("✅ Analysis complete!")
                                        st.session_state.analysis_results = analysis_results
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
                                else:
                                    st.warning("⚠️ Analysis failed. Please try selecting a different area.")
                    else:
                        st.info("🎯 Please draw a shape on the image above to select the analysis area.")
                if 'analysis_results' in st.session_state:
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    display_analysis_results(st.session_state.analysis_results, context="unified_results")
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
