import streamlit as st
import io
import os
from PIL import Image, ImageDraw
from core.bg_remover import (
    remove_background,
    extract_garment_design,
    calculate_regions_bbox,
    draw_region_preview,
    normalize_region
)
from core.enhancer import enhance_raster_image, AVAILABLE_ENGINES, is_model_downloaded
from core.vectorizer import vectorize_image
from core.exporter import export_print_package

st.set_page_config(
    page_title="T-Shirt Vector Design Maker",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge-local-ai {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .highlight-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👕 T-Shirt Design Maker & Vectorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Convert low-res client PNG & JPG files into crisp, scalable vector graphics (SVG, PDF, 300 DPI PNG) with local AI super-resolution and quote/text preservation for direct garment printing.</div>', unsafe_allow_html=True)

# Initialize session state for multi-region preservation
if "preserved_regions" not in st.session_state:
    st.session_state.preserved_regions = []

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("⚙️ Print & Vector Settings")
    
    st.subheader("1. Background & Garment Isolation")
    bg_mode = st.radio(
        "Isolation Mode",
        options=[
            "Auto-Detect Garment Fabric (Best for T-Shirt Mockups & Photos)",
            "AI Subject Segmentation (rembg)",
            "Keep Background (No Removal)"
        ],
        index=0,
        help="Use 'Auto-Detect Garment Fabric' when your image is a photo of a t-shirt or mockup—it keys out the garment cloth so quotes and graphics remain. Use 'AI Subject Segmentation' for standalone graphics."
    )
    remove_bg = bg_mode != "Keep Background (No Removal)"
    
    alpha_threshold = 25
    if remove_bg:
        alpha_threshold = st.slider(
            "Alpha Edge Cutoff", 5, 80, 25, 
            help="Eliminates faint semi-transparent edge halos to prevent DTF white ink bleeding."
        )

    fabric_tolerance = 30
    if "Garment Fabric" in bg_mode:
        fabric_tolerance = st.slider(
            "Fabric Color Tolerance", 10, 80, 30,
            help="Threshold for identifying and removing the t-shirt cloth color behind the design."
        )

    preserve_quote_mode = st.checkbox(
        "Interactive Area Selection & Cropping Tool",
        value=True,
        help="Enable interactive selection box to crop directly to the design or quote area on the t-shirt."
    )
    
    clean_text_bg = True
    text_threshold = 25
    if preserve_quote_mode and "rembg" in bg_mode:
        clean_text_bg = st.checkbox(
            "Auto-Clean Text Background",
            value=True,
            help="Automatically keys out the background behind letters inside the selected box so text is transparently integrated without a solid rectangular block."
        )
        text_threshold = st.slider(
            "Text Stroke Sensitivity",
            5, 80, 25,
            help="Sensitivity for distinguishing letter strokes from the background inside the selected region."
        )
    
    st.markdown("---")
    st.subheader("2. Local AI Super-Resolution")
    engine_options = {
        "realesrgan_anime": "Real-ESRGAN Anime 6B (Vector & Line Art - Recommended)",
        "realesrgan_general": "Real-ESRGAN x4plus (Photorealistic Art)",
        "lanczos": "Classical Lanczos + Bilateral (Fast / Non-AI)"
    }
    selected_engine_key = st.selectbox(
        "Enhancement Engine",
        options=list(engine_options.keys()),
        format_func=lambda k: engine_options[k],
        index=0,
        help="Local AI neural models remove pixelation and staircase artifacts from low-res graphics before vector tracing."
    )
    
    engine_meta = AVAILABLE_ENGINES.get(selected_engine_key, {})
    if engine_meta.get("is_ai"):
        downloaded = is_model_downloaded(selected_engine_key)
        status_text = "Ready (Cached locally)" if downloaded else "Will auto-download on first run (~18MB)"
        st.caption(f"🧠 **Local ONNX Neural Engine**: {status_text}")
    else:
        st.caption("⚡ **Classical Interpolation**: Fast non-neural scaling with bilateral edge-preserving smoothing.")

    upscale_factor = st.slider(
        "Resolution Multiplier",
        1.0, 4.0, 2.0, 0.5,
        help="Target upscale factor. Real-ESRGAN applies 4x neural reconstruction with supersampled anti-aliasing."
    )

    with st.expander("Fine-Tune AI Enhancement"):
        denoise_strength = st.slider("Extra Denoise Strength", 0, 10, 0, help="Optional bilateral filter for very noisy JPEG scans (0 = disabled).")
        sharpen_amount = st.slider("Edge Sharpen Multiplier", 0.8, 2.0, 1.0, 0.1, help="Sharpens boundaries prior to vector path fitting.")
        contrast_boost = st.slider("Contrast Adjustment", 0.8, 1.5, 1.0, 0.05, help="Increases separation between light and dark print tones.")

    st.markdown("---")
    st.subheader("3. Printing Mode")
    print_mode = st.radio(
        "Target Output",
        options=["Full Color (DTF / DTG / Digital)", "Monochrome (Vinyl / Screen Print)"],
        index=0,
        help="Full color uses stacked color layers; Monochrome produces high-contrast 1-bit outlines for plotters and single-color silkscreen."
    )
    colormode = "color" if "Full Color" in print_mode else "binary"

    st.markdown("---")
    st.subheader("4. Garment Print Dimensions")
    size_preset = st.selectbox(
        "Size Preset",
        ["Full Front (12\" × 16\")", "Standard Chest (10\" × 12\")", "Pocket / Logo (4\" × 4\")", "Oversized (14\" × 18\")", "Custom..."]
    )
    
    if size_preset == "Full Front (12\" × 16\")":
        target_w, target_h = 12.0, 16.0
    elif size_preset == "Standard Chest (10\" × 12\")":
        target_w, target_h = 10.0, 12.0
    elif size_preset == "Pocket / Logo (4\" × 4\")":
        target_w, target_h = 4.0, 4.0
    elif size_preset == "Oversized (14\" × 18\")":
        target_w, target_h = 14.0, 18.0
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_w = st.number_input("Width (inches)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
        with c2:
            target_h = st.number_input("Height (inches)", min_value=1.0, max_value=40.0, value=16.0, step=0.5)

    dpi = st.number_input("Target DPI", min_value=150, max_value=600, value=300, step=50, help="Industry standard for garment printing is 300 DPI.")

    st.markdown("---")
    st.subheader("5. Vector Path Settings")
    with st.expander("Vector Tuning"):
        speckle_filter = st.slider("Speckle Noise Filter", 1, 25, 4, help="Filters out small noisy pixel clusters.")
        corner_threshold = st.slider("Curve Smoothness", 20, 100, 60, help="Higher values create smoother curves; lower preserves sharp corners.")
        color_precision = st.slider("Color Precision", 3, 8, 6, help="Color depth clustering (lower values group similar colors into single print tones).")

# ----------------- MAIN AREA -----------------
def generate_sample_badge():
    """Create a sample badge with both an emblem and a quote slogan to demonstrate text preservation."""
    img = Image.new("RGBA", (400, 460), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Circle emblem
    draw.ellipse([50, 30, 350, 330], fill="#1E3A8A", outline="#F59E0B", width=8)
    draw.ellipse([80, 60, 320, 300], fill="#FFFFFF", outline="#1E3A8A", width=4)
    # Shapes inside badge
    draw.rectangle([130, 150, 270, 210], fill="#EF4444")
    draw.polygon([(200, 85), (235, 140), (165, 140)], fill="#F59E0B")
    
    # Typography quote below badge
    draw.text((105, 360), "VINTAGE MOTOR CLUB", fill="#1E3A8A")
    draw.text((135, 395), "ESTABLISHED 1984", fill="#4B5563")
    draw.text((120, 425), "• ORIGINAL HERITAGE •", fill="#F59E0B")
    return img

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

uploaded_file = st.file_uploader("Upload client image (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

col_test1, col_test2 = st.columns([1, 4])
with col_test1:
    if st.button("🧪 Use Sample Graphic with Quote", use_container_width=True):
        st.session_state.use_sample = True

input_image = None
current_image_key = None

if uploaded_file is not None:
    st.session_state.use_sample = False
    input_image = Image.open(uploaded_file)
    current_image_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
elif st.session_state.use_sample:
    input_image = generate_sample_badge()
    current_image_key = "sample_badge"

# Automatically reset preserved regions if image changes to prevent cross-image coordinate contamination
if current_image_key:
    if st.session_state.get("active_image_key") != current_image_key:
        st.session_state.preserved_regions = []
        st.session_state.active_image_key = current_image_key

if input_image is not None:
    st.markdown("---")
    
    # Active selection box state
    active_selection_box = None
    regions_to_use = []
    crop_scope = "🖼️ Keep Full Image Canvas (Process Whole Image)"

    # ----------------- INTERACTIVE QUOTE / TEXT SELECTION TOOL -----------------
    if remove_bg and preserve_quote_mode:
        st.subheader("✂️ Interactive Design & Text Area Selector")
        st.caption("Select the print or quote area on the t-shirt to extract ONLY that design (discarding the t-shirt cloth, sleeves, and photo background).")

        cropper_col, preview_col = st.columns([1.2, 1])

        with cropper_col:
            st.markdown("**1. Move & Resize Selection Box Over the Design / Quote**")
            
            # Smart default coords: place initial box around the center 50% of the image
            def_y1 = int(round(input_image.height * 0.25))
            def_y2 = int(round(input_image.height * 0.75))
            def_x1 = int(round(input_image.width * 0.20))
            def_x2 = int(round(input_image.width * 0.80))
            default_coords = (def_x1, def_x2, def_y1, def_y2)

            from streamlit_cropper import st_cropper
            cropped_img, active_selection_box = st_cropper(
                input_image,
                realtime_update=True,
                box_color="#06B6D4",
                aspect_ratio=None,
                default_coords=default_coords,
                return_type="both",
                key=f"quote_cropper_{current_image_key}"
            )

        with preview_col:
            st.markdown("**2. Selection Area & Output Scope**")
            if cropped_img is not None:
                st.image(cropped_img, caption=f"Active Selection Area ({cropped_img.width} × {cropped_img.height} px)", use_container_width=True)

            crop_scope = st.radio(
                "🎯 Output Scope",
                options=[
                    "🎯 Crop to Selected Design Area (Extract Artwork Only)",
                    "🖼️ Keep Full Image Canvas (Process Whole Image)"
                ],
                index=0,
                help="When 'Crop to Selected Design Area' is chosen, the tool extracts ONLY the design inside your selection (discarding the t-shirt garment, sleeves, and photo background). This is ideal when working with photos of t-shirts!"
            )

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("➕ Add to Preserved List", use_container_width=True, help="Save this selection so you can select additional quotes or areas."):
                    if active_selection_box:
                        st.session_state.preserved_regions.append(dict(active_selection_box))
                        st.rerun()
            with b_col2:
                if st.button("🗑️ Clear Preserved List", use_container_width=True):
                    st.session_state.preserved_regions = []
                    st.rerun()

            # Check if active box is distinct from any saved region
            active_coords = normalize_region(active_selection_box, input_image.width, input_image.height) if active_selection_box else None
            is_active_distinct = False
            regions_to_use = list(st.session_state.preserved_regions)

            if active_coords:
                is_duplicate = False
                for r in st.session_state.preserved_regions:
                    rc = normalize_region(r, input_image.width, input_image.height)
                    if rc and abs(rc[0] - active_coords[0]) <= 5 and abs(rc[1] - active_coords[1]) <= 5 and abs(rc[2] - active_coords[2]) <= 5 and abs(rc[3] - active_coords[3]) <= 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    is_active_distinct = True
                    regions_to_use.append(dict(active_selection_box))

            # Display saved regions list
            if st.session_state.preserved_regions:
                st.markdown(f"**📌 Preserved Regions ({len(st.session_state.preserved_regions)} Saved):**")
                for i, r in enumerate(st.session_state.preserved_regions):
                    r_c1, r_c2 = st.columns([3, 1])
                    with r_c1:
                        st.write(f"• **Region #{i+1}**: `X:{r.get('left',0)}, Y:{r.get('top',0)}, W:{r.get('width',0)}, H:{r.get('height',0)}`")
                    with r_c2:
                        if st.button("❌", key=f"del_region_{i}", help=f"Remove Region #{i+1}"):
                            st.session_state.preserved_regions.pop(i)
                            st.rerun()
                if is_active_distinct:
                    st.caption("💡 *Current active box (cyan) is also included automatically.*")
            else:
                st.info("💡 **Auto-Extraction Active**: The active bounding box above will be extracted and vectorized when you generate the print package.")

        st.markdown("---")

    # ----------------- PREVIEW & PRINT SPECS -----------------
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📷 Design Overview & Selection Map")
        # Show annotated preview if regions are selected
        if preserve_quote_mode and remove_bg and (regions_to_use or active_selection_box):
            preview_img = draw_region_preview(
                input_image,
                regions=st.session_state.preserved_regions,
                active_box=active_selection_box if (is_active_distinct or not st.session_state.preserved_regions) else None
            )
            st.image(preview_img, caption=f"Highlighted Design Selection ({input_image.width} × {input_image.height} px)", use_container_width=True)
        else:
            st.image(input_image, caption=f"Original Size: {input_image.width} × {input_image.height} px ({input_image.mode})", use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Print Target Info")
        target_px_w = int(round(target_w * dpi))
        target_px_h = int(round(target_h * dpi))
        engine_label = engine_options.get(selected_engine_key, selected_engine_key)
        
        preservation_status = f"Active ({len(regions_to_use)} region(s) selected)" if (preserve_quote_mode and remove_bg and regions_to_use) else "Disabled"
        
        st.info(f"""
        - **Target Physical Size:** {target_w:.1f}" × {target_h:.1f}"
        - **Print Resolution:** {dpi} DPI ({target_px_w} × {target_px_h} px)
        - **Output Scope:** {crop_scope}
        - **AI Super-Resolution:** {engine_label} ({upscale_factor:.1f}x)
        - **Background Isolation:** {bg_mode}
        - **Area Selector:** {preservation_status}
        - **Output Color Mode:** {'Full Color Stacked SVG' if colormode == 'color' else 'Monochrome Outlines'}
        """)

        process_button = st.button("🚀 Generate Print-Ready Vector Package", type="primary", use_container_width=True)

    if process_button:
        with st.spinner("Processing image (Extracting design ➔ AI Super-Resolution ➔ Vector Tracing ➔ 300 DPI Export)..."):
            # Determine unified bounding box if cropping to selected area
            should_crop_to_selection = preserve_quote_mode and (crop_scope.startswith("🎯"))
            selected_bbox = None
            if should_crop_to_selection:
                selected_bbox = calculate_regions_bbox(regions_to_use, input_image.width, input_image.height, padding=8)
                if selected_bbox is None and active_selection_box:
                    selected_bbox = normalize_region(active_selection_box, input_image.width, input_image.height)

            if should_crop_to_selection and selected_bbox is not None:
                # 1. Extract only the design from the garment mockup / crop area
                if "Garment Fabric" in bg_mode:
                    processed_raster = extract_garment_design(
                        input_image,
                        bbox=selected_bbox,
                        method="fabric_key",
                        tolerance=fabric_tolerance,
                        alpha_threshold=alpha_threshold
                    )
                elif "rembg" in bg_mode:
                    processed_raster = extract_garment_design(
                        input_image,
                        bbox=selected_bbox,
                        method="rembg",
                        alpha_threshold=alpha_threshold
                    )
                else:
                    processed_raster = extract_garment_design(
                        input_image,
                        bbox=selected_bbox,
                        method="none"
                    )
            else:
                # Process full image
                if "Garment Fabric" in bg_mode:
                    processed_raster = extract_garment_design(
                        input_image,
                        bbox=None,
                        method="fabric_key",
                        tolerance=fabric_tolerance,
                        alpha_threshold=alpha_threshold
                    )
                elif "rembg" in bg_mode:
                    processed_raster = remove_background(
                        input_image,
                        alpha_threshold=alpha_threshold,
                        preserve_regions=regions_to_use if preserve_quote_mode else None,
                        clean_text_background=clean_text_bg,
                        text_threshold=text_threshold
                    )
                else:
                    processed_raster = input_image.copy()

            # 2. Local AI Super-Resolution & Enhancement
            enhanced_raster = enhance_raster_image(
                processed_raster,
                engine=selected_engine_key,
                upscale_factor=upscale_factor,
                denoise_strength=denoise_strength,
                sharpen_amount=sharpen_amount,
                contrast_boost=contrast_boost
            )

            # 3. Vectorization with VTracer
            svg_content = vectorize_image(
                enhanced_raster,
                colormode=colormode,
                hierarchical="stacked",
                filter_speckle=speckle_filter,
                corner_threshold=corner_threshold,
                color_precision=color_precision
            )

            # 4. Generate Export Package (SVG, 300 DPI Master PNG, Vector PDF)
            package = export_print_package(
                svg_content=svg_content,
                master_image=enhanced_raster,
                target_width_inches=target_w,
                target_height_inches=target_h,
                dpi=dpi
            )

            st.success("🎉 Vectorization and print preparation completed successfully! Both illustration and quote typography preserved.")

            # Display Tabs
            tab_vector, tab_comparison, tab_download = st.tabs([
                "✨ Vector Preview (SVG)",
                "🔍 Preserved Master vs Original",
                "📥 Download Print Package"
            ])

            with tab_vector:
                st.subheader("Scalable Vector Output (Artwork + Preserved Quote)")
                st.components.v1.html(
                    f"""
                    <div style="display:flex; justify-content:center; align-items:center; background:#F9FAFB; padding:20px; border-radius:8px; border:1px solid #E5E7EB; min-height:400px;">
                        <div style="max-width:100%; max-height:500px;">
                            {svg_content}
                        </div>
                    </div>
                    """,
                    height=520
                )
                st.caption("Zoom in as much as you want — both the graphic artwork and quote text are scalable vector curves.")

            with tab_comparison:
                st.subheader("Comparison: Original vs Cleaned Transparent Master")
                c_orig, c_enh = st.columns(2)
                with c_orig:
                    st.markdown("**Original Upload (With Background)**")
                    st.image(input_image, caption=f"{input_image.width} × {input_image.height} px", use_container_width=True)
                with c_enh:
                    st.markdown(f"**Cleaned Master with Preserved Quote ({engine_meta.get('name', 'Enhanced')})**")
                    st.image(enhanced_raster, caption=f"{enhanced_raster.width} × {enhanced_raster.height} px (Graphic + Quote Preserved on Transparent Canvas)", use_container_width=True)

            with tab_download:
                st.subheader("📦 Download Files for Your Printer")
                st.markdown("Select any format below to send straight to your printer, cutter, or graphics software:")

                dcol1, dcol2, dcol3 = st.columns(3)

                with dcol1:
                    st.markdown("**1. Scalable Vector (SVG)**")
                    st.caption("For Adobe Illustrator, CorelDRAW, Inkscape, or vinyl cutter software.")
                    st.download_button(
                        label="📥 Download SVG File",
                        data=package["svg"],
                        file_name="tshirt_design_vector.svg",
                        mime="image/svg+xml",
                        use_container_width=True
                    )

                with dcol2:
                    st.markdown(f"**2. Print-Ready PDF ({target_w:.1f}\" × {target_h:.1f}\")**")
                    st.caption("Calibrated at 1:1 physical garment inches for professional print houses.")
                    st.download_button(
                        label="📥 Download Vector PDF",
                        data=package["pdf"],
                        file_name="tshirt_print_ready.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                with dcol3:
                    st.markdown(f"**3. 300 DPI Master PNG**")
                    st.caption(f"{target_px_w} × {target_px_h} px with clean transparency for DTF/DTG RIP software.")
                    st.download_button(
                        label="📥 Download 300 DPI PNG",
                        data=package["png"],
                        file_name="tshirt_master_300dpi.png",
                        mime="image/png",
                        use_container_width=True
                    )

else:
    st.info("👆 Upload an image or click **'🧪 Use Sample Graphic with Quote'** above to get started.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>T-Shirt Design Maker • Powered by Real-ESRGAN AI, VTracer & Rembg</div>", unsafe_allow_html=True)
