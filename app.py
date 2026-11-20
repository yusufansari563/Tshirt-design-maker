import streamlit as st
import io
import os
from PIL import Image, ImageDraw
from core.bg_remover import remove_background
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👕 T-Shirt Design Maker & Vectorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Convert low-res client PNG & JPG files into crisp, scalable vector graphics (SVG, PDF, 300 DPI PNG) with local AI super-resolution for direct garment printing.</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("⚙️ Print & Vector Settings")
    
    st.subheader("1. Background Isolation")
    remove_bg = st.checkbox("Remove Background", value=True, help="Automatically cuts out white or noisy backgrounds using AI segmentation.")
    alpha_threshold = 25
    if remove_bg:
        alpha_threshold = st.slider("Alpha Edge Cutoff", 5, 80, 25, 
                                    help="Eliminates faint semi-transparent edge halos to prevent DTF white ink bleeding.")
    
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
    """Create a sample low-resolution badge for immediate testing."""
    img = Image.new("RGBA", (350, 350), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Circle badge
    draw.ellipse([25, 25, 325, 325], fill="#1E3A8A", outline="#F59E0B", width=8)
    draw.ellipse([50, 50, 300, 300], fill="#FFFFFF", outline="#1E3A8A", width=4)
    # Shapes & Text
    draw.rectangle([100, 140, 250, 210], fill="#EF4444")
    draw.polygon([(175, 70), (210, 130), (140, 130)], fill="#F59E0B")
    return img

uploaded_file = st.file_uploader("Upload client image (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

col_test1, col_test2 = st.columns([1, 4])
with col_test1:
    use_sample = st.button("🧪 Use Sample Low-Res Graphic", use_container_width=True)

input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif use_sample:
    input_image = generate_sample_badge()

if input_image is not None:
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📷 Original Upload")
        st.image(input_image, caption=f"Original Size: {input_image.width} × {input_image.height} px ({input_image.mode})", use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Print Target Info")
        target_px_w = int(round(target_w * dpi))
        target_px_h = int(round(target_h * dpi))
        engine_label = engine_options.get(selected_engine_key, selected_engine_key)
        st.info(f"""
        - **Target Physical Size:** {target_w:.1f}" × {target_h:.1f}"
        - **Print Resolution:** {dpi} DPI ({target_px_w} × {target_px_h} px)
        - **AI Super-Resolution:** {engine_label} ({upscale_factor:.1f}x)
        - **Background Removal:** {'Active' if remove_bg else 'Disabled (Keep Original)'}
        - **Output Color Mode:** {'Full Color Stacked SVG' if colormode == 'color' else 'Monochrome Outlines'}
        """)

        process_button = st.button("🚀 Generate Print-Ready Vector Package", type="primary", use_container_width=True)

    if process_button:
        with st.spinner("Processing image (Background removal ➔ AI De-pixelation ➔ Vector Tracing ➔ 300 DPI Export)..."):
            # 1. Background removal
            if remove_bg:
                processed_raster = remove_background(input_image, alpha_threshold=alpha_threshold)
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

            st.success("🎉 Vectorization and print preparation completed successfully!")

            # Display Tabs
            tab_vector, tab_comparison, tab_download = st.tabs([
                "✨ Vector Preview (SVG)",
                "🔍 AI Enhanced Master vs Original",
                "📥 Download Print Package"
            ])

            with tab_vector:
                st.subheader("Scalable Vector Output")
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
                st.caption("Zoom in as much as you want — vector edges never pixelate or blur.")

            with tab_comparison:
                st.subheader("AI Super-Resolution Comparison")
                c_orig, c_enh = st.columns(2)
                with c_orig:
                    st.markdown("**Original Upload (Low Resolution)**")
                    st.image(input_image, caption=f"{input_image.width} × {input_image.height} px", use_container_width=True)
                with c_enh:
                    st.markdown(f"**AI Super-Resolved Master ({engine_meta.get('name', 'Enhanced')})**")
                    st.image(enhanced_raster, caption=f"{enhanced_raster.width} × {enhanced_raster.height} px (Smooth Anti-Aliased Curves)", use_container_width=True)

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
    st.info("👆 Upload an image or click **'🧪 Use Sample Low-Res Graphic'** above to get started.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>T-Shirt Design Maker • Powered by Real-ESRGAN AI, VTracer & Rembg</div>", unsafe_allow_html=True)
