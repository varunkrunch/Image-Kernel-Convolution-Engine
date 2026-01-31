"""
Streamlit Frontend for Image Convolution Engine
Visual interface for applying convolution kernels to images.
"""

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import json
from typing import Optional


# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Image Convolution Engine",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stImage img {
        max-height: 320px;
        object-fit: contain;
        border-radius: 0.75rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    }
    </style>
""", unsafe_allow_html=True)


def fetch_available_kernels():
    """Fetch list of available kernels from API"""
    try:
        response = requests.get(f"{API_URL}/kernels/list")
        if response.status_code == 200:
            data = response.json()
            return data["kernels"]
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def apply_convolution_api(image_bytes, kernel_name=None, custom_kernel=None):
    """Call the API to apply convolution"""
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        data = {}
        
        if kernel_name:
            data["kernel_name"] = kernel_name
        if custom_kernel:
            data["custom_kernel"] = json.dumps(custom_kernel)
        
        response = requests.post(
            f"{API_URL}/transform/convolution",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None


def validate_custom_kernel_api(kernel_matrix):
    """Validate custom kernel via API"""
    try:
        response = requests.post(
            f"{API_URL}/kernels/custom",
            json={"matrix": kernel_matrix}
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("valid", False), data.get("message", "Unknown response"), data.get("kernel_size")
        else:
            return False, response.json().get("detail", "Unknown error"), None
    except Exception as e:
        return False, f"Error validating kernel: {e}", None


def display_kernel_matrix(kernel_matrix):
    """Display kernel matrix in a formatted way"""
    kernel_array = np.array(kernel_matrix)
    
    # Create formatted string
    matrix_str = "[\n"
    for row in kernel_array:
        formatted_row = [f"{val:7.3f}" for val in row]
        matrix_str += "  [" + ", ".join(formatted_row) + "]\n"
    matrix_str += "]"
    
    st.code(matrix_str, language="python")


def main():
    # Header
    st.markdown('<div class="main-header">🖼️ Image Convolution Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Apply mathematical kernels to transform your images</div>', unsafe_allow_html=True)

    # Persisted state for compact layout
    st.session_state.setdefault("prebuilt_original", None)
    st.session_state.setdefault("prebuilt_filename", None)
    st.session_state.setdefault("prebuilt_result", None)
    st.session_state.setdefault("prebuilt_kernel_name", None)
    st.session_state.setdefault("custom_original", None)
    st.session_state.setdefault("custom_filename", None)
    st.session_state.setdefault("custom_result", None)
    st.session_state.setdefault("custom_kernel", None)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["Pre-built Kernels", "Custom Kernel"],
            help="Choose between pre-built filters or create your own"
        )
        
        st.divider()
        
        with st.expander("ℹ️ About Convolution"):
            st.write(
                """
                **Convolution** applies a small matrix (kernel) to an image to create effects:
                - **Blur**: Smooths the image
                - **Sharpen**: Enhances edges
                - **Emboss**: Creates 3D effect
                - **Edge Detection**: Highlights boundaries
                """
            )
    
    # Main content
    if mode == "Pre-built Kernels":
        kernels = fetch_available_kernels()
        
        if kernels:
            col_select, col_upload, col_result = st.columns([1.1, 1.35, 1.35])

            with col_select:
                st.subheader("📋 Select Kernel")

                kernel_names = list(kernels.keys())
                selected_kernel = st.selectbox(
                    "Choose a filter",
                    kernel_names,
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="prebuilt_kernel_selector"
                )

                st.write("**Kernel Matrix:**")
                display_kernel_matrix(kernels[selected_kernel])

            with col_upload:
                st.subheader("📤 Upload & Apply")
                uploaded_file = st.file_uploader(
                    "Select an image",
                    type=["png", "jpg", "jpeg"],
                    help="Maximum file size: 2MB",
                    key="prebuilt_uploader"
                )

                if uploaded_file is not None:
                    image_bytes = uploaded_file.getvalue()
                    st.session_state["prebuilt_original"] = image_bytes
                    st.session_state["prebuilt_filename"] = uploaded_file.name

                if st.session_state["prebuilt_original"] is not None:
                    preview_image = Image.open(io.BytesIO(st.session_state["prebuilt_original"]))
                    st.image(
                        preview_image,
                        caption=st.session_state.get("prebuilt_filename", "Uploaded image")
                    )
                else:
                    st.info("Upload an image to preview it here.")

                if st.button("✨ Apply Convolution", type="primary", key="apply_prebuilt"):
                    if st.session_state["prebuilt_original"] is None:
                        st.warning("Please upload an image before applying the filter.")
                    else:
                        with st.spinner("Processing..."):
                            result_bytes = apply_convolution_api(
                                st.session_state["prebuilt_original"],
                                kernel_name=selected_kernel
                            )

                        if result_bytes:
                            st.session_state["prebuilt_result"] = result_bytes
                            st.session_state["prebuilt_kernel_name"] = selected_kernel
                            st.success("✅ Done!")

            with col_result:
                st.subheader("🔍 Result")

                if st.session_state["prebuilt_result"]:
                    result_image = Image.open(io.BytesIO(st.session_state["prebuilt_result"]))
                    kernel_title = (st.session_state.get("prebuilt_kernel_name") or "Filtered").replace("_", " ").title()
                    st.image(result_image, caption=f"{kernel_title}")

                    st.download_button(
                        label="⬇️ Download Result",
                        data=st.session_state["prebuilt_result"],
                        file_name=f"convolved_{st.session_state.get('prebuilt_kernel_name', 'image')}.png",
                        mime="image/png",
                        key="download_prebuilt"
                    )
                else:
                    st.info("Apply a kernel to see the transformed image here.")
    else:
        col_define, col_upload, col_result = st.columns([1.2, 1.3, 1.3])

        with col_define:
            st.subheader("🧪 Build Your Custom Kernel")

            default_kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
            kernel_text = st.text_area(
                "Enter kernel matrix (JSON format)",
                value=json.dumps(default_kernel, indent=2),
                help="Provide a square matrix with odd dimensions (e.g., 3x3, 5x5).",
                height=180,
                key="custom_kernel_text"
            )

            st.caption("Example: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]")

            kernel_matrix = None
            if kernel_text.strip():
                try:
                    kernel_matrix = json.loads(kernel_text)
                    if not isinstance(kernel_matrix, list):
                        raise ValueError
                except Exception:
                    st.error("Invalid kernel format. Please provide valid JSON (e.g., [[0,-1,0],[-1,5,-1],[0,-1,0]]).")
                    kernel_matrix = None

            if kernel_matrix is not None:
                is_valid, message, kernel_size = validate_custom_kernel_api(kernel_matrix)
                if is_valid:
                    st.session_state["custom_kernel"] = kernel_matrix
                    st.success(f"Kernel valid (size: {kernel_size[0]}x{kernel_size[1]})")
                    display_kernel_matrix(kernel_matrix)
                else:
                    st.session_state["custom_kernel"] = None
                    st.error(message)
            else:
                st.session_state["custom_kernel"] = None

        with col_upload:
            st.subheader("📤 Upload & Apply")
            uploaded_file = st.file_uploader(
                "Select an image",
                type=["png", "jpg", "jpeg"],
                help="Maximum file size: 2MB",
                key="custom_uploader"
            )

            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                st.session_state["custom_original"] = image_bytes
                st.session_state["custom_filename"] = uploaded_file.name

            if st.session_state["custom_original"] is not None:
                preview_image = Image.open(io.BytesIO(st.session_state["custom_original"]))
                st.image(
                    preview_image,
                    caption=st.session_state.get("custom_filename", "Uploaded image")
                )
            else:
                st.info("Upload an image to preview it here.")

            apply_disabled = st.session_state["custom_original"] is None or st.session_state["custom_kernel"] is None

            if st.button(
                "✨ Apply Custom Convolution",
                type="primary",
                key="apply_custom",
                disabled=apply_disabled
            ):
                with st.spinner("Processing with custom kernel..."):
                    result_bytes = apply_convolution_api(
                        st.session_state["custom_original"],
                        custom_kernel=st.session_state["custom_kernel"]
                    )

                if result_bytes:
                    st.session_state["custom_result"] = result_bytes
                    st.success("✅ Done!")

        with col_result:
            st.subheader("🔍 Result")

            if st.session_state["custom_result"]:
                result_image = Image.open(io.BytesIO(st.session_state["custom_result"]))
                st.image(result_image, caption="Custom Kernel Result")

                st.download_button(
                    label="⬇️ Download Result",
                    data=st.session_state["custom_result"],
                    file_name="convolved_custom.png",
                    mime="image/png",
                    key="download_custom"
                )
            else:
                st.info("Define a valid kernel and apply it to see the transformed image here.")


if __name__ == "__main__":
    main()