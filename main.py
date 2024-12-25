import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import watershed, clear_border
from skimage.morphology import binary_dilation
from scipy.ndimage import label, distance_transform_edt
import matplotlib.pyplot as plt
import json
from streamlit_lottie import st_lottie
import io
import pywt

# Set the page configuration
st.set_page_config(page_title="Photo Uploader and Enhancer", layout="wide")

# Function to handle navigation between pages
def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name

# Initialize session state for current page
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Welcome"

# Function definitions for image processing
def power_law_transformation(image, gamma):
    normalized = image / 255.0
    transformed = np.power(normalized, gamma) * 255
    return transformed.astype(np.uint8)

def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)

def calculate_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    else:  # Color image
        hist = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
    return hist

def plot_histograms(hist, title, color=None):
    fig, ax = plt.subplots()
    if color is None:  # Grayscale
        ax.plot(hist, color='black')
    else:  # Color
        for h, c in zip(hist, color):
            ax.plot(h, color=c)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def perform_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleared = clear_border(binary_image)
    bw_eroded = binary_dilation(cleared)
    D = distance_transform_edt(binary_image)
    D = -D
    markers, _ = label(bw_eroded)
    L = watershed(D, markers, mask=bw_eroded)
    L[~bw_eroded] = 0
    segmented_image = label2rgb(L, image=None, colors=('red', 'green', 'blue'), alpha=0.5, bg_label=0)
    return segmented_image
def threshold_segmentation(image, threshold_value):

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def wavelet_compression(image, wavelet="haar", threshold=0.1):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    coeffs = pywt.wavedec2(image, wavelet, level=2)
    coeffs_thresholded = [
        (pywt.threshold(c, threshold * np.max(c), mode='soft') if isinstance(c, np.ndarray) else c)
        for c in coeffs
    ]

    compressed_image = pywt.waverec2(coeffs_thresholded, wavelet)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    original_size = image.nbytes
    compressed_size = sum(c.nbytes for c in coeffs_thresholded if isinstance(c, np.ndarray))

    return compressed_image, original_size, compressed_size

def psycho_visual_reduction(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv_image)
    u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))
    v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))
    return y, u, v

def reconstruct_from_psychovisual(y, u, v):
    u = cv2.resize(u, (y.shape[1], y.shape[0]))
    v = cv2.resize(v, (y.shape[1], y.shape[0]))
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

def calculate_redundancy(original_size, compressed_size):
    return 1 - (compressed_size / original_size)

# Sidebar navigation
st.sidebar.title("Navigation")
page_options = ["Image Enhancement", "Filters", "Image Compression", "Segmentation"]
for page in page_options:
    if st.sidebar.button(page):
        navigate_to_page(page)

# Display content based on the selected page
if st.session_state["current_page"] == "Welcome":


    st.title("Welcome Image Enhancer ")
    st.write("This app allows you to upload, enhance, compress, and segment your photos.")
      # You can adjust the height and width as needed


else:
    # File uploader
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)

        if st.session_state["current_page"] == "Image Enhancement":
            st.title("Image Enhancement")
            enhancement_option = st.radio("Choose an enhancement technique:",
                                          ["Power Law Transformation", "Histogram Equalization"])

            if enhancement_option == "Power Law Transformation":
                gamma = st.slider("Select Gamma Value (0.1 - 3.0):", 0.1, 3.0, 1.0)
                enhanced_image = power_law_transformation(image_np, gamma)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_np, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)

            elif enhancement_option == "Histogram Equalization":
                enhanced_image = histogram_equalization(image_np)

                original_hist = calculate_histogram(image_np)
                enhanced_hist = calculate_histogram(enhanced_image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_np, caption="Original Image", use_container_width=True)
                    if len(original_hist) == 1:  # Grayscale
                        st.pyplot(plot_histograms(original_hist[0], "Original Histogram"))
                    else:  # Color
                        st.pyplot(plot_histograms(original_hist, "Original Histogram", color=['red', 'green', 'blue']))

                with col2:
                    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
                    if len(enhanced_hist) == 1:  # Grayscale
                        st.pyplot(plot_histograms(enhanced_hist[0], "Enhanced Histogram"))
                    else:  # Color
                        st.pyplot(plot_histograms(enhanced_hist, "Enhanced Histogram", color=['red', 'green', 'blue']))

            if st.button("Save Enhanced Image"):
                save_path = "enhanced_image.jpg"
                cv2.imwrite(save_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
                st.success(f"Enhanced image saved as {save_path}")

        if st.session_state["current_page"] == "Filters":
            st.title("Image Filters")
            st.write("Apply filters to your uploaded image.")

            if uploaded_file:
                # Display the original image
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image_np, caption="Original Image", use_container_width=True)

                # Filter selection and processing
                filter_type = st.radio("Select a filter:", ["Median Filter", "Gaussian Filter"])

                if filter_type == "Median Filter":
                    kernel_size = st.slider("Select Kernel Size (odd number):", 1, 15, 3, step=2)
                    filtered_image = apply_median_filter(image_np, kernel_size)

                elif filter_type == "Gaussian Filter":
                    kernel_size = st.slider("Select Kernel Size (odd number):", 1, 15, 3, step=2)
                    filtered_image = apply_gaussian_filter(image_np, kernel_size)

                # Display the filtered image
                with col2:
                    st.image(filtered_image, caption=f"Filtered Image ({filter_type})", use_container_width=True)

                # Option to download the filtered image
                filtered_pil_image = Image.fromarray(filtered_image)

                # Save to a buffer for download
                buf = io.BytesIO()
                filtered_pil_image.save(buf, format="JPEG")
                byte_data = buf.getvalue()

                st.download_button(
                    label="Download Filtered Image",
                    data=byte_data,
                    file_name=f"filtered_image_{filter_type.lower().replace(' ', '_')}.jpg",
                    mime="image/jpeg",
                )
            else:
                st.write("Please upload an image to apply filters.")



        if st.session_state["current_page"] == "Image Compression":

            st.title("Image Compression")

            if uploaded_file:

                image = Image.open(uploaded_file)

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image_np = np.array(image)

                compression_method = st.radio("Choose a compression method:",
                                              ["Wavelet Transform", "Psycho-Visual Redundancy"])

                if compression_method == "Wavelet Transform":
                    threshold = st.slider("Set Threshold (0.01 - 0.5)", 0.01, 0.5, 0.1)
                    compressed_image, original_size, compressed_size = wavelet_compression(image_np,
                                                                                           threshold=threshold)

                    redundancy = calculate_redundancy(original_size, compressed_size)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_np, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(compressed_image, caption="Compressed Image (Wavelet Transform)",
                                 use_container_width=True)

                    st.write(f"Data Redundancy: {redundancy:.2%}")

                    # Convert the compressed image to PIL format for downloading
                    compressed_pil_image = Image.fromarray(compressed_image)

                    # Save to a buffer for download
                    buf = io.BytesIO()
                    compressed_pil_image.save(buf, format="JPEG")
                    byte_data = buf.getvalue()

                    # Add the download button
                    st.download_button(
                        label="Download Compressed Image (Wavelet Transform)",
                        data=byte_data,
                        file_name="compressed_image_wavelet.jpg",
                        mime="image/jpeg",
                    )





                elif compression_method == "Psycho-Visual Redundancy":

                    y, u, v = psycho_visual_reduction(image_np)

                    compressed_image = reconstruct_from_psychovisual(y, u, v)

                    original_size = image_np.nbytes  # Original size in bytes

                    compressed_size = y.nbytes + u.nbytes + v.nbytes  # Compressed size in bytes

                    redundancy = calculate_redundancy(original_size, compressed_size)

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(image_np, caption="Original Image", use_container_width=True)

                    with col2:

                        st.image(compressed_image, caption="Compressed Image (Psycho-Visual)", use_container_width=True)

                    st.write(f"Data Redundancy: {redundancy:.2%}")

                    # Convert the compressed image to a PIL format for downloading

                    compressed_pil_image = Image.fromarray(compressed_image)

                    # Save to a buffer for download

                    buf = io.BytesIO()

                    compressed_pil_image.save(buf, format="JPEG")

                    byte_data = buf.getvalue()

                    # Add the download button

                    st.download_button(

                        label="Download Compressed Image (Psycho-Visual)",

                        data=byte_data,

                        file_name="compressed_image_psychovisual.jpg",

                        mime="image/jpeg",

                    )
        if st.session_state["current_page"] == "Segmentation":
            st.title("Image Segmentation")

            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Ensure grayscale images have a 3-channel format
                if len(image_np.shape) == 2:  # Grayscale image
                    image_np = np.stack([image_np] * 3, axis=-1)

                # Ensure the image is in uint8 format for compatibility
                if image_np.dtype != "uint8":
                    image_np = image_np.astype("uint8")

                # Segmentation method selection
                segmentation_method = st.radio("Choose a segmentation method:", ["Watershed", "Thresholding"])

                # Perform segmentation
                if segmentation_method == "Watershed":
                    segmented_image = perform_segmentation(image_np)
                elif segmentation_method == "Thresholding":
                    threshold_value = st.slider("Set Threshold Value (0-255):", 0, 255, 128)
                    segmented_image = threshold_segmentation(image_np, threshold_value)

                # Ensure the segmented image has the correct data type and shape
                if segmented_image.dtype != "uint8":
                    segmented_image = (segmented_image * 255).astype("uint8")  # Normalize if necessary
                if len(segmented_image.shape) == 2:  # Grayscale output
                    segmented_image = np.stack([segmented_image] * 3, axis=-1)  # Convert to RGB

                # Convert segmented image to a PIL format for saving
                segmented_image_pil = Image.fromarray(segmented_image)

                # Save the segmented image to a buffer for download
                buffer = io.BytesIO()
                segmented_image_pil.save(buffer, format="PNG")
                buffer.seek(0)

                # Display original and segmented images side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image_np, caption="Original Image", use_container_width=True)

                with col2:
                    st.image(segmented_image, caption=f"Segmented Image ({segmentation_method})",
                             use_container_width=True)

                # Provide a download button for the segmented image
                st.download_button(
                    label="Download Segmented Image",
                    data=buffer,
                    file_name="segmented_image.png",
                    mime="image/png",
                )
            else:
                st.warning("Please upload an image to perform segmentation.")


        # Main code


    # inform the user to upload an image


    else:

        st.write("Please upload a photo to see it displayed here.")