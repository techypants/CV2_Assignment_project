import cv2
import numpy as np
import streamlit as st

# Streamlit app
st.title("Image Transformation App made in Python Computer Vision")
st.header("Upload or choose from system an Image and Apply Transformations ")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Transformation options
st.subheader("Select Transformations")
translate = st.checkbox("Translate the image")
rotate = st.checkbox("Rotate the image")
scale = st.checkbox("Scale the image")
shear = st.checkbox("Shear the image")
flip = st.checkbox("Flip the image")
crop = st.checkbox("Crop the image")
perspective = st.checkbox("Perspective of image")

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Error loading the image. Ensure it is a valid image file.")
    else:
        # Display the original image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        rows, cols, _ = image.shape
        processed_images = []

        # Apply transformations based on selected checkboxes
        if translate:
            tx, ty = 50, 100
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
            processed_images.append(("Translated", translated_image))

        if rotate:
            angle = 45
            center = (cols // 2, rows // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            processed_images.append(("Rotated", rotated_image))

        if scale:
            scale_x, scale_y = 1.5, 1.5
            scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
            processed_images.append(("Scaled", scaled_image))

        if shear:
            shear_x, shear_y = 0.2, 0.3
            shearing_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            sheared_image = cv2.warpAffine(image, shearing_matrix, (cols, rows))
            processed_images.append(("Sheared", sheared_image))

        if flip:
            flipped_image = cv2.flip(image, 1)
            processed_images.append(("Flipped", flipped_image))

        if crop:
            cropped_image = image[50:200, 100:300]
            processed_images.append(("Cropped", cropped_image))

        if perspective:
            pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
            pts2 = np.float32([[10, 100], [180, 50], [50, 250], [200, 220]])
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            perspective_image = cv2.warpPerspective(image, perspective_matrix, (cols, rows))
            processed_images.append(("Perspective", perspective_image))

        # Display processed images
        for name, img in processed_images:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=name, use_column_width=True)
