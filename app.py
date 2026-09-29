import streamlit as st
import cv2
import numpy as np
import os


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="centered"
)


# ============================================
# TITLE
# ============================================

st.title("🎨 AI Image Colorization")

st.write(
    "Upload any black-and-white image and "
    "convert it into a color image using AI."
)


# ============================================
# CREATE OUTPUT FOLDER
# ============================================

os.makedirs("output", exist_ok=True)


# ============================================
# LOAD AI MODEL
# ============================================

@st.cache_resource
def load_model():

    net = cv2.dnn.readNetFromCaffe(
        "models/colorization_deploy_v2.prototxt",
        "models/colorization_release_v2.caffemodel"
    )

    pts = np.load(
        "models/pts_in_hull.npy"
    )

    pts = pts.transpose().reshape(
        2,
        313,
        1,
        1
    )

    net.getLayer(
        net.getLayerId("class8_ab")
    ).blobs = [
        pts.astype(np.float32)
    ]

    net.getLayer(
        net.getLayerId("conv8_313_rh")
    ).blobs = [
        np.full(
            [1, 313],
            2.606,
            dtype=np.float32
        )
    ]

    return net


# ============================================
# COLORIZATION FUNCTION
# ============================================

def colorize_image(image, net):

    # BGR → RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float
    image_float = (
        image_rgb.astype(np.float32) / 255.0
    )

    # RGB → LAB
    lab = cv2.cvtColor(
        image_float,
        cv2.COLOR_RGB2LAB
    )

    # Extract L channel
    l_channel = lab[:, :, 0]

    # Resize for AI model
    resized_l = cv2.resize(
        l_channel,
        (224, 224)
    )

    # Normalize
    resized_l = resized_l - 50

    # Create blob
    blob = cv2.dnn.blobFromImage(
        resized_l
    )

    # Send to neural network
    net.setInput(blob)

    # Predict AB channels
    ab = net.forward()[0]

    ab = ab.transpose(
        (1, 2, 0)
    )

    # Resize AB to original image size
    ab = cv2.resize(
        ab,
        (image.shape[1], image.shape[0])
    )

    # Combine L + AB
    lab_output = np.concatenate(
        (
            l_channel[:, :, np.newaxis],
            ab
        ),
        axis=2
    )

    # LAB → RGB
    colorized = cv2.cvtColor(
        lab_output.astype(np.float32),
        cv2.COLOR_LAB2RGB
    )

    # Convert to 8-bit
    colorized = np.clip(
        colorized * 255,
        0,
        255
    ).astype(np.uint8)

    return colorized


# ============================================
# FIND NEXT OUTPUT NUMBER
# ============================================

def get_next_output_number():

    existing_files = os.listdir("output")

    numbers = []

    for filename in existing_files:

        if (
            filename.startswith("colorized_")
            and filename.endswith(".jpg")
        ):

            number_text = (
                filename
                .replace("colorized_", "")
                .replace(".jpg", "")
            )

            try:

                number = int(number_text)

                numbers.append(number)

            except ValueError:

                pass

    if len(numbers) == 0:

        return 1

    return max(numbers) + 1


# ============================================
# IMAGE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "📤 Upload a black-and-white image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================
# PROCESS IMAGE
# ============================================

if uploaded_file is not None:

    # Read uploaded file
    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    # Decode image
    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Check image
    if image is None:

        st.error(
            "❌ Unable to read the image."
        )

    else:

        st.success(
            f"Image uploaded: {uploaded_file.name}"
        )

        # ====================================
        # SHOW UPLOADED IMAGE
        # ====================================

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width="content"
        )


        # ====================================
        # COLORIZE BUTTON
        # ====================================

        if st.button(
            "🎨 Colorize Image",
            type="primary"
        ):

            with st.spinner(
                "🤖 AI is colorizing your image..."
            ):

                try:

                    # Load AI model
                    net = load_model()

                    # Colorize image
                    result = colorize_image(
                        image,
                        net
                    )

                    # =================================
                    # CREATE NUMBERED OUTPUT
                    # =================================

                    next_number = (
                        get_next_output_number()
                    )

                    output_filename = (
                        f"colorized_{next_number}.jpg"
                    )

                    output_path = os.path.join(
                        "output",
                        output_filename
                    )


                    # =================================
                    # RGB → BGR
                    # =================================

                    result_bgr = cv2.cvtColor(
                        result,
                        cv2.COLOR_RGB2BGR
                    )


                    # =================================
                    # SAVE IMAGE
                    # =================================

                    success = cv2.imwrite(
                        output_path,
                        result_bgr
                    )

                    if not success:

                        raise Exception(
                            "Could not save the output image."
                        )


                    # =================================
                    # SHOW RESULT
                    # =================================

                    st.success(
                        "✅ Colorization completed!"
                    )

                    st.subheader(
                        "🌈 Colorized Image"
                    )

                    st.image(
                        result,
                        width="content"
                    )


                    # =================================
                    # DOWNLOAD BUTTON
                    # =================================

                    with open(
                        output_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="⬇️ Download Colorized Image",
                            data=file.read(),
                            file_name=output_filename,
                            mime="image/jpeg"
                        )


                    # =================================
                    # OUTPUT INFORMATION
                    # =================================

                    st.info(
                        f"💾 Saved to: output/{output_filename}"
                    )


                except Exception as e:

                    st.error(
                        "❌ Error during colorization"
                    )

                    st.exception(e)