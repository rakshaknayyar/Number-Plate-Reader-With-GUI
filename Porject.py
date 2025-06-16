import streamlit as st
import cv2
import easyocr
import numpy as np
import imutils
from PIL import Image

reader = easyocr.Reader(['en'])

def gentle_correct_ocr_text(text):
    # Only correct very common, one-way OCR misreads (optional)
    corrections = {
        '8': '8',  # Do not convert to B
        '0': '0',  # Do not convert to O
        '1': '1',  # Do not convert to I
        '5': '5',  # Do not convert to S
        '6': '6',  # Do not convert to G
        '2': '2',  # Do not convert to Z
        '4': '4',  # Do not convert to A
        '7': '7',  # Do not convert to T
    }
    return ''.join([corrections.get(c, c) for c in text])

def enhance_image(img):
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    # Thresholding (binarization)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_plate_image(image):
    # Convert PIL Image to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return img, None, "No plate detected"

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], -1, 255, -1)
    (x, y) = np.where(mask==255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]

    cropped = enhance_image(cropped)

    results = reader.readtext(cropped)
    if results:
        print("Raw OCR results:", results)  # For debugging in terminal
        # Show all detected texts with confidence (for debugging)
        for r in results:
            print(f"Detected: {r[1]} (Confidence: {r[2]:.2f})")
        # Use the most confident result
        filtered = [r for r in results if r[2] > 0.3]
        if filtered:
            text = filtered[0][1]
            corrected_text = gentle_correct_ocr_text(text)
            print("Plate number:", corrected_text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (h, w) = img.shape[:2]
            text_size = cv2.getTextSize(corrected_text, font, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 40
            cv2.putText(img, corrected_text, (text_x, text_y), font, 0.8, (0, 0, 0), 2)
            # Red line from text to plate
            line_start = (text_x + text_size[0]//2, text_y + 10)
            line_end = (topy + (bottomy - topy)//2, topx)
            cv2.line(img, line_start, line_end, (0, 0, 255), 2)
            light_red = (102, 102, 255)
            cv2.rectangle(img, (topy, topx), (bottomy, bottomx), light_red, 2)
            return img, corrected_text, None
        else:
            return img, None, "No confident text detected"
    else:
        return img, None, "No text detected on plate"

# Streamlit app
st.title('License Plate Detection and OCR')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_img, plate_text, error = detect_plate_image(image)
    # Convert BGR to RGB for display
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    st.image(processed_img, caption='Processed Image', use_column_width=True)
    if plate_text:
        st.success(f'Detected Plate Number: {plate_text}')
    if error:
        st.warning(error)
else:
    st.info('Please upload an image file to detect license plate.')
