import os
import cv2
import numpy as np
from deepface import DeepFace
import google.generativeai as genai
from PIL import Image, ImageEnhance
import pytesseract
import re
import streamlit as st
import tempfile
import mediapipe as mp
from scipy import ndimage
import concurrent.futures
import warnings
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
warnings.filterwarnings('ignore')  # Suppress warnings

# Configure Tesseract OCR path - Update this path to match your Tesseract installation
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    st.warning("Warning: Tesseract OCR not found at the specified path. Please install Tesseract OCR and update the path.")

# Configure Google Gemini API
GOOGLE_API_KEY = "AIzaSyCqPy8J7K3Orccta6-JSo16GqqeKoYMzaQ"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_id(image_path):
    """Extract text from ID card using OCR with optimized preprocessing."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Failed to read image"}

        # Optimize image size for faster processing
        max_dimension = 1500  # Reduced from original size
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply faster preprocessing
        # Use simpler thresholding instead of adaptive thresholding
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply bilateral filter with reduced parameters for speed
        denoised = cv2.bilateralFilter(threshold, 5, 50, 50)
        
        # Skip CLAHE for faster processing
        # Convert to PIL Image for Tesseract
        pil_image = Image.fromarray(denoised)
        
        # Configure Tesseract for faster processing
        custom_config = r'--oem 1 --psm 6'  # Using LSTM only mode for better speed
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # Split text into lines and clean
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        extracted_info = {
            'college': '',
            'name': '',
            'program': '',
            'branch': '',
            'roll_number': '',
            'year': '',
            'address': ''
        }
        
        # Optimize pattern matching
        college_pattern = re.compile(r'SRI|AP|SRM', re.IGNORECASE)
        name_pattern = re.compile(r'^[A-Z]{3,}\s[A-Z]{3,}$')
        roll_pattern = re.compile(r'AP\d{11}')
        year_pattern = re.compile(r'20\d{2}-20\d{2}')
        branch_pattern = re.compile(r'(CSE|ECE|EEE)', re.IGNORECASE)
        
        # Process each line with optimized patterns
        for i, line in enumerate(lines):
            line = re.sub(r'[^\w\s-]', '', line).strip()
            
            # College name
            if college_pattern.search(line):
                extracted_info['college'] = 'SRM AP'
            
            # Address
            if 'Neerukonda' in line or 'Mangalagiri' in line:
                extracted_info['address'] = line
            
            # Name
            if (name_pattern.match(line) or 
                (not any(c.isdigit() for c in line) and 
                 i < len(lines) - 1 and 
                 lines[i + 1].strip().startswith(('B', 'M', 'P')))) and not extracted_info['name']:
                extracted_info['name'] = line
            
            # Program
            if any(degree in line.upper() for degree in ['B.TECH', 'BTECH', 'B TECH']):
                extracted_info['program'] = 'B.Tech'
            
            # Branch
            branch_match = branch_pattern.search(line)
            if branch_match:
                extracted_info['branch'] = branch_match.group(0).upper()
            
            # Roll Number
            roll_match = roll_pattern.search(line)
            if roll_match:
                extracted_info['roll_number'] = roll_match.group(0)
            
            # Year
            year_match = year_pattern.search(line)
            if year_match:
                extracted_info['year'] = year_match.group(0)
        
        # Clean up empty values
        extracted_info = {k: v for k, v in extracted_info.items() if v}
        
        # Add minimal debug information
        extracted_info['debug'] = {
            'processed_lines': lines[:5],  # Only store first 5 lines for debugging
            'preprocessing': 'Optimized preprocessing applied'
        }
        
        return extracted_info

    except Exception as e:
        return {
            'error': f"Error in text extraction: {str(e)}",
            'college': '',
            'name': '',
            'program': '',
            'branch': '',
            'roll_number': '',
            'year': '',
            'address': ''
        }

def apply_adaptive_histogram_equalization(image):
    """Optimized adaptive histogram equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.merge((cl,a,b))

def remove_noise(image):
    """Remove noise using a combination of filters."""
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(bilateral, None, 10, 10, 7, 21)
    
    return denoised

def enhance_image(image):
    """Optimize image for face detection."""
    if image is None:
        return None
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Resize large images while maintaining aspect ratio
    max_dimension = 800
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_rotation(image, angle):
    """Process a single rotation angle."""
    if angle != 0:
        rotated = ndimage.rotate(image, angle, reshape=False)
    else:
        rotated = image
    
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7
    ) as face_detection:
        results = face_detection.process(rotated)
        return results, angle

def detect_faces(image):
    """Detect faces using cascade classifier with optimized parameters."""
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces found, try with more lenient parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    return faces

def verify_faces(model_name, metric, id_path, webcam_path, weight):
    """Verify faces using a specific model and metric."""
    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=webcam_path,
            model_name=model_name,
            detector_backend='retinaface',
            distance_metric=metric,
            enforce_detection=True
        )
        return (1 - result['distance']) * 100 * weight
    except Exception:
        return None

def verify_face_similarity(id_image_path, webcam_image_path):
    """Verify face similarity using DeepFace."""
    try:
        result = DeepFace.verify(
            img1_path=id_image_path,
            img2_path=webcam_image_path,
            model_name='VGG-Face',
            detector_backend='opencv',
            distance_metric='cosine',
            enforce_detection=False
        )
        return (1 - result['distance']) * 100
    except Exception as e:
        st.error(f"Face verification error: {str(e)}")
        return 0

def compare_faces(id_image_path, webcam_image_path):
    """Compare faces with improved error handling."""
    try:
        # Read images
        id_image = cv2.imread(id_image_path)
        webcam_image = cv2.imread(webcam_image_path)
        
        if id_image is None or webcam_image is None:
            return 0, "Error reading images", None, None
        
        # Enhance images
        id_image = enhance_image(id_image)
        webcam_image = enhance_image(webcam_image)
        
        # Detect faces
        id_faces = detect_faces(id_image)
        webcam_faces = detect_faces(webcam_image)
        
        if len(id_faces) == 0 and len(webcam_faces) == 0:
            return 0, "No faces detected in both images", None, None
        elif len(id_faces) == 0:
            return 0, "No face detected in ID card", None, None
        elif len(webcam_faces) == 0:
            return 0, "No face detected in webcam image", None, None
        
        # Verify face similarity
        similarity_score = verify_face_similarity(id_image_path, webcam_image_path)
        
        # Draw face boxes
        for (x, y, w, h) in id_faces:
            cv2.rectangle(id_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(id_image, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        for (x, y, w, h) in webcam_faces:
            cv2.rectangle(webcam_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(webcam_image, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert to RGB for display
        id_image_rgb = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        
        return similarity_score, "Face verification completed", id_image_rgb, webcam_image_rgb
    
    except Exception as e:
        st.error(f"Error during face comparison: {str(e)}")
        return 0, str(e), None, None

def verify_identity(id_image_path, webcam_image_path):
    """Main function to verify identity."""
    # Extract text from ID card
    ocr_data = extract_text_from_id(id_image_path)
    
    # Compare faces
    similarity_score, message, id_image_rgb, webcam_image_rgb = compare_faces(id_image_path, webcam_image_path)
    
    # Create verification prompt
    prompt = """
    You are an intelligent identity verification system. A student's ID card has been scanned, and their face has been captured via webcam. Your task is to validate if the identity on the ID card matches the live person based on both the textual and facial data. The extracted name is "{name}", roll number is "{roll_number}", and branch is "{branch}". The face similarity score (between the ID photo and live webcam feed) is {face_match_score} out of 100. First, evaluate if the name and roll number are plausible and match a real student format (e.g., SRM AP format). Then, assess if the face match score is high enough (scores above 60 are generally considered valid). Consider minor OCR mistakes and still try to infer the correct information if possible. Based on all this, provide a short analysis covering any issues or observations, and finally give a clear decision: either '✅ MATCH CONFIRMED' or '❌ MISMATCH DETECTED'. Your response should be clear, concise, and professional.
    """
    
    final_prompt = prompt.format(
        name=ocr_data['name'],
        roll_number=ocr_data['roll_number'],
        branch=ocr_data['branch'],
        face_match_score=similarity_score
    )
    
    # Get verification result from Gemini
    response = model.generate_content(final_prompt)
    return response.text

def main():
    st.title("Face Verification System")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .info-box {
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin: 10px 0;
        }
        .info-item {
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for verification type selection
    st.sidebar.title("Verification Type")
    verification_type = st.sidebar.radio(
        "Select Verification Type",
        ["College ID Card", "Other ID Proof"]
    )
    
    if verification_type == "College ID Card":
        # File uploader for ID card
        st.header("Step 1: Upload College ID Card")
        id_card = st.file_uploader("Upload ID Card Image", type=['jpg', 'jpeg', 'png'], key="college_id")
        
        if id_card is not None:
            # Save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(id_card.getvalue())
                id_path = tmp_file.name
            
            # Display the uploaded ID card
            st.image(id_card, caption="Uploaded ID Card", use_column_width=True)
            
            # Extract and display ID card details
            with st.spinner("Extracting ID card details..."):
                extracted_info = extract_text_from_id(id_path)
                
                st.markdown("### 📋 Extracted Information")
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                
                # Display extracted information
                if 'error' not in extracted_info:
                    for field, value in extracted_info.items():
                        if field not in ['raw_text', 'debug'] and value:  # Skip raw text, debug info and empty fields
                            st.markdown(f'<div class="info-item"><strong>{field.replace("_", " ").title()}:</strong> {value}</div>', 
                                      unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Option to show debug information
                with st.expander("Show Debug Information"):
                    if 'debug' in extracted_info:
                        st.json(extracted_info['debug'])
            
            st.header("Step 2: Take Live Photo")
            st.subheader("Take a Selfie")
            st.write("Please look directly at the camera and ensure good lighting")
            selfie = st.camera_input("Take a photo", key="camera_college")
            
            if id_card and selfie:
                try:
                    # Save selfie temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_selfie:
                        tmp_selfie.write(selfie.getvalue())
                        selfie_path = tmp_selfie.name
                    
                    # Perform verification
                    with st.spinner("🔍 Analyzing faces..."):
                        similarity_score, message, id_image_rgb, webcam_image_rgb = compare_faces(
                            id_path, 
                            selfie_path
                        )
                        
                        # Display verification result
                        st.markdown("### 📊 Verification Result")
                        
                        if similarity_score > 0:
                            # Display progress bar
                            st.progress(similarity_score / 100)
                            
                            # Display score with color coding
                            score_color = "green" if similarity_score >= 70 else "red"
                            st.markdown(f"<h2 style='color: {score_color}'>Similarity Score: {similarity_score:.1f}%</h2>", 
                                      unsafe_allow_html=True)
                            
                            if id_image_rgb is not None and webcam_image_rgb is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(id_image_rgb, caption="ID Card Analysis", use_column_width=True)
                                with col2:
                                    st.image(webcam_image_rgb, caption="Live Photo Analysis", use_column_width=True)
                            
                            if similarity_score >= 70:
                                st.success("✅ MATCH CONFIRMED")
                                st.balloons()
                            else:
                                st.error("❌ MISMATCH DETECTED")
                        else:
                            st.error(message)
                    
                    # Clean up temporary files
                    try:
                        os.unlink(id_path)
                        os.unlink(selfie_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"Error during verification: {str(e)}")
    
    else:  # Other ID Proof
        st.header("Other ID Proof Verification")
        st.write("Upload any government-issued ID card for face verification")
        
        # File uploader for other ID
        id_card = st.file_uploader("Upload ID Card Image", type=['jpg', 'jpeg', 'png'], key="other_id")
        
        if id_card is not None:
            # Save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(id_card.getvalue())
                id_path = tmp_file.name
            
            # Display the uploaded ID card
            st.image(id_card, caption="Uploaded ID Card", use_column_width=True)
            
            st.header("Take Live Photo")
            st.subheader("Take a Selfie")
            st.write("Please look directly at the camera and ensure good lighting")
            selfie = st.camera_input("Take a photo", key="camera_other")
            
            if id_card and selfie:
                try:
                    # Save selfie temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_selfie:
                        tmp_selfie.write(selfie.getvalue())
                        selfie_path = tmp_selfie.name
                    
                    # Perform verification
                    with st.spinner("🔍 Analyzing faces..."):
                        similarity_score, message, id_image_rgb, webcam_image_rgb = compare_faces(
                            id_path, 
                            selfie_path
                        )
                        
                        # Display verification result
                        st.markdown("### 📊 Verification Result")
                        
                        if similarity_score > 0:
                            # Display progress bar
                            st.progress(similarity_score / 100)
                            
                            # Display score with color coding
                            score_color = "green" if similarity_score >= 70 else "red"
                            st.markdown(f"<h2 style='color: {score_color}'>Similarity Score: {similarity_score:.1f}%</h2>", 
                                      unsafe_allow_html=True)
                            
                            if id_image_rgb is not None and webcam_image_rgb is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(id_image_rgb, caption="ID Card Analysis", use_column_width=True)
                                with col2:
                                    st.image(webcam_image_rgb, caption="Live Photo Analysis", use_column_width=True)
                            
                            if similarity_score >= 70:
                                st.success("✅ MATCH CONFIRMED")
                                st.balloons()
                            else:
                                st.error("❌ MISMATCH DETECTED")
                        else:
                            st.error(message)
                    
                    # Clean up temporary files
                    try:
                        os.unlink(id_path)
                        os.unlink(selfie_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"Error during verification: {str(e)}")

if __name__ == "__main__":
    main() 