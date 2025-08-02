import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import sqlite3
import os
import pandas as pd
import time
import glob
import uuid
import bcrypt
import face_recognition
import pickle
import dlib
import math
import urllib.request
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import random
from scipy import ndimage

# Set page configuration
st.set_page_config(
    page_title="Smart Face Recognition Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs("images", exist_ok=True)
os.makedirs("evidence", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Global variables and cache
if 'face_model_loaded' not in st.session_state:
    st.session_state.face_model_loaded = False
if 'face_database' not in st.session_state:
    st.session_state.face_database = {}
if 'face_recognition_accuracy' not in st.session_state:
    st.session_state.face_recognition_accuracy = 0.0
if 'model_training_progress' not in st.session_state:
    st.session_state.model_training_progress = 0.0
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'last_model_training_date' not in st.session_state:
    st.session_state.last_model_training_date = None

# Global state for user login
if 'user' not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.user_data = None
if 'verification_stage' not in st.session_state:
    st.session_state.verification_stage = "initial"
if 'detected_student' not in st.session_state:
    st.session_state.detected_student = None

# Load YOLO model for face detection
@st.cache_resource
def load_yolo_model():
    return torch.cuda.hub.load('ultralytics/yolov5', 'yolov5s')

def get_student_by_id(student_id):
    """Mengambil data mahasiswa berdasarkan ID"""
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE id=?", (student_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return dict(result)
    return None

def get_student_by_nim(nim):
    """Mengambil data mahasiswa berdasarkan NIM"""
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE nim=?", (nim,))
    result = c.fetchone()
    conn.close()
    if result:
        return dict(result)
    return None

def add_or_update_student(nim, name, email, password, class_name, student_id=None):
    """Menambah atau memperbarui data mahasiswa"""
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    if student_id:  # Update existing
        c.execute("""UPDATE students SET 
                    nim=?, name=?, email=?, password=?, class=?
                    WHERE id=?""",
                 (nim, name, email, hashed_pw, class_name, student_id))
    else:  # Insert new
        c.execute("""INSERT INTO students 
                    (nim, name, email, password, class)
                    VALUES (?, ?, ?, ?, ?)""",
                 (nim, name, email, hashed_pw, class_name))
    
    conn.commit()
    conn.close()

def get_course_by_id(course_id):
    """Mengambil data mata kuliah berdasarkan ID"""
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM courses WHERE id=?", (course_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return dict(result)
    return None

def add_or_update_course(code, name, lecturer, course_id=None):
    """Menambah atau memperbarui data mata kuliah"""
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    if course_id:  # Update existing
        c.execute("""UPDATE courses SET 
                    course_code=?, course_name=?, lecturer=?
                    WHERE id=?""",
                 (code, name, lecturer, course_id))
    else:  # Insert new
        c.execute("""INSERT INTO courses 
                    (course_code, course_name, lecturer)
                    VALUES (?, ?, ?)""",
                 (code, name, lecturer))
    
    conn.commit()
    conn.close()

def update_attendance_status(attendance_id, status, approved):
    """Memperbarui status absensi"""
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("""UPDATE attendances SET 
                status=?, admin_approved=?
                WHERE id=?""",
             (status, approved, attendance_id))
    conn.commit()
    conn.close()

def create_db():
    # Check if database already exists
    db_exists = os.path.exists("attendance.db")
    
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    # Students table
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nim TEXT UNIQUE,
                    name TEXT,
                    email TEXT UNIQUE,
                    password TEXT,
                    class TEXT,
                    face_image_path TEXT)''')
    
    # Courses table
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_code TEXT UNIQUE,
                    course_name TEXT,
                    lecturer TEXT)''')
    
    # Attendances table
    c.execute('''CREATE TABLE IF NOT EXISTS attendances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    course_id INTEGER,
                    attendance_date TEXT,
                    attendance_time TEXT,
                    status TEXT,
                    reason TEXT,
                    evidence_path TEXT,
                    admin_approved INTEGER DEFAULT 0,
                    FOREIGN KEY(student_id) REFERENCES students(id),
                    FOREIGN KEY(course_id) REFERENCES courses(id))''')
    
    # Admins table
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    password TEXT)''')
    
    # Only insert sample data if the database is being created for the first time
    if not db_exists:
        # Insert default admin
        c.execute("INSERT OR IGNORE INTO admins (email, password) VALUES (?, ?)", 
                ("admin@example.com", bcrypt.hashpw("admin".encode('utf-8'), bcrypt.gensalt())))
        
        # Insert sample student
        c.execute("INSERT OR IGNORE INTO students (nim, name, email, password, class, face_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                ("12345678", "Siswa Test", "siswa@example.com", bcrypt.hashpw("siswa".encode('utf-8'), bcrypt.gensalt()), "TI-1A", ""))
        
        # Insert sample student
        c.execute("INSERT OR IGNORE INTO students (nim, name, email, password, class, face_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                ("31231321", "Siswa Test 2", "siswa2@example.com", bcrypt.hashpw("siswa2".encode('utf-8'), bcrypt.gensalt()), "TI-1A", ""))
        
        # Insert sample courses
        c.execute("INSERT OR IGNORE INTO courses (course_code, course_name, lecturer) VALUES (?, ?, ?)",
                ("IF101", "Pengantar Informatika", "Dr. Budi"))
        c.execute("INSERT OR IGNORE INTO courses (course_code, course_name, lecturer) VALUES (?, ?, ?)",
                ("IF102", "Algoritma dan Pemrograman", "Dr. Ani"))
    
    conn.commit()
    conn.close()

def capture_attendance(student_id, course_id, status, reason="", evidence_path=""):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    # Debug: Print parameter yang diterima
    print(f"Menangkap kehadiran: student_id={student_id}, course_id={course_id}, status={status}, evidence_path={evidence_path}")
    
    c.execute("SELECT id FROM attendances WHERE student_id=? AND course_id=? AND attendance_date=?", 
              (student_id, course_id, date_str))
    existing = c.fetchone()
    
    if existing:
        c.execute("UPDATE attendances SET status=?, reason=?, evidence_path=?, attendance_time=? WHERE id=?",
                  (status, reason, evidence_path, time_str, existing[0]))
        message = f"Kehadiran diperbarui: {date_str} {time_str} - {status}"
    else:
        c.execute("""INSERT INTO attendances 
                    (student_id, course_id, attendance_date, attendance_time, status, reason, evidence_path, admin_approved) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                  (student_id, course_id, date_str, time_str, status, reason, evidence_path, 
                   1 if status == "Hadir" else 0))
        message = f"Kehadiran dicatat: {date_str} {time_str} - {status}"
    
    # Debug: Periksa perubahan di database
    c.execute("SELECT * FROM attendances WHERE student_id=? AND course_id=? AND attendance_date=?",
              (student_id, course_id, date_str))
    inserted = c.fetchone()
    print(f"Data yang baru dimasukkan: {inserted}")
    
    conn.commit()
    conn.close()
    return message

def login(email, password, role):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    if role == "Mahasiswa":
        c.execute("SELECT id, nim, name, class, password FROM students WHERE email = ?", (email,))
    else:
        c.execute("SELECT id, email, password FROM admins WHERE email = ?", (email,))
    
    user = c.fetchone()
    print("user = ", user)
    conn.close()
    
    if user:
        return {"id": user[0], **{k: v for k, v in zip(["nim", "name", "class"] if role == "Mahasiswa" else ["email"], user[1:-1])}}
    return None

def get_courses():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT id, course_code, course_name, lecturer FROM courses")
    courses = [{"id": row[0], "code": row[1], "name": row[2], "lecturer": row[3]} for row in c.fetchall()]
    conn.close()
    return courses

def get_students():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT id, nim, name, email, class FROM students")
    students = [{"id": row[0], "nim": row[1], "name": row[2], "email": row[3], "class": row[4]} for row in c.fetchall()]
    conn.close()
    return students

def get_attendance_data(student_id=None, course_id=None, date=None, status=None):
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = """
    SELECT 
        a.id, a.attendance_date, a.attendance_time, a.status, a.reason, a.admin_approved,
        s.nim, s.name, s.class,
        c.course_code, c.course_name, c.lecturer
    FROM attendances a
    JOIN students s ON a.student_id = s.id
    JOIN courses c ON a.course_id = c.id
    WHERE 1=1
    """
    params = []
    
    if student_id:
        query += " AND a.student_id = ?"
        params.append(student_id)
    
    if course_id:
        query += " AND a.course_id = ?"
        params.append(course_id)
    
    if date:
        query += " AND a.attendance_date = ?"
        params.append(date)
    
    if status:
        query += " AND a.status = ?"
        params.append(status)
    
    query += " ORDER BY a.attendance_date DESC, a.attendance_time DESC"
    
    c.execute(query, params)
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def save_uploaded_file(uploaded_file, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def detect_face(image_np):
    model = load_yolo_model()  # Load YOLO model
    results = model(image_np)  # Run inference
    faces = results.xyxy[0].cpu().numpy()  # Get detected faces
    if len(faces) > 0:
        x1, y1, x2, y2, conf, cls = faces[0]  # Get the first detected face
        face_image = image_np[int(y1):int(y2), int(x1):int(x2)]  # Crop the face
        return face_image
    return None

def extract_face_encoding(image_np):
    """
    Extract face encoding (feature vector) from an image.
    """
    face_locations = face_recognition.face_locations(image_np)
    if len(face_locations) == 0:
        return None
    face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
    return face_encoding

def compare_faces(face_encoding, reference_encoding):
    """
    Compare two face encodings using cosine similarity.
    """
    # Use a stricter tolerance (lower value = more strict)
    return face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.45)[0]

def load_face_encodings():
    """
    Load face encodings from a file.
    """
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            encodings = pickle.load(f)
            
            # Convert old format (single encoding per student) to new format (list of encodings)
            for student_id, encoding in encodings.items():
                if not isinstance(encoding, list):
                    encodings[student_id] = [encoding]
            
            return encodings
    return {}


def save_face_encodings(face_encodings):
    """
    Save face encodings to a file.
    """
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(face_encodings, f)

def augment_image(image_np):
    """
    Apply augmentation to create variations of the input image.
    Returns a list of augmented images including the original.
    """
    augmented_images = [image_np]  # Start with the original image
    
    # Rotation augmentations (slight angles)
    for angle in [-15, -10, -5, 5, 10, 15]:
        rotated = ndimage.rotate(image_np, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated)
    
    # Brightness augmentations
    for factor in [0.8, 1.2]:
        brightness = cv2.convertScaleAbs(image_np, alpha=factor, beta=0)
        augmented_images.append(brightness)
    
    # Slight horizontal shifts
    for shift in [-20, 20]:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        shifted = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]))
        augmented_images.append(shifted)
    
    # Horizontal flip (only if it makes sense for your use case)
    # flipped = cv2.flip(image_np, 1)
    # augmented_images.append(flipped)
    
    # Gaussian noise
    noise = np.copy(image_np)
    noise_factor = 0.05
    noise = noise + noise_factor * np.random.normal(0, 1, noise.shape)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    augmented_images.append(noise)
    
    return augmented_images

def register_face(student_id, image_np):
    """
    Register a face by extracting its encoding and saving it.
    Now with augmentation support to create multiple training samples.
    """
    # First check if the original image contains a valid face
    original_face_encoding = extract_face_encoding(image_np)
    if original_face_encoding is None:
        return False, 0
    
    # Load existing encodings
    face_encodings = load_face_encodings()
    
    # Initialize a list for this student if not already present
    if student_id not in face_encodings:
        face_encodings[student_id] = []
    
    # Add the original encoding
    face_encodings[student_id].append(original_face_encoding)
    
    # Generate augmented images
    augmented_images = augment_image(image_np)
    
    # Extract and add encodings from augmented images
    augmentation_count = 0
    for aug_img in augmented_images[1:]:  # Skip the first one (original)
        aug_face_encoding = extract_face_encoding(aug_img)
        if aug_face_encoding is not None:
            face_encodings[student_id].append(aug_face_encoding)
            augmentation_count += 1
    
    # Save all encodings
    save_face_encodings(face_encodings)
    return True, augmentation_count

def verify_face(image_np, student_id):
    """
    Verify if the detected face matches the registered face and return the accuracy.
    Modified to handle multiple encodings per student.
    """
    face_encodings = load_face_encodings()
    if student_id not in face_encodings or not face_encodings[student_id]:
        return False, 0.0
    
    face_encoding = extract_face_encoding(image_np)
    if face_encoding is None:
        return False, 0.0
    
    # Compare against all encodings for this student
    student_encodings = face_encodings[student_id]
    
    # Calculate face distances (lower is better)
    face_distances = face_recognition.face_distance(student_encodings, face_encoding)
    
    # Get the best match (lowest distance)
    best_match_idx = np.argmin(face_distances)
    best_distance = face_distances[best_match_idx]
    
    # Convert face distance to accuracy percentage (higher is better)
    accuracy = (1 - best_distance) * 100
    
    # Compare faces with a tolerance
    match = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.6)
    
    # If any encoding matches, consider it a match
    return any(match), accuracy

def get_face_landmarks(image_np):
    """Detect facial landmarks using dlib"""
    detector = dlib.DLIB_USE_CUDA.get_frontal_face_detector()
    predictor = dlib.DLIB_USE_CUDA.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return None
    
    # Get landmarks for the first face
    landmarks = predictor(gray, faces[0])
    return landmarks

def calculate_head_pose(landmarks, image_np):
    """Calculate head pose angles"""
    # Get 2D facial landmarks
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    size = image_np.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Calculate rotation angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    return angles

def verify_movement(landmarks, movement_type, image_np):
    """Verify if the user performed the requested movement"""
    angles = calculate_head_pose(landmarks, image_np)
    
    if movement_type == "blink":
        # Check for eye blinking by measuring eye aspect ratio (EAR)
        left_eye = []
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        right_eye = []
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # Calculate eye aspect ratio
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Threshold for blink detection
        return ear < 0.2  # Lower EAR means eyes are more closed
    elif movement_type == "left":
        # Check for horizontal movement to left (yaw)
        # Negative yaw angle means looking to the right from camera perspective
        # Positive yaw angle means looking to the left from camera perspective
        return angles[1] > 15  # Threshold for looking left
    elif movement_type == "right":
        # Check for horizontal movement to right (yaw)
        return angles[1] < -15  # Threshold for looking right
    return False

def calculate_ear(eye):
    """Calculate eye aspect ratio for blink detection"""
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def realtime_face_movement_verification(face_encodings):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    # Initialize variables
    verification_started = False
    movement_verified = False
    face_verified = False
    detected_student_id = None
    detected_face_encoding = None  # Menyimpan encoding wajah yang sudah terverifikasi
    current_movement = get_random_movement()
    movement_instructions = {
        "blink": "Kedipkan mata Anda",
        "left": "Lihat ke kiri",
        "right": "Lihat ke kanan"
    }
    
    # Load face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Store verification results
    verification_complete = False
    evidence_path = ""
    verification_attempts = 0
    max_verification_attempts = 3
    matched_student_id = None
    match_confidence = 0.0
    
    while not verification_complete:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip horizontal supaya mirror
        if not ret:
            st.error("Tidak dapat mengakses kamera")
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector(frame_rgb)
        
        if len(faces) > 0:
            # Get landmarks for the primary face (first face detected)
            landmarks = predictor(frame_rgb, faces[0])
            
            # Draw face bounding box
            x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate head pose
            angles = calculate_head_pose(landmarks, frame)
            
            # Show instructions
            if not verification_started:
                cv2.putText(frame, "Tunjukkan wajah Anda", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Auto-start verification when face is centered
                face_center = ((x1+x2)//2, (y1+y2)//2)
                frame_center = (frame.shape[1]//2, frame.shape[0]//2)
                if abs(face_center[0] - frame_center[0]) < 50 and abs(face_center[1] - frame_center[1]) < 50:
                    verification_started = True
            
            if verification_started and not face_verified:
                # Verify face
                current_face_encoding = extract_face_encoding(frame_rgb)
                if current_face_encoding is None:
                    face_verified = False
                else:
                    # Track multiple attempts and use consensus
                    verification_attempts += 1
                    
                    # Check against all registered faces
                    best_match_id = None
                    best_match_distance = 1.0  # Lower is better
                    
                    for student_id, registered_encodings in face_encodings.items():
                        # Calculate face distance (lower is better) against all encodings for this student
                        if isinstance(registered_encodings, list):
                            # Multiple encodings per student
                            distances = face_recognition.face_distance(registered_encodings, current_face_encoding)
                            min_distance = np.min(distances) if len(distances) > 0 else 1.0
                        else:
                            # Single encoding (legacy format)
                            min_distance = face_recognition.face_distance([registered_encodings], current_face_encoding)[0]
                        
                        # If this is a better match than previous ones
                        if min_distance < best_match_distance and min_distance < 0.55:  # Threshold for considering a match
                            best_match_distance = min_distance
                            best_match_id = student_id
                    
                    # If we found a match
                    if best_match_id is not None:
                        # Calculate confidence percentage (higher is better)
                        confidence = (1 - best_match_distance) * 100
                        
                        # Track the most consistent match across attempts
                        if matched_student_id is None:
                            matched_student_id = best_match_id
                            match_confidence = confidence
                        elif matched_student_id == best_match_id:
                            # Same student matched again, increase confidence
                            match_confidence = (match_confidence + confidence) / 2
                        else:
                            # Different student matched, reset
                            matched_student_id = best_match_id
                            match_confidence = confidence
                            verification_attempts = 1
                        
                        # Display match information
                        cv2.putText(frame, f"Match: {int(confidence)}%", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # After several consistent matches with high confidence, accept the face
                        if verification_attempts >= max_verification_attempts and match_confidence > 60:
                            face_verified = True
                            detected_student_id = matched_student_id
                            detected_face_encoding = current_face_encoding  # Simpan encoding wajah yang terverifikasi
                            current_movement = get_random_movement()
                    else:
                        # Reset if no match found
                        verification_attempts = 0
                        matched_student_id = None
                        match_confidence = 0.0
                        cv2.putText(frame, "Wajah tidak dikenali", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if not face_verified and verification_attempts > 0:
                    cv2.putText(frame, f"Verifikasi: {verification_attempts}/{max_verification_attempts}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            if face_verified and not movement_verified:
                # 1. Verifikasi wajah masih sama dengan yang terdaftar
                current_face_encoding = extract_face_encoding(frame_rgb)
                if current_face_encoding is None:
                    face_verified = False  # Wajah hilang, reset verifikasi
                    cv2.putText(frame, "Wajah tidak terdeteksi!", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Bandingkan dengan wajah yang sudah terverifikasi
                    registered_encodings = face_encodings[detected_student_id]
                    if isinstance(registered_encodings, list):
                        face_distance = face_recognition.face_distance(registered_encodings, current_face_encoding)
                        min_distance = np.min(face_distance)
                    else:
                        min_distance = face_recognition.face_distance([registered_encodings], current_face_encoding)[0]
                    
                    if min_distance > 0.55:  # Wajah berbeda
                        face_verified = False
                        cv2.putText(frame, "Wajah berubah! Verifikasi dibatalkan.", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # 2. Lanjutkan verifikasi gerakan
                        cv2.putText(frame, movement_instructions[current_movement], (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Visualize eye landmarks for blink detection
                        if current_movement == "blink":
                            # Draw eye landmarks
                            for i in range(36, 48):  # Eye landmarks
                                x = landmarks.part(i).x
                                y = landmarks.part(i).y
                                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        
                        # Add direction indicators for left/right movements
                        if current_movement == "left":
                            cv2.arrowedLine(frame, 
                                           (frame.shape[1]//2, 30), 
                                           (frame.shape[1]//2 - 50, 30), 
                                           (255, 255, 0), 2)
                        elif current_movement == "right":
                            cv2.arrowedLine(frame, 
                                           (frame.shape[1]//2, 30), 
                                           (frame.shape[1]//2 + 50, 30), 
                                           (255, 255, 0), 2)
                        
                        # Verify movement
                        if verify_movement(landmarks, current_movement, frame):
                            movement_verified = True
                            cv2.putText(frame, "Verifikasi berhasil!", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Get student data
                            student = get_student_by_id(detected_student_id)
                            
                            # Save evidence
                            evidence_filename = f"{student['nim']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                            evidence_path = os.path.join("evidence", evidence_filename)
                            cv2.imwrite(evidence_path, frame)
                            verification_complete = True
            
            # Draw head pose angles
            cv2.putText(frame, f"Pitch: {angles[0]:.1f}", (10, frame.shape[0]-60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Yaw: {angles[1]:.1f}", (10, frame.shape[0]-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Roll: {angles[2]:.1f}", (10, frame.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        stframe.image(frame, channels="BGR", use_container_width=True)
        
        # Exit if verification complete or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    return detected_student_id, evidence_path

def draw_landmarks_and_pose(image_np, landmarks, angles):
    """Draw facial landmarks and pose angles on image"""
    # Draw landmarks
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image_np, (x, y), 2, (0, 255, 0), -1)
    
    # Display pose angles
    cv2.putText(image_np, f"Pitch: {angles[0]:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image_np, f"Yaw: {angles[1]:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image_np, f"Roll: {angles[2]:.1f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image_np

def get_random_movement():
    """Get a random movement instruction"""
    movements = ["blink", "left", "right"]
    return np.random.choice(movements)

def download_dlib_model():
    """Download the dlib face landmark predictor model if it doesn't exist"""
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        st.info("Mengunduh model deteksi wajah...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, model_path + ".bz2")
        
        import bz2
        with bz2.BZ2File(model_path + ".bz2", 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(model_path + ".bz2")
        st.success("Model berhasil diunduh!")

def delete_student(student_id):
    """Menghapus data mahasiswa berdasarkan ID"""
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    # Hapus face encoding terlebih dahulu jika ada
    face_encodings = load_face_encodings()
    if student_id in face_encodings:
        del face_encodings[student_id]
        save_face_encodings(face_encodings)
    
    # Hapus data kehadiran terkait
    c.execute("DELETE FROM attendances WHERE student_id=?", (student_id,))
    
    # Hapus data mahasiswa
    c.execute("DELETE FROM students WHERE id=?", (student_id,))
    
    conn.commit()
    conn.close()
    return True

def delete_course(course_id):
    """Menghapus data mata kuliah berdasarkan ID"""
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    # Hapus data kehadiran terkait
    c.execute("DELETE FROM attendances WHERE course_id=?", (course_id,))
    
    # Hapus data mata kuliah
    c.execute("DELETE FROM courses WHERE id=?", (course_id,))
    
    conn.commit()
    conn.close()
    return True

def main():
    create_db()
    download_dlib_model()  # Download dlib model at startup
    
    # Sidebar for navigation
    st.sidebar.header("Navigasi")
    role = st.sidebar.radio("Menu", ["Login Mahasiswa", "Login Admin", "Absensi"])

    # Define menu for each role
    if role == "Login Mahasiswa":
        if st.session_state.role != "Mahasiswa":
            menu = st.sidebar.radio("Menu", ["Login"])
        else:
            menu = st.sidebar.radio("Menu", ["Dashboard", "Lihat Data Presensi", "Logout"])
    elif role == "Login Admin":
        if st.session_state.role != "Admin":
            menu = st.sidebar.radio("Menu", ["Login"])
        else:
            menu = st.sidebar.radio("Menu", [
                "Dashboard", "Kelola Data Mahasiswa", "Kelola Data Mata Kuliah",
                "Validasi Kehadiran", "Registrasi Wajah", "Laporan Kehadiran", "Logout"
            ])
    elif role == "Absensi":
        menu = "Absensi"  # Only one menu for this role

    # Mahasiswa section
    if role == "Login Mahasiswa":
        st.sidebar.subheader("Login Mahasiswa")
        
        if menu == "Login" or st.session_state.role != "Mahasiswa":
            st.header("Login Mahasiswa")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                user = login(email, password, "Mahasiswa")
                if user:
                    st.session_state.user = user["id"]
                    st.session_state.role = "Mahasiswa"
                    st.session_state.user_data = user
                    st.success("Login berhasil!")
                    st.rerun()
                else:
                    st.error("Login gagal. Periksa kembali email dan password.")
        
        elif menu == "Dashboard":
            st.header(f"Selamat Datang, {st.session_state.user_data['name']}")
            st.subheader("Informasi Mahasiswa")
            st.write(f"NIM: {st.session_state.user_data['nim']}")
            st.write(f"Kelas: {st.session_state.user_data['class']}")
            
            st.subheader("Ringkasan Kehadiran")
            attendance_data = get_attendance_data(student_id=st.session_state.user)
            
            if attendance_data:
                # Count by status
                status_counts = {}
                for item in attendance_data:
                    status = item['status']
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        status_counts[status] = 1
                
                st.write("Jumlah Kehadiran:")
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")
                
                # Show recent attendance
                st.subheader("Absensi Terbaru")
                recent = attendance_data[:5]  # Get 5 most recent
                
                for item in recent:
                    status_color = "green" if item['status'] == "Hadir" else "orange" if item['status'] in ["Izin", "Sakit"] else "red"
                    approval_status = "✓ Disetujui" if item['admin_approved'] == 1 else "⚠️ Menunggu Persetujuan"
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 5px solid {status_color}; margin-bottom: 10px;">
                        <p><strong>{item['course_code']}</strong>: {item['course_name']}</p>
                        <p>Tanggal: {item['attendance_date']} {item['attendance_time']}</p>
                        <p>Status: <span style="color:{status_color}">{item['status']}</span> - {approval_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Belum ada data kehadiran.")
        
        elif menu == "Lihat Data Presensi":
            st.header("Data Presensi Mahasiswa")
            
            # Filter options
            courses = get_courses()
            course_options = {"Semua Mata Kuliah": None}
            course_options.update({f"{c['code']} - {c['name']}": c['id'] for c in courses})
            
            selected_course = st.selectbox("Filter Mata Kuliah", options=list(course_options.keys()))
            course_filter = course_options[selected_course]
            
            status_filter = st.selectbox("Filter Status", ["Semua", "Hadir", "Izin", "Sakit", "Alpa"])
            status_filter = None if status_filter == "Semua" else status_filter
            
            # Get attendance data
            attendance_data = get_attendance_data(student_id=st.session_state.user, course_id=course_filter, status=status_filter)
            
            if attendance_data:
                # Convert to DataFrame for display
                df = pd.DataFrame(attendance_data)
                df = df[['attendance_date', 'attendance_time', 'course_code', 'course_name', 'status', 'reason', 'admin_approved']]
                df.columns = ['Tanggal', 'Waktu', 'Kode MK', 'Mata Kuliah', 'Status', 'Alasan', 'Disetujui']
                
                # Format approval status
                df['Disetujui'] = df['Disetujui'].apply(lambda x: "Ya" if x == 1 else "Menunggu")
                
                st.dataframe(df)
            else:
                st.info("Tidak ada data kehadiran yang ditemukan.")
        
        elif menu == "Logout":
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.user_data = None
            st.success("Logout berhasil")
            st.rerun()
    
    
    elif role == "Login Admin":
        st.sidebar.subheader("Login Admin")
        
        if menu == "Login" or st.session_state.role != "Admin":
            st.header("Login Admin")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                user = login(email, password, "Admin")
                if user:
                    st.session_state.user = user["id"]
                    st.session_state.role = "Admin"
                    st.session_state.user_data = user
                    st.success("Login berhasil!")
                    st.rerun()
                else:
                    st.error("Login gagal. Periksa kembali email dan password.")
    
        elif menu == "Dashboard":
            st.header("Dashboard Admin")
            
            # Summary stats
            st.subheader("Ringkasan")
            
            # Get all attendance data
            all_attendance = get_attendance_data()
            
            if all_attendance:
                # Count by status
                status_counts = {}
                for item in all_attendance:
                    status = item['status']
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        status_counts[status] = 1
                
                # Count pending approvals
                pending_count = sum(1 for item in all_attendance if item['admin_approved'] == 0)
                
                # Display stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Hadir", status_counts.get("Hadir", 0))
                
                with col2:
                    st.metric("Izin", status_counts.get("Izin", 0))
                
                with col3:
                    st.metric("Sakit", status_counts.get("Sakit", 0))
                
                with col4:
                    st.metric("Alpa", status_counts.get("Alpa", 0))
                
                st.metric("Permintaan Menunggu Persetujuan", pending_count)
                
                # Recent activity
                st.subheader("Aktivitas Terbaru")
                recent = all_attendance[:10]  # Get 10 most recent
                
                for item in recent:
                    status_color = "green" if item['status'] == "Hadir" else "orange" if item['status'] in ["Izin", "Sakit"] else "red"
                    approval_status = "✓ Disetujui" if item['admin_approved'] == 1 else "⚠️ Menunggu Persetujuan"
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 5px solid {status_color}; margin-bottom: 10px;">
                        <p><strong>{item['nim']} - {item['name']}</strong></p>
                        <p>{item['course_code']}: {item['course_name']}</p>
                        <p>Tanggal: {item['attendance_date']} {item['attendance_time']}</p>
                        <p>Status: <span style="color:{status_color}">{item['status']}</span> - {approval_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Belum ada data kehadiran.")
        
        elif menu == "Kelola Data Mahasiswa":
            st.header("Kelola Data Mahasiswa")
            
            tab1, tab2 = st.tabs(["Daftar Mahasiswa", "Tambah/Edit Mahasiswa"])
            
            with tab1:
                students = get_students()
                if students:
                    df = pd.DataFrame(students)
                    df = df[['id', 'nim', 'name', 'email', 'class']]
                    df.columns = ['ID', 'NIM', 'Nama', 'Email', 'Kelas']
                    
                    st.dataframe(df)
                    
                    # Select student to edit or delete
                    student_options = [f"{s['nim']} - {s['name']}" for s in students]
                    student_to_edit = st.selectbox("Pilih Mahasiswa", 
                                                options=student_options,
                                                key="student_select")
                    
                    if student_to_edit:
                        selected_index = student_options.index(student_to_edit)
                        student_id = students[selected_index]['id']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Edit Mahasiswa", key="btn_edit_student"):
                                st.session_state.edit_student = student_id
                                st.rerun()
                        
                        with col2:
                            if st.button("Hapus Mahasiswa", key="btn_delete_student"):
                                if delete_student(student_id):
                                    st.success(f"Mahasiswa {student_to_edit} berhasil dihapus!")
                                    st.rerun()
                                else:
                                    st.error("Gagal menghapus data mahasiswa")
                else:
                    st.info("Belum ada data mahasiswa.")
            
            with tab2:
                # Check if editing existing student
                editing_student = None
                if hasattr(st.session_state, 'edit_student'):
                    student_id = st.session_state.edit_student
                    editing_student = get_student_by_id(student_id)
                    
                    # Cek apakah data mahasiswa ditemukan
                    if editing_student is None:
                        st.error(f"Data mahasiswa dengan ID {student_id} tidak ditemukan. Mungkin sudah dihapus.")
                        del st.session_state.edit_student
                        st.subheader("Tambah Mahasiswa Baru")
                    else:
                        st.subheader(f"Edit Mahasiswa: {editing_student['name']}")
                        # Store the ID in editing_student for later use
                        editing_student['id'] = student_id
                        # Don't delete session state until after save
                else:
                    st.subheader("Tambah Mahasiswa Baru")
                
                # Form fields
                nim = st.text_input("NIM", value=editing_student['nim'] if editing_student else "")
                name = st.text_input("Nama", value=editing_student['name'] if editing_student else "")
                email = st.text_input("Email", value=editing_student['email'] if editing_student else "")
                password = st.text_input("Password", type="password", 
                                    help="Biarkan kosong jika tidak ingin mengubah password" if editing_student else "")
                class_name = st.text_input("Kelas", value=editing_student['class'] if editing_student else "")
                
                if st.button("Simpan" if editing_student else "Tambah"):
                    if nim and name and email and class_name:
                        try:
                            # Jika edit dan password kosong, gunakan password lama
                            if editing_student and not password:
                                # Ambil password lama dari database
                                conn = sqlite3.connect("attendance.db")
                                c = conn.cursor()
                                c.execute("SELECT password FROM students WHERE id=?", (editing_student['id'],))
                                result = c.fetchone()
                                conn.close()
                                
                                if result is None:
                                    st.error("Data mahasiswa tidak ditemukan. Mungkin sudah dihapus.")
                                    st.rerun()
                                    
                                old_password = result[0]
                                
                                # Update tanpa mengubah password
                                conn = sqlite3.connect("attendance.db")
                                c = conn.cursor()
                                c.execute("""UPDATE students SET 
                                            nim=?, name=?, email=?, class=?
                                            WHERE id=?""",
                                        (nim, name, email, class_name, editing_student['id']))
                                conn.commit()
                                conn.close()
                            else:
                                # Update dengan password baru atau tambah baru
                                add_or_update_student(
                                    nim, name, email, password if password else "password", class_name, 
                                    student_id=editing_student['id'] if editing_student else None
                                )
                            
                            # Clear edit state after successful save
                            if hasattr(st.session_state, 'edit_student'):
                                del st.session_state.edit_student
                                
                            st.success(f"Mahasiswa berhasil {'diperbarui' if editing_student else 'ditambahkan'}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                    else:
                        st.error("NIM, Nama, Email, dan Kelas harus diisi!")
        
        elif menu == "Kelola Data Mata Kuliah":
            st.header("Kelola Data Mata Kuliah")
            
            tab1, tab2 = st.tabs(["Daftar Mata Kuliah", "Tambah/Edit Mata Kuliah"])
            
            with tab1:
                courses = get_courses()
                if courses:
                    df = pd.DataFrame(courses)
                    df = df[['id', 'code', 'name', 'lecturer']]
                    df.columns = ['ID', 'Kode', 'Nama', 'Dosen']
                    
                    st.dataframe(df)
                    
                    # Select course to edit or delete
                    course_options = [f"{c['code']} - {c['name']}" for c in courses]
                    course_to_edit = st.selectbox("Pilih Mata Kuliah", 
                                                options=course_options,
                                                key="course_select")
                    
                    if course_to_edit:
                        selected_index = course_options.index(course_to_edit)
                        course_id = courses[selected_index]['id']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Edit Mata Kuliah", key="btn_edit_course"):
                                st.session_state.edit_course = course_id
                                st.rerun()
                        
                        with col2:
                            if st.button("Hapus Mata Kuliah", key="btn_delete_course"):
                                if delete_course(course_id):
                                    st.success(f"Mata Kuliah {course_to_edit} berhasil dihapus!")
                                    st.rerun()
                                else:
                                    st.error("Gagal menghapus data mata kuliah")
                else:
                    st.info("Belum ada data mata kuliah.")
            
            with tab2:
                # Check if editing existing course
                editing_course = None
                if hasattr(st.session_state, 'edit_course'):
                    course_id = st.session_state.edit_course
                    editing_course = get_course_by_id(course_id)
                    
                    # Cek apakah data mata kuliah ditemukan
                    if editing_course is None:
                        st.error(f"Data mata kuliah dengan ID {course_id} tidak ditemukan. Mungkin sudah dihapus.")
                        del st.session_state.edit_course
                        st.subheader("Tambah Mata Kuliah Baru")
                    else:
                        st.subheader(f"Edit Mata Kuliah: {editing_course['course_name']}")
                        # Store the ID in editing_course for later use
                        editing_course['id'] = course_id
                        # Don't delete session state until after save
                else:
                    st.subheader("Tambah Mata Kuliah Baru")
                
                # Form fields
                code = st.text_input("Kode Mata Kuliah", value=editing_course['course_code'] if editing_course else "")
                name = st.text_input("Nama Mata Kuliah", value=editing_course['course_name'] if editing_course else "")
                lecturer = st.text_input("Dosen", value=editing_course['lecturer'] if editing_course else "")
                
                if st.button("Simpan" if editing_course else "Tambah"):
                    if code and name and lecturer:
                        try:
                            add_or_update_course(
                                code, name, lecturer, 
                                course_id=editing_course['id'] if editing_course else None
                            )
                            
                            # Clear edit state after successful save
                            if hasattr(st.session_state, 'edit_course'):
                                del st.session_state.edit_course
                                
                            st.success(f"Mata kuliah berhasil {'diperbarui' if editing_course else 'ditambahkan'}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                    else:
                        st.error("Semua field harus diisi!")
        
        elif menu == "Validasi Kehadiran":
            st.header("Validasi Kehadiran")
            
            # Get pending approval requests
            pending_requests = [a for a in get_attendance_data() if a['admin_approved'] == 0]
            
            if pending_requests:
                st.write(f"{len(pending_requests)} permintaan menunggu persetujuan")
                
                for req in pending_requests:
                    with st.expander(f"{req['name']} - {req['course_name']} - {req['attendance_date']}"):
                        st.write(f"Mahasiswa: {req['nim']} - {req['name']}")
                        st.write(f"Mata Kuliah: {req['course_code']} - {req['course_name']}")
                        st.write(f"Tanggal: {req['attendance_date']} {req['attendance_time']}")
                        st.write(f"Status yang Diminta: {req['status']}")
                        
                        if req['reason']:
                            st.write(f"Alasan: {req['reason']}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"Setujui ({req['status']})", key=f"approve_{req['id']}"):
                                update_attendance_status(req['id'], req['status'], 1)
                                st.success("Permintaan disetujui")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"Tandai sebagai Alpa", key=f"alpa_{req['id']}"):
                                update_attendance_status(req['id'], "Alpa", 1)
                                st.success("Ditandai sebagai Alpa")
                                st.rerun()
        
        elif menu == "Registrasi Wajah":
            st.header("Registrasi Wajah Mahasiswa")
            
            # Select student to register face
            students = get_students()
            student_options = {f"{s['nim']} - {s['name']}": s['id'] for s in students}
            
            selected_student = st.selectbox("Pilih Mahasiswa", options=list(student_options.keys()))
            student_id = student_options[selected_student]
            
            # Option for augmentation
            use_augmentation = st.checkbox("Gunakan Augmentasi Data (menghasilkan variasi gambar tambahan)", value=True)
            
            if use_augmentation:
                st.info("Augmentasi data akan menghasilkan beberapa variasi dari setiap gambar yang diunggah untuk meningkatkan akurasi pengenalan wajah.")
            
            # Capture multiple face images
            uploaded_files = st.file_uploader("Unggah Foto Wajah", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            
            if uploaded_files:
                success_count = 0
                total_augmentations = 0
                
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    
                    # Konversi PNG ke JPG jika format file adalah PNG
                    if uploaded_file.type == "image/png":
                        # Konversi ke RGB (menghapus alpha channel jika ada)
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                        
                        # Membuat file temporary untuk menyimpan hasil konversi
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                            image.save(temp_file.name, format='JPEG', quality=90)
                            # Reload image from the temporary file
                            image = Image.open(temp_file.name)
                            st.info(f"Gambar {uploaded_file.name} dikonversi dari PNG ke JPG")
                    
                    image_np = np.array(image)

                    # Detect faces using YOLO
                    results = load_yolo_model()(image_np)
                    faces = results.xyxy[0].cpu().numpy()

                    if len(faces) > 0:
                        st.image(image, caption="Foto Wajah yang Diunggah", use_container_width=True)
                        
                        # Register face
                        if use_augmentation:
                            success, aug_count = register_face(student_id, image_np)
                            if success:
                                success_count += 1
                                total_augmentations += aug_count
                                st.success(f"Wajah pada {uploaded_file.name} terdeteksi dengan {aug_count} variasi tambahan.")
                            else:
                                st.error(f"Wajah pada {uploaded_file.name} tidak terdeteksi.")
                        else:
                            # Legacy registration without augmentation
                            face_encoding = extract_face_encoding(image_np)
                            if face_encoding is not None:
                                face_encodings = load_face_encodings()
                                if student_id not in face_encodings:
                                    face_encodings[student_id] = []
                                face_encodings[student_id].append(face_encoding)
                                save_face_encodings(face_encodings)
                                success_count += 1
                                st.success(f"Wajah pada {uploaded_file.name} terdeteksi.")
                            else:
                                st.error(f"Wajah pada {uploaded_file.name} tidak terdeteksi.")

                if success_count > 0:
                    if use_augmentation:
                        st.success(f"{success_count} wajah berhasil diregistrasi dengan total {total_augmentations} variasi tambahan!")
                    else:
                        st.success(f"{success_count} wajah berhasil diregistrasi!")
                else:
                    st.error("Tidak ada wajah yang berhasil diregistrasi. Pastikan foto jelas.")
        
        elif menu == "Laporan Kehadiran":
            st.header("Laporan Kehadiran")
            
            # Filter options
            st.subheader("Filter Laporan")
            
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Tanggal Mulai")
            with col2:
                end_date = st.date_input("Tanggal Selesai")
            
            # Course filter
            courses = get_courses()
            course_options = {"Semua Mata Kuliah": None}
            course_options.update({f"{c['code']} - {c['name']}": c['id'] for c in courses})
            selected_course = st.selectbox("Filter Mata Kuliah", options=list(course_options.keys()))
            course_filter = course_options[selected_course]
            
            # Student filter
            students = get_students()
            student_options = {"Semua Mahasiswa": None}
            student_options.update({f"{s['nim']} - {s['name']}": s['id'] for s in students})
            selected_student = st.selectbox("Filter Mahasiswa", options=list(student_options.keys()))
            student_filter = student_options[selected_student]
            
            # Status filter
            status_filter = st.selectbox("Filter Status", ["Semua", "Hadir", "Izin", "Sakit", "Alpa"])
            status_filter = None if status_filter == "Semua" else status_filter
            
            # Get attendance data
            attendance_data = get_attendance_data(student_id=student_filter, course_id=course_filter, status=status_filter)
            
            # Filter by date range
            if start_date and end_date:
                attendance_data = [a for a in attendance_data if start_date <= datetime.strptime(a['attendance_date'], "%Y-%m-%d").date() <= end_date]
            
            if attendance_data:
                # Convert to DataFrame for display
                df = pd.DataFrame(attendance_data)
                df = df[['attendance_date', 'attendance_time', 'nim', 'name', 'course_code', 'course_name', 'status', 'reason', 'admin_approved']]
                df.columns = ['Tanggal', 'Waktu', 'NIM', 'Nama', 'Kode MK', 'Mata Kuliah', 'Status', 'Alasan', 'Disetujui']
                
                # Format approval status
                df['Disetujui'] = df['Disetujui'].apply(lambda x: "Ya" if x == 1 else "Tidak")
                
                st.dataframe(df)
                
                # Create PDF report
                def create_pdf(df):
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
                    elements = []
                    
                    # Add title
                    styles = getSampleStyleSheet()
                    title_style = styles['Heading1']
                    title = Paragraph("Laporan Kehadiran Mahasiswa", title_style)
                    elements.append(title)
                    
                    # Add date range
                    date_style = styles['Normal']
                    if start_date and end_date:
                        date_range = Paragraph(f"Periode: {start_date.strftime('%d-%m-%Y')} s/d {end_date.strftime('%d-%m-%Y')}", date_style)
                        elements.append(date_range)
                    
                    # Add course filter
                    if selected_course != "Semua Mata Kuliah":
                        course_info = Paragraph(f"Mata Kuliah: {selected_course}", date_style)
                        elements.append(course_info)
                    
                    # Add student filter
                    if selected_student != "Semua Mahasiswa":
                        student_info = Paragraph(f"Mahasiswa: {selected_student}", date_style)
                        elements.append(student_info)
                    
                    # Add status filter
                    if status_filter:
                        status_info = Paragraph(f"Status: {status_filter}", date_style)
                        elements.append(status_info)
                    
                    # Add space
                    elements.append(Paragraph("<br/><br/>", styles['Normal']))
                    
                    # Convert DataFrame to table data
                    data = [df.columns.tolist()] + df.values.tolist()
                    
                    # Create table
                    table = Table(data)
                    
                    # Add table style
                    style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ])
                    
                    # Add alternating row colors
                    for i in range(1, len(data)):
                        if i % 2 == 0:
                            style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
                    
                    table.setStyle(style)
                    elements.append(table)
                    
                    # Build PDF
                    doc.build(elements)
                    buffer.seek(0)
                    return buffer
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh Laporan (CSV)",
                        data=csv,
                        file_name="laporan_kehadiran.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    pdf_buffer = create_pdf(df)
                    st.download_button(
                        label="Unduh Laporan (PDF)",
                        data=pdf_buffer,
                        file_name="laporan_kehadiran.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("Tidak ada data kehadiran yang ditemukan.")
        
        elif menu == "Logout":
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.user_data = None
            st.success("Logout berhasil")
            st.rerun()
            
    if menu == "Absensi":
        st.header("Absensi Mahasiswa")
        
        # Get courses for dropdown
        courses = get_courses()
        course_options = {f"{c['code']} - {c['name']}": c['id'] for c in courses}
        
        selected_course = st.selectbox("Pilih Mata Kuliah", options=list(course_options.keys()))
        course_id = course_options[selected_course]
        
        attendance_status = st.radio("Status Kehadiran", ["Hadir", "Izin", "Sakit"])
        
        if attendance_status in ["Izin", "Sakit"]:
            # Input NIM for absence requests
            nim = st.text_input("Masukkan NIM Anda")
            reason = st.text_area("Alasan", help="Berikan alasan ketidakhadiran")
            evidence_path = ""
            
            uploaded_file = st.file_uploader("Bukti (Opsional)", type=["jpg", "png", "jpeg", "pdf"])
            if uploaded_file:
                file_ext = uploaded_file.name.split(".")[-1]
                filename = f"{nim}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_ext}"
                evidence_path = save_uploaded_file(uploaded_file, "evidence", filename)
                st.success("Bukti berhasil diunggah")
            
            if st.button("Kirim Permintaan"):
                if not nim:
                    st.error("Mohon masukkan NIM Anda")
                    return
                
                # Check if NIM exists in database
                student = get_student_by_nim(nim)
                if not student:
                    st.error("NIM tidak ditemukan dalam database")
                    return
                
                message = capture_attendance(student['id'], course_id, attendance_status, reason, evidence_path)
                st.success(f"{message}\nPermintaan Anda akan diverifikasi oleh admin.")
        
        elif attendance_status == "Hadir":
            # Tahap awal - tombol untuk memulai verifikasi
            if st.session_state.verification_stage == "initial":
                if st.button("Mulai Verifikasi Wajah"):
                    st.session_state.verification_stage = "verifying"
                    st.rerun()
            
            # Tahap verifikasi - menampilkan kamera dan melakukan verifikasi
            elif st.session_state.verification_stage == "verifying":
                st.write("Mendeteksi wajah...")
                
                # Load face encodings
                face_encodings = load_face_encodings()
                
                # Use the realtime face movement verification function
                detected_student_id, evidence_path = realtime_face_movement_verification(face_encodings)
                
                if detected_student_id:
                    # Get student details
                    student = get_student_by_id(detected_student_id)
                    
                    # Store detected student
                    st.session_state.detected_student = student
                    st.session_state.evidence_path = evidence_path
                    st.session_state.verification_stage = "verified"
                    st.rerun()
                else:
                    st.error("Wajah tidak dikenali atau verifikasi gerakan gagal. Silakan coba lagi.")
                    st.session_state.verification_stage = "initial"
                    st.rerun()
            
            # Tahap terverifikasi - menampilkan hasil dan memungkinkan konfirmasi
            elif st.session_state.verification_stage == "verified":
                student = st.session_state.detected_student
                st.success(f"Verifikasi berhasil! Mahasiswa terdeteksi: {student['name']} ({student['nim']})")
                
                if "evidence_path" in st.session_state:
                    st.image(st.session_state.evidence_path, caption="Bukti Verifikasi", width=300)
                
                if st.button("Konfirmasi Kehadiran"):
                    try:
                        message = capture_attendance(
                            student['id'], 
                            course_id, 
                            "Hadir",
                            "-",
                            st.session_state.evidence_path
                        )
                        st.success(message)
                        
                        # Reset state setelah berhasil
                        if 'evidence_path' in st.session_state:
                            del st.session_state.evidence_path
                        if 'detected_student' in st.session_state:
                            del st.session_state.detected_student
                        st.session_state.verification_stage = "initial"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal menyimpan kehadiran: {str(e)}")
                
                if st.button("Batal", key="cancel_verification"):
                    st.session_state.verification_stage = "initial"
                    if 'evidence_path' in st.session_state:
                        del st.session_state.evidence_path
                    if 'detected_student' in st.session_state:
                        del st.session_state.detected_student
                    st.rerun()

if __name__ == "__main__":
    main()