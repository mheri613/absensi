import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import sqlite3
import os
import pandas as pd
import bcrypt
import face_recognition
import pickle
import dlib
import urllib.request
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
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

def get_student_by_id(student_id):
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
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    if student_id:
        c.execute("""UPDATE students SET nim=?, name=?, email=?, password=?, class=? WHERE id=?""", (nim, name, email, hashed_pw, class_name, student_id))
    else:
        c.execute("""INSERT INTO students (nim, name, email, password, class) VALUES (?, ?, ?, ?, ?)""", (nim, name, email, hashed_pw, class_name))
    conn.commit()
    conn.close()

def get_course_by_id(course_id):
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
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    if course_id:
        c.execute("""UPDATE courses SET course_code=?, course_name=?, lecturer=? WHERE id=?""", (code, name, lecturer, course_id))
    else:
        c.execute("""INSERT INTO courses (course_code, course_name, lecturer) VALUES (?, ?, ?)""", (code, name, lecturer))
    conn.commit()
    conn.close()

def update_attendance_status(attendance_id, status, approved):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("""UPDATE attendances SET status=?, admin_approved=? WHERE id=?""", (status, approved, attendance_id))
    conn.commit()
    conn.close()

def create_db():
    db_exists = os.path.exists("attendance.db")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT, nim TEXT UNIQUE, name TEXT, email TEXT UNIQUE, password TEXT, class TEXT, face_image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS courses (id INTEGER PRIMARY KEY AUTOINCREMENT, course_code TEXT UNIQUE, course_name TEXT, lecturer TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendances (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, course_id INTEGER, attendance_date TEXT, attendance_time TEXT, status TEXT, reason TEXT, evidence_path TEXT, admin_approved INTEGER DEFAULT 0, FOREIGN KEY(student_id) REFERENCES students(id), FOREIGN KEY(course_id) REFERENCES courses(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS admins (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password TEXT)''')
    if not db_exists:
        c.execute("INSERT OR IGNORE INTO admins (email, password) VALUES (?, ?)", ("admin@example.com", bcrypt.hashpw("admin".encode('utf-8'), bcrypt.gensalt())))
        c.execute("INSERT OR IGNORE INTO students (nim, name, email, password, class, face_image_path) VALUES (?, ?, ?, ?, ?, ?)", ("12345678", "Siswa Test", "siswa@example.com", bcrypt.hashpw("siswa".encode('utf-8'), bcrypt.gensalt()), "TI-1A", ""))
        c.execute("INSERT OR IGNORE INTO students (nim, name, email, password, class, face_image_path) VALUES (?, ?, ?, ?, ?, ?)", ("31231321", "Siswa Test 2", "siswa2@example.com", bcrypt.hashpw("siswa2".encode('utf-8'), bcrypt.gensalt()), "TI-1A", ""))
        c.execute("INSERT OR IGNORE INTO courses (course_code, course_name, lecturer) VALUES (?, ?, ?)", ("IF101", "Pengantar Informatika", "Dr. Budi"))
        c.execute("INSERT OR IGNORE INTO courses (course_code, course_name, lecturer) VALUES (?, ?, ?)", ("IF102", "Algoritma dan Pemrograman", "Dr. Ani"))
    conn.commit()
    conn.close()

def capture_attendance(student_id, course_id, status, reason="", evidence_path=""):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    print(f"Menangkap kehadiran: student_id={student_id}, course_id={course_id}, status={status}, evidence_path={evidence_path}")
    c.execute("SELECT id FROM attendances WHERE student_id=? AND course_id=? AND attendance_date=?", (student_id, course_id, date_str))
    existing = c.fetchone()
    if existing:
        c.execute("UPDATE attendances SET status=?, reason=?, evidence_path=?, attendance_time=? WHERE id=?", (status, reason, evidence_path, time_str, existing[0]))
        message = f"Kehadiran diperbarui: {date_str} {time_str} - {status}"
    else:
        c.execute("""INSERT INTO attendances (student_id, course_id, attendance_date, attendance_time, status, reason, evidence_path, admin_approved) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (student_id, course_id, date_str, time_str, status, reason, evidence_path, 1 if status == "Hadir" else 0))
        message = f"Kehadiran dicatat: {date_str} {time_str} - {status}"
    c.execute("SELECT * FROM attendances WHERE student_id=? AND course_id=? AND attendance_date=?", (student_id, course_id, date_str))
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
    SELECT a.id, a.attendance_date, a.attendance_time, a.status, a.reason, a.admin_approved, s.nim, s.name, s.class, c.course_code, c.course_name, c.lecturer FROM attendances a JOIN students s ON a.student_id = s.id JOIN courses c ON a.course_id = c.id WHERE 1=1
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
    model = load_yolo_model()
    results = model(image_np)
    faces = results.xyxy[0].cpu().numpy()
    if len(faces) > 0:
        x1, y1, x2, y2, conf, cls = faces[0]
        face_image = image_np[int(y1):int(y2), int(x1):int(x2)]
        return face_image
    return None

def extract_face_encoding(image_np):
    face_locations = face_recognition.face_locations(image_np)
    if len(face_locations) == 0:
        return None
    face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
    return face_encoding

def compare_faces(face_encoding, reference_encoding):
    return face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.45)[0]

def load_face_encodings():
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            encodings = pickle.load(f)
            for student_id, encoding in encodings.items():
                if not isinstance(encoding, list):
                    encodings[student_id] = [encoding]
            return encodings
    return {}

def save_face_encodings(face_encodings):
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(face_encodings, f)

def augment_image(image_np):
    augmented_images = [image_np]
    for angle in [-15, -10, -5, 5, 10, 15]:
        rotated = ndimage.rotate(image_np, angle, reshape=False, mode='nearest')
        augmented_images.append(rotated)
    for factor in [0.8, 1.2]:
        brightness = cv2.convertScaleAbs(image_np, alpha=factor, beta=0)
        augmented_images.append(brightness)
    for shift in [-20, 20]:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        shifted = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]))
        augmented_images.append(shifted)
    noise = np.copy(image_np)
    noise_factor = 0.05
    noise = noise + noise_factor * np.random.normal(0, 1, noise.shape)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    augmented_images.append(noise)
    return augmented_images

def register_face(student_id, image_np):
    original_face_encoding = extract_face_encoding(image_np)
    if original_face_encoding is None:
        return False, 0
    face_encodings = load_face_encodings()
    if student_id not in face_encodings:
        face_encodings[student_id] = []
    face_encodings[student_id].append(original_face_encoding)
    augmented_images = augment_image(image_np)
    augmentation_count = 0
    for aug_img in augmented_images[1:]:
        aug_face_encoding = extract_face_encoding(aug_img)
        if aug_face_encoding is not None:
            face_encodings[student_id].append(aug_face_encoding)
            augmentation_count += 1
    save_face_encodings(face_encodings)
    return True, augmentation_count

def verify_face(image_np, student_id):
    face_encodings = load_face_encodings()
    if student_id not in face_encodings or not face_encodings[student_id]:
        return False, 0.0
    face_encoding = extract_face_encoding(image_np)
    if face_encoding is None:
        return False, 0.0
    student_encodings = face_encodings[student_id]
    face_distances = face_recognition.face_distance(student_encodings, face_encoding)
    best_match_idx = np.argmin(face_distances)
    best_distance = face_distances[best_match_idx]
    accuracy = (1 - best_distance) * 100
    match = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.6)
    return any(match), accuracy

def download_dlib_model():
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
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    face_encodings = load_face_encodings()
    if student_id in face_encodings:
        del face_encodings[student_id]
        save_face_encodings(face_encodings)
    c.execute("DELETE FROM attendances WHERE student_id=?", (student_id,))
    c.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    conn.close()
    return True

def delete_course(course_id):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("DELETE FROM attendances WHERE course_id=?", (course_id,))
    c.execute("DELETE FROM courses WHERE id=?", (course_id,))
    conn.commit()
    conn.close()
    return True

# --- ABSENSI TANPA VERIFIKASI GERAKAN ---
def absensi_face_verification(face_encodings):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    detected_student_id = None
    evidence_path = ""
    match_confidence = 0.0
    max_verification_attempts = 3
    verification_attempts = 0
    matched_student_id = None
    best_match_distance = 1.0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            st.error("Tidak dapat mengakses kamera")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = load_yolo_model()(frame_rgb)
        faces = results.xyxy[0].cpu().numpy()
        if len(faces) > 0:
            x1, y1, x2, y2, conf, cls = faces[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Gunakan frame_rgb penuh, bukan crop
            face_encoding = extract_face_encoding(frame_rgb)
            if face_encoding is not None:
                verification_attempts += 1
                for student_id, registered_encodings in face_encodings.items():
                    if isinstance(registered_encodings, list):
                        distances = face_recognition.face_distance(registered_encodings, face_encoding)
                        min_distance = np.min(distances) if len(distances) > 0 else 1.0
                    else:
                        min_distance = face_recognition.face_distance([registered_encodings], face_encoding)[0]
                    if min_distance < best_match_distance and min_distance < 0.55:
                        best_match_distance = min_distance
                        matched_student_id = student_id
                if matched_student_id is not None:
                    confidence = (1 - best_match_distance) * 100
                    match_confidence = confidence
                    cv2.putText(frame, f"Match: {int(confidence)}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if verification_attempts >= max_verification_attempts and match_confidence > 60:
                        detected_student_id = matched_student_id
                        student = get_student_by_id(detected_student_id)
                        evidence_filename = f"{student['nim']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                        evidence_path = os.path.join("evidence", evidence_filename)
                        cv2.imwrite(evidence_path, frame)
                        cap.release()
                        stframe.image(frame, channels="BGR", use_container_width=True)
                        return detected_student_id, evidence_path
                else:
                    cv2.putText(frame, "Wajah tidak dikenali", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Wajah tidak terdeteksi", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stframe.image(frame, channels="BGR", use_container_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return None, ""

# --- END ABSENSI TANPA VERIFIKASI GERAKAN ---

# --- Hanya halaman absensi mahasiswa ---
def main():
    create_db()
    download_dlib_model()
    st.title("Absensi Mahasiswa - Face Recognition Tanpa Verifikasi Gerakan")
    st.write("Silakan lakukan absensi kehadiran dengan verifikasi wajah.")

    # Pilih mata kuliah
    courses = get_courses()
    if not courses:
        st.warning("Belum ada data mata kuliah.")
        return
    course_options = {f"{c['code']} - {c['name']}": c['id'] for c in courses}
    selected_course = st.selectbox("Pilih Mata Kuliah", options=list(course_options.keys()))
    course_id = course_options[selected_course]

    # Pilih status kehadiran
    attendance_status = st.radio("Status Kehadiran", ["Hadir", "Izin", "Sakit"])

    if attendance_status in ["Izin", "Sakit"]:
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
            student = get_student_by_nim(nim)
            if not student:
                st.error("NIM tidak ditemukan dalam database")
                return
            message = capture_attendance(student['id'], course_id, attendance_status, reason, evidence_path)
            st.success(f"{message}\nPermintaan Anda akan diverifikasi oleh admin.")
    elif attendance_status == "Hadir":
        if 'verification_stage' not in st.session_state:
            st.session_state.verification_stage = "initial"
        if st.session_state.verification_stage == "initial":
            if st.button("Mulai Verifikasi Wajah"):
                st.session_state.verification_stage = "verifying"
                st.rerun()
        elif st.session_state.verification_stage == "verifying":
            st.write("Mendeteksi wajah...")
            face_encodings = load_face_encodings()
            detected_student_id, evidence_path = absensi_face_verification(face_encodings)
            if detected_student_id:
                student = get_student_by_id(detected_student_id)
                st.session_state.detected_student = student
                st.session_state.evidence_path = evidence_path
                st.session_state.verification_stage = "verified"
                st.rerun()
            else:
                st.error("Wajah tidak dikenali. Silakan coba lagi.")
                st.session_state.verification_stage = "initial"
                st.rerun()
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