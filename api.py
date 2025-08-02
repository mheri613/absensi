from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import sqlite3
import os
import face_recognition
import pickle
import dlib
import math
from datetime import datetime
import traceback  # Add this import for detailed error tracking
from fastapi.responses import JSONResponse

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database functions
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

def get_courses():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT id, course_code, course_name, lecturer FROM courses")
    courses = [{"id": row[0], "code": row[1], "name": row[2], "lecturer": row[3]} for row in c.fetchall()]
    conn.close()
    return courses

def capture_attendance(student_id, course_id, status, reason="", evidence_path=""):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
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
    
    conn.commit()
    conn.close()
    return message

# Face recognition functions
def load_face_encodings():
    """Load face encodings from a file."""
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            encodings = pickle.load(f)
            
            # Convert old format (single encoding per student) to new format (list of encodings)
            for student_id, encoding in encodings.items():
                if not isinstance(encoding, list):
                    encodings[student_id] = [encoding]
            
            return encodings
    return {}

def extract_face_encoding(image_np):
    print("extract_face_encoding: mulai")
    print(f"extract_face_encoding: image_np.shape={image_np.shape}, dtype={image_np.dtype}")
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        print("extract_face_encoding: channel gambar bukan 3 (RGB/BGR)")
        return None
    rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    print(f"extract_face_encoding: face_locations={face_locations}")
    if len(face_locations) == 0:
        print("extract_face_encoding: tidak ada wajah")
        return None
    face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
    print("extract_face_encoding: face_encoding berhasil")
    return face_encoding

def verify_face(image_np):
    print("Masuk ke verify_face")
    face_encodings = load_face_encodings()
    if not face_encodings:
        print("Tidak ada face_encodings")
        return None, 0.0

    try:
        face_encoding = extract_face_encoding(image_np)
    except Exception as e:
        print(f"Error di extract_face_encoding: {e}")
        import traceback; traceback.print_exc()
        return None, 0.0

    if face_encoding is None:
        print("extract_face_encoding gagal")
        return None, 0.0
    
    # Compare against all registered faces
    best_match_id = None
    best_match_distance = 1.0  # Lower is better
    
    for student_id, registered_encodings in face_encodings.items():
        # Calculate face distance (lower is better) against all encodings for this student
        if isinstance(registered_encodings, list):
            # Multiple encodings per student
            distances = face_recognition.face_distance(registered_encodings, face_encoding)
            min_distance = np.min(distances) if len(distances) > 0 else 1.0
        else:
            # Single encoding (legacy format)
            min_distance = face_recognition.face_distance([registered_encodings], face_encoding)[0]
        
        # If this is a better match than previous ones
        if min_distance < best_match_distance and min_distance < 0.55:  # Threshold for considering a match
            best_match_distance = min_distance
            best_match_id = student_id
    
    if best_match_id is not None:
        # Calculate confidence percentage (higher is better)
        confidence = (1 - best_match_distance) * 100
        return best_match_id, confidence
    
    return None, 0.0

def get_face_landmarks(image_np):
    """Detect facial landmarks using dlib"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
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
    
    # Tambahkan logging untuk debugging
    print(f"Head pose angles: {angles}")
    
    return angles

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
        
        # Threshold for blink detection (dibuat lebih mudah)
        return ear < 0.25  # Sebelumnya 0.2
    elif movement_type == "left":
        # Check for horizontal movement to left (yaw)
        # Threshold untuk gerakan ke kiri (dibuat lebih mudah)
        return angles[1] > 10  # Sebelumnya 15
    elif movement_type == "right":
        # Check for horizontal movement to right (yaw)
        # Threshold untuk gerakan ke kanan (dibuat lebih mudah)
        return angles[1] < -10  # Sebelumnya -15
    return False

# Load YOLO model for face detection (gunakan cache agar tidak load berulang)
yolo_model = None
def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        import torch
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    return yolo_model

# API endpoints
@app.get("/api/courses")
async def get_courses_api():
    try:
        courses = get_courses()
        return {"success": True, "courses": courses}
    except Exception as e:
        print(f"Error in /api/courses: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/verify")
async def verify_attendance(
    course_id: int = Query(...),
    image: UploadFile = File(...)
):
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Verify face
        detected_student_id, confidence = verify_face(image_np)
        
        if detected_student_id and confidence > 60:
            # Get student details
            student = get_student_by_id(detected_student_id)
            
            # Check if already attended today
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            
            conn = sqlite3.connect("attendance.db")
            c = conn.cursor()
            c.execute("""
                SELECT id, attendance_time 
                FROM attendances 
                WHERE student_id=? 
                AND course_id=? 
                AND attendance_date=?
            """, (student['id'], course_id, date_str))
            existing = c.fetchone()
            conn.close()
            
            if existing:
                attendance_time = existing[1]
                return {
                    "success": False,
                    "message": f"Anda sudah melakukan absensi untuk mata kuliah ini pada tanggal {date_str} pukul {attendance_time}",
                    "student": {
                        "id": student['id'],
                        "nim": student['nim'],
                        "name": student['name']
                    }
                }
            
            # Return success with movement verification flag
            return {
                "success": False,
                "needs_movement_verification": True,
                "student": {
                    "id": student['id'],
                    "nim": student['nim'],
                    "name": student['name']
                },
                "confidence": confidence
            }
        else:
            raise HTTPException(status_code=400, detail="Face not recognized or confidence too low")
            
    except Exception as e:
        print(f"Error in /api/attendance/verify: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/verify-movement")
async def verify_movement_attendance(
    movement: str = Form(...),
    student_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get landmarks
        landmarks = get_face_landmarks(image_np)
        if landmarks is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        # Verify movement
        if verify_movement(landmarks, movement, image_np):
            # Get student details
            student = get_student_by_id(int(student_id))
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")
            
            # Save evidence
            evidence_filename = f"{student['nim']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            evidence_path = os.path.join("evidence", evidence_filename)
            os.makedirs("evidence", exist_ok=True)
            cv2.imwrite(evidence_path, image_np)
            
            return {
                "success": True,
                "message": "Movement verification successful"
            }
        else:
            raise HTTPException(status_code=400, detail="Movement verification failed")
            
    except Exception as e:
        print(f"Error in /api/attendance/verify-movement: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/complete")
async def complete_attendance(
    course_id: int = Query(...),
    student_id: str = Query(...),
    image: UploadFile = File(...)
):
    try:
        # Baca file gambar
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Verifikasi wajah sekali lagi untuk memastikan
        face_encodings = load_face_encodings()
        # Konversi student_id ke integer untuk pencocokan
        student_id_int = int(student_id)
        
        if student_id_int not in face_encodings or not face_encodings[student_id_int]:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Data wajah tidak ditemukan"}
            )
        
        # Verifikasi wajah
        face_encoding = extract_face_encoding(image_np)
        if face_encoding is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Wajah tidak terdeteksi"}
            )
        
        # Bandingkan dengan data wajah yang tersimpan
        student_encodings = face_encodings[student_id_int]
        if isinstance(student_encodings, list):
            face_distances = face_recognition.face_distance(student_encodings, face_encoding)
            min_distance = np.min(face_distances)
        else:
            min_distance = face_recognition.face_distance([student_encodings], face_encoding)[0]
        
        if min_distance > 0.6:  # Threshold untuk kecocokan wajah
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Wajah tidak cocok"}
            )
        
        # Simpan bukti absensi
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        evidence_filename = f"evidence_{student_id}_{timestamp}.jpg"
        evidence_path = os.path.join("evidence", evidence_filename)
        cv2.imwrite(evidence_path, image_np)
        
        # Catat kehadiran
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        
        # Cek apakah sudah ada absensi hari ini
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("""
            SELECT id FROM attendances 
            WHERE student_id = ? AND course_id = ? AND attendance_date = ?
        """, (student_id_int, course_id, today))
        
        existing = c.fetchone()
        if existing:
            # Update absensi yang sudah ada
            c.execute("""
                UPDATE attendances 
                SET status = 'Hadir', evidence_path = ?, attendance_time = ?
                WHERE id = ?
            """, (evidence_path, datetime.now().strftime("%H:%M:%S"), existing[0]))
        else:
            # Buat absensi baru
            c.execute("""
                INSERT INTO attendances 
                (student_id, course_id, attendance_date, attendance_time, status, evidence_path, admin_approved)
                VALUES (?, ?, ?, ?, 'Hadir', ?, 1)
            """, (
                student_id_int, 
                course_id, 
                today,
                datetime.now().strftime("%H:%M:%S"),
                evidence_path
            ))
        
        conn.commit()
        
        # Ambil data mahasiswa untuk response
        c.execute("SELECT name, nim FROM students WHERE id = ?", (student_id_int,))
        student = c.fetchone()
        conn.close()
        
        return {
            "success": True,
            "message": "Absensi berhasil",
            "student": {
                "id": student_id,
                "name": student[0],
                "nim": student[1]
            }
        }
        
    except Exception as e:
        print(f"Error in complete_attendance: {str(e)}")  # Tambahkan logging
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"}
        )

@app.get("/api/attendance/user/{student_id}")
async def get_attendance_by_user(student_id: int = Path(...)):
    try:
        conn = sqlite3.connect("attendance.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''
            SELECT a.attendance_date, a.attendance_time, a.status, a.course_id, c.course_name, c.course_code, s.name
            FROM attendances a
            JOIN courses c ON a.course_id = c.id
            JOIN students s ON a.student_id = s.id
            WHERE a.student_id = ?
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
        ''', (student_id,))
        rows = c.fetchall()
        attendances = [dict(row) for row in rows]
        conn.close()
        return {"success": True, "attendances": attendances}
    except Exception as e:
        print(f"Error in /api/attendance/user/{{student_id}}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/verify-test")
async def verify_attendance_test(
    course_id: int = Query(...),
    image: UploadFile = File(...)
):
    try:
        try:
            print("Menerima request verify-test")
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("Gambar berhasil dibaca")

            model = get_yolo_model()
            results = model(image_np)
            faces = results.xyxy[0].cpu().numpy()
            print(f"YOLO mendeteksi {len(faces)} wajah")
            if len(faces) == 0:
                return {"success": False, "message": "Tidak ada wajah terdeteksi oleh YOLO"}

            x1, y1, x2, y2, conf, cls = faces[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face_crop = image_np[y1:y2, x1:x2]
            print(f"face_crop.shape: {face_crop.shape}")
            print("Crop wajah berhasil")
            if face_crop is None or face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                print("Crop wajah kosong/invalid")
                return {"success": False, "message": "Crop wajah gagal"}

            detected_student_id, confidence = verify_face(face_crop)
            print(f"Hasil face recognition: {detected_student_id}, confidence: {confidence}")
            if detected_student_id and confidence > 60:
                student = get_student_by_id(detected_student_id)
                return {
                    "success": True,
                    "student": {
                        "id": student['id'],
                        "nim": student['nim'],
                        "name": student['name']
                    },
                    "confidence": confidence
                }
            else:
                return {
                    "success": False,
                    "message": "Face not recognized or confidence too low"
                }
        except Exception as e:
            print(f"Error in /api/attendance/verify-test inner: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Internal error: {str(e)}"}
    except Exception as e:
        print(f"Error in /api/attendance/verify-test outer: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Fatal error: {str(e)}"}

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000) 
    uvicorn.run(app, port=8000) 