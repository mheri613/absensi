import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_absensi/global.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:math' as math;
import 'daftar_absensi.dart';

// API Configuration

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Attendance System',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: AttendancePage(cameras: cameras),
    );
  }
}

class AttendancePage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const AttendancePage({super.key, required this.cameras});

  @override
  State<AttendancePage> createState() => _AttendancePageState();
}

class _AttendancePageState extends State<AttendancePage> {
  late CameraController _controller;
  List<Map<String, dynamic>> courses = [];
  String? selectedCourse;
  String statusMessage = 'Siap untuk absensi';
  bool isCameraInitialized = false;
  bool isLoadingCourses = true;
  bool isProcessing = false;
  Timer? _attendanceTimer;
  bool isAutoAttendanceEnabled = true;

  // Verifikasi gerakan
  bool isVerifyingMovement = false;
  String? currentMovement;
  int verificationAttempts = 0;
  final int maxVerificationAttempts = 3;
  bool isFaceVerified = false;
  String? verifiedStudentId;
  double? faceConfidence;
  int? verifiedCourseId;

  // Gerakan yang tersedia
  final List<String> availableMovements = ['blink', 'left', 'right'];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    loadCourses();
    toggleAutoAttendance();
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) return;

    // Find front camera
    final frontCamera = widget.cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras[0],
    );

    _controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await _controller.initialize();
      await SystemChrome.setPreferredOrientations([
        DeviceOrientation.portraitUp,
      ]);
      setState(() {
        isCameraInitialized = true;
      });
    } catch (e) {
      setState(() {
        statusMessage = 'Gagal menginisialisasi kamera: $e';
      });
    }
  }

  Future<void> _refreshCamera() async {
    setState(() {
      isCameraInitialized = false;
    });
    await _controller.dispose();
    await _initializeCamera();
  }

  Future<void> loadCourses() async {
    try {
      final response = await http.get(
        Uri.parse('$apiBaseUrl/api/courses'),
      );
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['success']) {
          setState(() {
            courses = List<Map<String, dynamic>>.from(data['courses']);
            if (courses.isNotEmpty) {
              selectedCourse = '${courses[0]['code']} - ${courses[0]['name']}';
            }
            isLoadingCourses = false;
          });
        } else {
          setState(() {
            statusMessage = 'Gagal memuat daftar mata kuliah';
            isLoadingCourses = false;
          });
        }
      } else {
        setState(() {
          statusMessage =
              'Gagal memuat daftar mata kuliah: ${response.statusCode}';
          isLoadingCourses = false;
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Error memuat daftar mata kuliah: $e';
        isLoadingCourses = false;
      });
    }
  }

  String getMovementInstruction(String movement) {
    switch (movement) {
      case 'blink':
        return 'Kedipkan mata Anda';
      case 'left':
        return 'Lihat ke kiri';
      case 'right':
        return 'Lihat ke kanan';
      default:
        return '';
    }
  }

  void toggleAutoAttendance() {
    setState(() {
      isAutoAttendanceEnabled = !isAutoAttendanceEnabled;
      if (isAutoAttendanceEnabled) {
        _attendanceTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
          if (!isProcessing && !isVerifyingMovement && selectedCourse != null) {
            takeAttendance();
          }
        });
        statusMessage = 'Absensi otomatis aktif';
      } else {
        _attendanceTimer?.cancel();
        statusMessage = 'Absensi otomatis nonaktif';
      }
    });
  }

  Future<void> verifyMovement() async {
    if (!isFaceVerified ||
        currentMovement == null ||
        verifiedStudentId == null ||
        verifiedCourseId == null) return;

    setState(() {
      isVerifyingMovement = true;
      statusMessage =
          'Verifikasi gerakan: ${getMovementInstruction(currentMovement!)}';
    });

    String? tempPath;
    try {
      final XFile image = await _controller.takePicture();
      final tempDir = await getTemporaryDirectory();
      tempPath =
          '${tempDir.path}/temp_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(image.path).copy(tempPath);

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBaseUrl/api/attendance/verify-movement'),
      );

      request.fields['movement'] = currentMovement!;
      request.fields['student_id'] = verifiedStudentId!;
      request.files.add(await http.MultipartFile.fromPath('image', tempPath));

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final data = json.decode(responseData);

      if (response.statusCode == 200 && data['success']) {
        // Verifikasi gerakan berhasil, lanjutkan dengan absensi
        await completeAttendance(verifiedStudentId!, verifiedCourseId!);
      } else {
        setState(() {
          verificationAttempts++;
          if (verificationAttempts >= maxVerificationAttempts) {
            // Reset semua state verifikasi
            isVerifyingMovement = false;
            isFaceVerified = false;
            verifiedStudentId = null;
            verifiedCourseId = null;
            currentMovement = null;
            verificationAttempts = 0;
            statusMessage = 'Verifikasi gagal. Silakan coba lagi.';
          } else {
            // Pilih gerakan baru
            currentMovement = availableMovements[
                math.Random().nextInt(availableMovements.length)];
            statusMessage =
                'Gerakan tidak sesuai. Silakan coba lagi: ${getMovementInstruction(currentMovement!)}';
          }
        });

        // Lanjutkan verifikasi dengan gerakan baru
        if (verificationAttempts < maxVerificationAttempts) {
          await Future.delayed(const Duration(
              seconds: 1)); // Tunggu sebentar sebelum mencoba lagi
          verifyMovement();
        }
      }
    } catch (e) {
      setState(() {
        isVerifyingMovement = false;
        statusMessage = 'Error: $e';
      });
    } finally {
      // Hapus file temporary jika ada
      if (tempPath != null) {
        try {
          final file = File(tempPath);
          if (await file.exists()) {
            await file.delete();
          }
        } catch (e) {
          // Abaikan error saat menghapus file
          print('Error deleting temp file: $e');
        }
      }
    }
  }

  Future<void> completeAttendance(String studentId, int courseId) async {
    String? tempPath;
    try {
      final XFile image = await _controller.takePicture();
      final tempDir = await getTemporaryDirectory();
      tempPath =
          '${tempDir.path}/temp_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(image.path).copy(tempPath);

      final request = http.MultipartRequest(
        'POST',
        Uri.parse(
            '$apiBaseUrl/api/attendance/complete?course_id=$courseId&student_id=$studentId'),
      );

      request.files.add(await http.MultipartFile.fromPath('image', tempPath));

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final data = json.decode(responseData);

      if (response.statusCode == 200 && data['success']) {
        setState(() {
          statusMessage = 'Absensi berhasil untuk ${data['student']['name']}';
          isAutoAttendanceEnabled = false;
          _attendanceTimer?.cancel();
          // Reset semua state verifikasi
          isVerifyingMovement = false;
          isFaceVerified = false;
          verifiedStudentId = null;
          verifiedCourseId = null;
          currentMovement = null;
          verificationAttempts = 0;
          _refreshCamera();
        });
        Future.delayed(const Duration(milliseconds: 500), () {
          final userId = int.tryParse(data['student']['id'].toString());
          if (userId != null) {
            Navigator.of(context).pushReplacement(
              MaterialPageRoute(
                builder: (context) => DaftarAbsensi(
                  userId: userId,
                  cameras: widget.cameras,
                ),
              ),
            );
          } else {
            setState(() {
              statusMessage = 'Error: Invalid student ID format';
              isProcessing = false;
            });
          }
        });
      } else {
        setState(() {
          statusMessage = data['message'] ?? 'Gagal melakukan absensi';
          // Reset state verifikasi
          isVerifyingMovement = false;
          isFaceVerified = false;
          verifiedStudentId = null;
          verifiedCourseId = null;
          currentMovement = null;
          verificationAttempts = 0;
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Error: $e';
        // Reset state verifikasi
        isVerifyingMovement = false;
        isFaceVerified = false;
        verifiedStudentId = null;
        verifiedCourseId = null;
        currentMovement = null;
        verificationAttempts = 0;
      });
    } finally {
      // Hapus file temporary jika ada
      if (tempPath != null) {
        try {
          final file = File(tempPath);
          if (await file.exists()) {
            await file.delete();
          }
        } catch (e) {
          // Abaikan error saat menghapus file
          print('Error deleting temp file: $e');
        }
      }
    }
  }

  Future<void> takeAttendance() async {
    if (selectedCourse == null) {
      setState(() {
        statusMessage = 'Silakan pilih mata kuliah terlebih dahulu';
      });
      return;
    }

    if (isProcessing || isVerifyingMovement) return;

    setState(() {
      isProcessing = true;
      statusMessage = 'Memproses absensi...';
    });

    String? tempPath;
    try {
      final XFile image = await _controller.takePicture();
      final tempDir = await getTemporaryDirectory();
      tempPath =
          '${tempDir.path}/temp_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(image.path).copy(tempPath);

      final courseId = courses.firstWhere(
        (course) => '${course['code']} - ${course['name']}' == selectedCourse,
      )['id'];

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBaseUrl/api/attendance/verify?course_id=$courseId'),
      );

      request.files.add(await http.MultipartFile.fromPath('image', tempPath));

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final data = json.decode(responseData);

      if (response.statusCode == 200) {
        print('data = ' + data.toString());
        if (data.containsKey('message') &&
            data['message'].contains('sudah melakukan absensi')) {
          print('sudah melakukan absensi2');
          Future.delayed(const Duration(milliseconds: 500), () {
            final userId = int.tryParse(data['student']['id'].toString());
            if (userId != null) {
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(
                  builder: (context) => DaftarAbsensi(
                    userId: userId,
                    cameras: widget.cameras,
                  ),
                ),
              );
            } else {
              setState(() {
                statusMessage = 'Error: Invalid student ID format';
                isProcessing = false;
              });
            }
          });
        }
        if (data['success']) {
          setState(() {
            statusMessage = 'Absensi berhasil untuk ${data['student']['name']}';
            isAutoAttendanceEnabled = false;
            _attendanceTimer?.cancel();
            _refreshCamera();
          });
        } else if (data['needs_movement_verification'] == true) {
          setState(() {
            isFaceVerified = true;
            verifiedStudentId = data['student']['id'].toString();
            verifiedCourseId = courseId;
            faceConfidence = data['confidence'].toDouble();
            currentMovement = availableMovements[
                math.Random().nextInt(availableMovements.length)];
            verificationAttempts = 0;
            isProcessing = false;
          });
          verifyMovement();
        } else {
          setState(() {
            statusMessage = data['message'] ?? 'Gagal melakukan absensi';
            isProcessing = false;
          });
        }
      } else {
        setState(() {
          if (data['detail'] != null) {
            statusMessage = data['detail'];
          } else if (data['message'] != null) {
            statusMessage = data['message'];
          } else {
            statusMessage =
                'Gagal melakukan absensi. Kode status: ${response.statusCode}';
          }
          isProcessing = false;
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Error: $e';
        isProcessing = false;
      });
    } finally {
      // Hapus file temporary jika ada
      if (tempPath != null) {
        try {
          final file = File(tempPath);
          if (await file.exists()) {
            await file.delete();
          }
        } catch (e) {
          // Abaikan error saat menghapus file
          print('Error deleting temp file: $e');
        }
      }
    }
  }

  @override
  void dispose() {
    _attendanceTimer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sistem Absensi'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (isLoadingCourses)
              const Center(child: CircularProgressIndicator())
            else if (courses.isEmpty)
              const Center(child: Text('Tidak ada mata kuliah yang tersedia'))
            else
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8.0),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: DropdownButtonFormField<String>(
                  value: selectedCourse,
                  decoration: const InputDecoration(
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(horizontal: 16),
                    hintText: 'Pilih Mata Kuliah',
                  ),
                  isExpanded: true,
                  items: courses.map((course) {
                    final courseText = '${course['code']} - ${course['name']}';
                    return DropdownMenuItem<String>(
                      value: courseText,
                      child: Text(courseText),
                    );
                  }).toList(),
                  onChanged: (String? value) {
                    setState(() {
                      selectedCourse = value;
                    });
                  },
                ),
              ),
            const SizedBox(height: 16),
            if (isCameraInitialized)
              Expanded(
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    CameraPreview(_controller),
                    if (isProcessing || isVerifyingMovement)
                      Container(
                        color: Colors.black54,
                        child: const Center(
                          child: CircularProgressIndicator(
                            color: Colors.white,
                          ),
                        ),
                      ),
                    if (isVerifyingMovement && currentMovement != null)
                      Positioned(
                        top: 20,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            getMovementInstruction(currentMovement!),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              )
            else
              const Expanded(
                child: Center(
                  child: CircularProgressIndicator(),
                ),
              ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                statusMessage,
                style: Theme.of(context).textTheme.bodyLarge,
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: SizedBox(
                    height: 50,
                    child: ElevatedButton(
                      onPressed: toggleAutoAttendance,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: isAutoAttendanceEnabled
                            ? Colors.red
                            : Theme.of(context).colorScheme.primary,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        isAutoAttendanceEnabled
                            ? 'Nonaktifkan Auto'
                            : 'Aktifkan Auto',
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: SizedBox(
                    height: 50,
                    child: ElevatedButton(
                      onPressed: isProcessing ? null : takeAttendance,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Theme.of(context).colorScheme.primary,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        isProcessing ? 'Memproses...' : 'Ambil Absensi',
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
