import 'package:flutter/material.dart';
import 'package:flutter_absensi/global.dart';
import 'package:table_calendar/table_calendar.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'main.dart';
import 'package:camera/camera.dart';

class DaftarAbsensi extends StatefulWidget {
  final int? userId;
  final List<CameraDescription> cameras;
  const DaftarAbsensi({super.key, this.userId, required this.cameras});

  @override
  State<DaftarAbsensi> createState() => _DaftarAbsensiState();
}

class _DaftarAbsensiState extends State<DaftarAbsensi> {
  Map<DateTime, List<Map<String, dynamic>>> _attendanceEvents = {};
  DateTime _focusedDay = DateTime.now();
  DateTime? _selectedDay;
  bool _isLoading = true;
  List<Map<String, dynamic>> _selectedEvents = [];

  @override
  void initState() {
    super.initState();
    _fetchAttendance();
  }

  Future<void> _fetchAttendance() async {
    setState(() {
      _isLoading = true;
    });
    try {
      // TODO: Ganti endpoint di bawah ini agar sesuai dengan user yang sedang login
      // final response = await http.get(Uri.parse('http://192.168.1.4:8000/api/attendance/user/{userId}'));
      final response = await http
          .get(Uri.parse('$apiBaseUrl/api/attendance/user/${widget.userId}'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['success'] == true && data['attendances'] != null) {
          print('data = ' + data['attendances'].toString());
          Map<DateTime, List<Map<String, dynamic>>> events = {};
          for (var item in data['attendances']) {
            DateTime date = DateTime.parse(item['attendance_date']);
            date = DateTime(date.year, date.month, date.day); // normalize
            if (!events.containsKey(date)) {
              events[date] = [];
            }
            events[date]!.add(item);
          }
          setState(() {
            _attendanceEvents = events;
            DateTime today = DateTime.now();
            DateTime todayKey = DateTime(today.year, today.month, today.day);
            if (events.containsKey(todayKey)) {
              _selectedDay = todayKey;
              _focusedDay = todayKey;
              _selectedEvents = events[todayKey]!;
            } else if (events.isNotEmpty) {
              DateTime latest =
                  events.keys.reduce((a, b) => a.isAfter(b) ? a : b);
              _selectedDay = latest;
              _focusedDay = latest;
              _selectedEvents = events[latest]!;
            } else {
              _selectedDay = todayKey;
              _focusedDay = todayKey;
              _selectedEvents = [];
            }
            _isLoading = false;
          });
        } else {
          setState(() {
            _attendanceEvents = {};
            _selectedEvents = [];
            _isLoading = false;
          });
        }
      } else {
        setState(() {
          _attendanceEvents = {};
          _selectedEvents = [];
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _attendanceEvents = {};
        _selectedEvents = [];
        _isLoading = false;
      });
    }
  }

  List<Map<String, dynamic>> _getEventsForDay(DateTime day) {
    return _attendanceEvents[DateTime(day.year, day.month, day.day)] ?? [];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Daftar Absensi"),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.of(context).pushReplacement(
              MaterialPageRoute(
                builder: (context) => AttendancePage(cameras: widget.cameras),
              ),
            );
          },
        ),
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : Column(
              children: [
                TableCalendar(
                  firstDay: DateTime.utc(2020, 1, 1),
                  lastDay: DateTime.utc(2100, 12, 31),
                  focusedDay: _focusedDay,
                  selectedDayPredicate: (day) => isSameDay(_selectedDay, day),
                  eventLoader: _getEventsForDay,
                  onDaySelected: (selectedDay, focusedDay) {
                    setState(() {
                      _selectedDay = selectedDay;
                      _focusedDay = focusedDay;
                      _selectedEvents = _getEventsForDay(selectedDay);
                    });
                  },
                  calendarStyle: CalendarStyle(
                    markerDecoration: BoxDecoration(
                      color: Colors.blue,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  "Daftar Absensi pada ${_selectedDay != null ? _selectedDay!.toLocal().toString().split(' ')[0] : ''}",
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                Expanded(
                  child: _selectedEvents.isEmpty
                      ? Center(child: Text("Tidak ada absensi pada hari ini."))
                      : ListView.builder(
                          itemCount: _selectedEvents.length,
                          itemBuilder: (context, index) {
                            final event = _selectedEvents[index];
                            return ListTile(
                              leading: Icon(Icons.check_circle,
                                  color: event['status'] == 'Hadir'
                                      ? Colors.green
                                      : Colors.orange),
                              title: Text(event['course_name'] ?? '-'),
                              subtitle: Text(
                                  'Status: ${event['status']}\nWaktu: ${event['attendance_time']}'),
                            );
                          },
                        ),
                ),
              ],
            ),
    );
  }
}
