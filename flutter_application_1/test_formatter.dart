import 'package:intl/intl.dart';

class DateFormatter {
  static Map<String, dynamic>? getCurrentPeriod(List<dynamic> timetable) {
    if (timetable.isEmpty) return null;

    final now = DateTime.now();
    final currentDay = DateFormat('EEEE').format(now).toLowerCase();

    for (var period in timetable) {
      final dayOfWeek = (period['day_of_week'] as String?)?.toLowerCase() ?? '';
      if (dayOfWeek != currentDay) continue;

      // Safe cast
      final startStr = period['start_time']?.toString();
      final endStr = period['end_time']?.toString();

      if (startStr == null || endStr == null) {
        print("Skipping period ${period['period_no']} due to null time");
        continue;
      }

      try {
        DateTime? parseTime(String timeStr) {
          timeStr = timeStr.trim();
          try {
            if (timeStr.toLowerCase().contains('m')) {
              final format = DateFormat('h:mm a');
              final parsed = format.parse(timeStr);
              return DateTime(now.year, now.month, now.day, parsed.hour, parsed.minute);
            } else {
              final parts = timeStr.split(':');
              if (parts.length >= 2) {
                return DateTime(now.year, now.month, now.day, int.parse(parts[0]), int.parse(parts[1]));
              }
            }
          } catch (e) {
             print("Parse Error: $e");
          }
          return null;
        }

        DateTime? startTime = parseTime(startStr);
        DateTime? endTime = parseTime(endStr);
        DateTime currentTime = DateTime.now();

        if (startTime != null && endTime != null) {
          if (endTime.isBefore(startTime) || endTime.isAtSameMomentAs(startTime)) {
            endTime = endTime.add(const Duration(days: 1));
            if (currentTime.isBefore(startTime)) {
              currentTime = currentTime.add(const Duration(days: 1));
            }
          }

          if (currentTime.compareTo(startTime) >= 0 && currentTime.compareTo(endTime) <= 0) {
            String formatSafe(DateTime dt) => DateFormat('h:mm a').format(dt);
            
            return {
              'no': period['period_no'].toString(),
              'start': formatSafe(startTime),
              'end': formatSafe(endTime),
              'subject_name': period['subject_name']?.toString() ?? 'Subject',
            };
          }
        }
      } catch (e) {
        print("Loop error: $e");
      }
    }

    return null;
  }
}

void main() {
  final timetable = [
    {
      "day_of_week": DateFormat('EEEE').format(DateTime.now()),
      "id": 81,
      "period_no": 2,
      "subject": "CIA",
      // Notice missing start_time
    },
    {
      "day_of_week": DateFormat('EEEE').format(DateTime.now()),
      "id": 1,
      "period_no": 5,
      "subject_name": "Network Essentials ",
      "start_time": "21:20:00",
      "end_time": "02:15:00" // spans over midnight!
    }
  ];

  final result = DateFormatter.getCurrentPeriod(timetable);
  print(result);
}
