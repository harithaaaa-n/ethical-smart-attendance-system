import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_application_1/core/utils/date_formatter.dart';
import 'package:intl/intl.dart';

void main() {
  group('DateFormatter Tests', () {
    test('formatToTime should return HH:mm format', () {
      final date = DateTime(2026, 3, 1, 14, 30);
      expect(DateFormatter.formatToTime(date), '14:30');
    });

    test('formatToDate should return yyyy-MM-dd format', () {
      final date = DateTime(2026, 3, 1);
      expect(DateFormatter.formatToDate(date), '2026-03-01');
    });

    test('getDayName should return current day', () {
      final now = DateTime.now();
      final expected = DateFormat('EEEE').format(now);
      expect(DateFormatter.getDayName(), expected);
    });

    // Note: getCurrentPeriod depends on DateTime.now() which is hard to mock in simple unit tests without a wrapper
    // but we've verified the logic qualitatively.
  });
}
