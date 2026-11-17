import contextlib
import io
from pathlib import Path
import smtplib
import tempfile
import unittest
from unittest.mock import patch

import email_alerts as mail


class EmailAlertTests(unittest.TestCase):
    def test_commented_config_missing_fields_and_configured_values(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(mail, '__file__', str(Path(directory) / 'email_alerts.py')):
            with self.assertRaisesRegex(ValueError, 'Copy email_config'):
                mail.load_email_config()
            config = Path(directory) / 'email_config.py'
            config.write_text('# GMAIL_SENDER = "dummy@gmail.com"\n')
            with self.assertRaisesRegex(ValueError, 'Set GMAIL_SENDER'):
                mail.load_email_config()
            config.write_text('GMAIL_SENDER="demo@gmail.com"\nEMAIL_RECIPIENT="demo@icloud.com"\n'
                              'GMAIL_APP_PASSWORD="abcd efgh ijkl mnop"\n')
            self.assertEqual(mail.load_email_config(), ('demo@gmail.com', 'demo@icloud.com', 'abcdefghijklmnop'))

    def test_cooldown_and_bounded_queue(self):
        with patch.object(mail, 'load_email_config', return_value=('demo@gmail.com', 'demo@icloud.com', 'secret')), \
                patch('email_alerts.threading.Thread'), \
                patch('email_alerts.time.monotonic', side_effect=[100, 101, 160, 161, 220]), \
                contextlib.redirect_stdout(io.StringIO()):
            with mail.EmailAlerts(cooldown=60) as alerts:
                self.assertTrue(alerts.notify(0.9, 3, 6))
                first = alerts._queue.get_nowait()
                self.assertEqual(first['To'], 'demo@icloud.com')
                self.assertIn('3.0-6.0 seconds', first.get_content())
                self.assertFalse(alerts.notify(0.9, 6, 9))
                self.assertTrue(alerts.notify(0.9, 60, 63))
                self.assertFalse(alerts.notify(0.9, 63, 66))
                self.assertFalse(alerts.notify(0.9, 120, 123))  # Queue still contains the prior message.
            self.assertFalse(alerts.notify(0.9, 123, 126))

    def test_tls_smtp_authentication_and_message(self):
        with patch.object(mail, 'load_email_config', return_value=('demo@gmail.com', 'demo@icloud.com', 'secret')), \
                patch('email_alerts.threading.Thread'), \
                patch('email_alerts.smtplib.SMTP_SSL') as smtp_factory:
            with mail.EmailAlerts() as alerts:
                alerts.notify(0.85, 0, 3)
                message = alerts._queue.get_nowait()
                alerts._send(message)
            args, kwargs = smtp_factory.call_args
            self.assertEqual(args, ('smtp.gmail.com', 465))
            self.assertEqual(kwargs['timeout'], 10)
            self.assertTrue(kwargs['context'].check_hostname)
            smtp = smtp_factory.return_value.__enter__.return_value
            smtp.login.assert_called_once_with('demo@gmail.com', 'secret')
            smtp.send_message.assert_called_once_with(message)
            self.assertNotIn('secret', message.as_string())

    def test_worker_failure_is_contained_and_does_not_log_credentials(self):
        with patch.object(mail, 'load_email_config', return_value=('demo@gmail.com', 'demo@icloud.com', 'secret')), \
                patch('email_alerts.threading.Thread'), contextlib.redirect_stdout(io.StringIO()) as output:
            with mail.EmailAlerts() as alerts:
                alerts.notify(0.9, 0, 3)

                def fail(message):
                    alerts._stop.set()
                    raise smtplib.SMTPAuthenticationError(535, b'secret details')

                with patch.object(alerts, '_send', side_effect=fail):
                    alerts._worker()
            self.assertIn('Email failed (SMTPAuthenticationError)', output.getvalue())
            self.assertNotIn('secret', output.getvalue())


if __name__ == '__main__':
    unittest.main()
