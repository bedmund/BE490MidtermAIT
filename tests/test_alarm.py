import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alarm import AlarmOutputs


class FakeTimer:
    def __init__(self, seconds, callback, args):
        self.seconds = seconds
        self.callback = callback
        self.args = args
        self.start = Mock()
        self.cancel = Mock()

    def expire(self):
        self.callback(*self.args)


class AlarmTests(unittest.TestCase):
    def test_simulation_retrigger_expiry_and_cleanup(self):
        with patch('alarm.threading.Timer', side_effect=FakeTimer), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            alarm = AlarmOutputs(seconds=5)
            alarm.trigger()
            first = alarm._timer
            alarm.trigger()
            second = alarm._timer
            first.cancel.assert_called_once()
            first.expire()  # A canceled timer racing with a new detection cannot stop it.
            self.assertTrue(alarm._active)
            second.expire()
            self.assertFalse(alarm._active)
            alarm.trigger()
            final = alarm._timer
            alarm.close()
            alarm.close()  # Idempotent cleanup.
            final.cancel.assert_called_once()
            final.expire()
            self.assertFalse(alarm._active)
        self.assertIn('light ON, buzzer ON', output.getvalue())
        self.assertIn('light OFF, buzzer OFF', output.getvalue())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            alarm.trigger()

    def test_gpio_outputs_and_error_cleanup(self):
        led, buzzer = Mock(), Mock()
        gpio = SimpleNamespace(LED=Mock(return_value=led), Buzzer=Mock(return_value=buzzer))
        with patch.dict('sys.modules', gpiozero=gpio), \
                patch('alarm.threading.Timer', side_effect=FakeTimer), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                with AlarmOutputs('gpio', 17, 27) as alarm:
                    gpio.LED.assert_called_once_with(17, initial_value=False)
                    gpio.Buzzer.assert_called_once_with(27, initial_value=False)
                    alarm.trigger()
                    self.assertTrue(led.value)
                    self.assertTrue(buzzer.value)
                    alarm._timer.expire()
                    self.assertFalse(led.value)
                    self.assertFalse(buzzer.value)
                    alarm.trigger()
                    raise RuntimeError('capture failed')
            self.assertFalse(led.value)
            self.assertFalse(buzzer.value)
            led.close.assert_called_once()
            buzzer.close.assert_called_once()

    def test_partial_gpio_initialization_is_released(self):
        led = Mock()
        gpio = SimpleNamespace(LED=Mock(return_value=led), Buzzer=Mock(side_effect=RuntimeError('no pin')))
        with patch.dict('sys.modules', gpiozero=gpio):
            with self.assertRaisesRegex(RuntimeError, 'Cannot initialize GPIO'):
                AlarmOutputs('gpio')
        led.close.assert_called_once()

    def test_invalid_configuration(self):
        for kwargs in ({'seconds': 0}, {'seconds': float('nan')}, {'led_pin': 27}, {'led_pin': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AlarmOutputs(**kwargs)

    def test_timer_output_error_is_reported_to_monitoring_loop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            alarm = AlarmOutputs()
            with patch.object(alarm, '_set_state', side_effect=RuntimeError('pin failed')):
                alarm._expire(alarm._generation)
                with self.assertRaisesRegex(RuntimeError, 'Alarm output failed'):
                    alarm.check()
            alarm.close()


if __name__ == '__main__':
    unittest.main()
