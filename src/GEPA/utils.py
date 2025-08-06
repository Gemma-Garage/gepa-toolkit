import time


def log_message(message, type='info'):
    """Helper to format log messages with a timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    if type == 'success':
        return f"[{timestamp}] ✅ SUCCESS: {message}"
    if type == 'fail':
        return f"[{timestamp}] ❌ FAIL: {message}"
    if type == 'best':
        return f"[{timestamp}] ⭐ BEST: {message}"
    return f"[{timestamp}] ℹ️ INFO: {message}"
    