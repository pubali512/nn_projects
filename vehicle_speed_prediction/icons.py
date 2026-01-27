# icons.py
# A simple icon + emoji utility library for Python

class Emoji:
    # Basic emojis
    SMILE = "😊"
    FIRE = "🔥"
    HEART = "❤️"
    ROCKET = "🚀"
    CHECK = "✅"
    CROSS = "❌"
    WARN = "⚠️"
    INFO = "ℹ️"
    PYTHON = "🐍"
    STAR = "⭐"
    SUCCESS = "🎉"
    HOURGLASS = "⏳"
    PIN = "📌"
    SAVE ="💾"
    STEPS = "🔄"
    LOAD = "📥"
    START = "▶️"
    DEVICE = "💻"
    


class CLI:
    # Console-safe symbols (ASCII/Unicode)
    CHECK = "[✓]"
    CROSS = "[✗]"
    WARN = "[!]"
    INFO = "[i]"
    DOT = "•"
    ARROW = "→"
    STAR = "*"
    SPINNER = "|/-\\"  # use in a loop


class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    WHITE = "\033[37m"



# Utility functions ------------------------------------------------

def device(msg):
    return f"{Color.WHITE}{Emoji.DEVICE} {msg}{Color.RESET}"

def start(msg):
    return f"{Color.WHITE}{Emoji.START} {msg}{Color.RESET}"

def steps(msg):
    return f"{Color.WHITE}{Emoji.STEPS} {msg}{Color.RESET}"

def info(msg):
    return f"{Color.WHITE}{Emoji.INFO} {msg}{Color.RESET}"

def save(msg):
    return f"{Color.WHITE}{Emoji.SAVE} {msg}{Color.RESET}"

def check(msg):
    return f"{Color.WHITE}{Emoji.CHECK} {msg}{Color.RESET}"

def banner(msg):
    return f"{Color.BOLD}{Emoji.STAR} {msg} {Emoji.STAR}{Color.RESET}"


# Optional: spinner generator --------------------------------------

def spinner():
    """Generator that cycles through spinner symbols."""
    while True:
        for ch in CLI.SPINNER:
            yield ch
