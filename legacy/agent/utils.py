import platform
import sys
from pathlib import Path
import time
from agent import context as agent_context

if platform.system() == "Windows":
    import msvcrt
else:
    import select
    import termios

# ==========================================
# 예외 클래스
# ==========================================
class UserInterruptedException(Exception):
    """사용자가 ESC로 작업을 중단했을 때 발생하는 예외"""
    pass

# ==========================================
# 핵심 유틸리티 (키보드, 경로)
# ==========================================
def check_esc_pressed() -> bool:
    """ESC 키가 눌렸는지 논블로킹으로 확인"""
    if platform.system() == "Windows":
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
            return True
    else:
        if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1) == '\x1b':
            return True
    return False

def clear_key_buffer():
    """입력 버퍼에 남아있는 키를 비움"""
    if platform.system() == "Windows":
        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

def is_safe_path(path_str: str, base_dir: Path) -> bool:
    """경로가 BASE_DIR 내에 있는지 확인"""
    try:
        target_path = Path(path_str)
        resolved_target = target_path.resolve() if target_path.is_absolute() else (base_dir / target_path).resolve()
        return base_dir == resolved_target or base_dir in resolved_target.parents
    except Exception:
        return False

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_path = agent_context.CODE_DIR / "temp_logs" / "chat_log.txt"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {str(msg)}\n")
    except Exception as e:
        print(f"[로그 저장 실패: {e}]")

def update_token_usage(msg):
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        agent_context.TOTAL_TOKEN_USAGE += msg.usage_metadata.get("total_tokens", 0)
        agent_context.INPUT_TOKEN_COUNT += msg.usage_metadata.get("input_tokens", 0)
        agent_context.OUTPUT_TOKEN_COUNT += msg.usage_metadata.get("output_tokens", 0)
