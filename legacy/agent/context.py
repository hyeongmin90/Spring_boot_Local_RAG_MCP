import threading
from pathlib import Path

# 애플리케이션의 여러 모듈에서 공유되는 전역 컨텍스트입니다.
# AgentApp 인스턴스가 생성될 때 app_instance에 할당됩니다.
app_instance = None

# 도구들 간의 동시 접근을 막기 위한 락
approval_lock = threading.Lock()

# 프로젝트의 기본 작업 디렉토리 (명령어 실행 위치)
BASE_DIR = Path.cwd()

# 코드가 위치한 디렉토리 (로그 파일 저장용) - agent 폴더의 부모
CODE_DIR = Path(__file__).resolve().parent.parent

TOTAL_TOKEN_USAGE = 0
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

# 서브 에이전트 실행 상태 추적
sub_agent_running = False