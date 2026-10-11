#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# prompt_reader 서비스 운영 스크립트 (start/stop/restart/status/logs)
# 이 저장소 안에서 독립적으로 동작하며 다른 프로젝트와 공유 코드가 없음.
# 다른 프로젝트에 적용할 때는 아래 "설정" 구간만 바꿔서 그대로 복사하면 됨.
# ---------------------------------------------------------------------------

# ── 설정 ────────────────────────────────────────────────────────────────
APP_NAME="prompt_reader"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOST="127.0.0.1"
APP_PORT=22000
APP_CMD=(/home/drasys/.local/bin/uv run "${APP_DIR}/main.py" --host "${APP_HOST}" --port "${APP_PORT}")
# 앱에 별도 헬스 엔드포인트가 없어 index 페이지로 확인
APP_HEALTH_PATH="/"
# 프로세스 식별 문자열 (cmdline에 포함 여부로 판별)
APP_PROC_PATTERN="${APP_DIR}/main.py"
APP_PID_FILE="${APP_DIR}/${APP_NAME}.pid"
APP_LOG_FILE="${APP_DIR}/${APP_NAME}.log"
# ────────────────────────────────────────────────────────────────────────

usage() {
  cat <<EOF
Usage: ./control.sh <command>

Commands:
  start                       ${APP_NAME} 백그라운드로 시작
  stop                        ${APP_NAME} 정지
  restart                     정지 후 재시작
  status                      실행 상태 / 포트 / 헬스체크 표시
  logs [--lines N] [--follow|--no-follow]
                              로그 확인 (기본: tail -n 100 -f)
EOF
}

pid_from_file() {
  [[ -f "${APP_PID_FILE}" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "${APP_PID_FILE}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

is_alive() {
  kill -0 "$1" >/dev/null 2>&1
}

# pid 파일의 PID가 실제로 이 서비스인지 cmdline에 APP_PROC_PATTERN 포함 여부로 확인
# (재사용된 PID에 잘못 신호를 보내는 것을 방지)
is_owned_pid() {
  local pid="$1" args
  args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ -n "${args}" ]] || return 1
  [[ "${args}" == *"${APP_PROC_PATTERN}"* ]]
}

current_pid() {
  local pid
  pid="$(pid_from_file)" || return 1
  if is_alive "${pid}" && is_owned_pid "${pid}"; then
    printf '%s\n' "${pid}"
    return 0
  fi
  return 1
}

clear_stale_pid() {
  if [[ -f "${APP_PID_FILE}" ]]; then
    rm -f "${APP_PID_FILE}"
    echo "[${APP_NAME}] 오래된 PID 파일을 정리했습니다: ${APP_PID_FILE}"
  fi
}

# pid 파일과 무관하게 동일 프로세스가 이미 떠 있는지 확인
# (수동 실행 등 control.sh 밖에서 기동된 프로세스와의 충돌 방지)
find_pid_by_grep() {
  ps -eo pid=,args= | grep -F "${APP_PROC_PATTERN}" | grep -v grep | grep -v "control.sh" | awk '{print $1}' | head -n1
}

start_cmd() {

  local existing_pid
  if existing_pid="$(current_pid 2>/dev/null)"; then
    echo "[${APP_NAME}] 이미 실행 중입니다 (PID ${existing_pid})"
    return 0
  fi

  local grep_pid
  grep_pid="$(find_pid_by_grep || true)"
  if [[ -n "${grep_pid}" ]]; then
    echo "[${APP_NAME}] control.sh 로 관리되지 않는 동일 프로세스가 이미 실행 중입니다 (PID ${grep_pid})"
    echo "[${APP_NAME}] 필요하면 먼저 해당 프로세스를 정리한 뒤 다시 시도하세요."
    return 2
  fi

  clear_stale_pid >/dev/null 2>&1 || true

  : >> "${APP_LOG_FILE}"
  (
    cd "${APP_DIR}"
    exec nohup "${APP_CMD[@]}" < /dev/null
  ) >> "${APP_LOG_FILE}" 2>&1 &
  local pid=$!

  printf '%s\n' "${pid}" > "${APP_PID_FILE}"
  sleep 0.3
  if ! is_alive "${pid}"; then
    echo "[${APP_NAME}] 기동에 실패했습니다. 로그를 확인하세요: ${APP_LOG_FILE}" >&2
    rm -f "${APP_PID_FILE}"
    return 1
  fi
  echo "[${APP_NAME}] 시작됨 (PID ${pid})"
  echo "[${APP_NAME}] 포트: ${APP_PORT}"
  echo "[${APP_NAME}] 로그: ${APP_LOG_FILE}"
}

stop_cmd() {
  local pid
  if ! pid="$(pid_from_file 2>/dev/null)"; then
    echo "[${APP_NAME}] 이미 정지 상태입니다"
    rm -f "${APP_PID_FILE}"
    return 0
  fi

  if ! is_alive "${pid}" || ! is_owned_pid "${pid}"; then
    clear_stale_pid
    return 0
  fi

  echo "[${APP_NAME}] 정지 중 (PID ${pid})"
  kill "${pid}" >/dev/null 2>&1 || true
  local i
  for i in {1..50}; do
    if ! is_alive "${pid}"; then
      rm -f "${APP_PID_FILE}"
      echo "[${APP_NAME}] 정지됨"
      return 0
    fi
    sleep 0.1
  done

  echo "[${APP_NAME}] SIGTERM 후에도 종료되지 않아 SIGKILL을 전송합니다" >&2
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  rm -f "${APP_PID_FILE}"
}

health_line() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "확인 불가 (curl 없음)"
    return 0
  fi
  local url="http://${APP_HOST}:${APP_PORT}${APP_HEALTH_PATH}"
  if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
    echo "ok (${url})"
  else
    echo "응답 없음 (${url})"
  fi
}

status_cmd() {
  local pid uptime health managed_by="control.sh"
  if ! pid="$(current_pid 2>/dev/null)"; then
    [[ -f "${APP_PID_FILE}" ]] && clear_stale_pid >/dev/null 2>&1 || true
    # control.sh 가 기동한 것은 아니지만 외부에서 이미 떠 있을 수 있어 함께 확인
    pid="$(find_pid_by_grep || true)"
    managed_by="외부 프로세스"
  fi

  if [[ -n "${pid}" ]]; then
    uptime="$(ps -p "${pid}" -o etime= 2>/dev/null | sed 's/^ *//' || true)"
    health="$(health_line)"
    echo "● ${APP_NAME} — 실행 중 (${managed_by})"
    echo "  PID:     ${pid}"
    echo "  Uptime:  ${uptime:-알 수 없음}"
    echo "  Port:    ${APP_PORT}"
    echo "  Log:     ${APP_LOG_FILE}"
    echo "  Health:  ${health}"
  else
    echo "● ${APP_NAME} — 정지됨"
    echo "  PID:     -"
    echo "  Port:    ${APP_PORT}"
    echo "  Log:     ${APP_LOG_FILE}"
    echo "  Health:  확인 안 함"
  fi
}

logs_cmd() {
  local lines=100 follow=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --lines)
        shift
        lines="${1:-}"
        [[ "${lines}" =~ ^[0-9]+$ ]] || { echo "[${APP_NAME}] --lines 뒤에는 숫자가 필요합니다" >&2; return 2; }
        ;;
      --lines=*)
        lines="${1#--lines=}"
        [[ "${lines}" =~ ^[0-9]+$ ]] || { echo "[${APP_NAME}] --lines 뒤에는 숫자가 필요합니다" >&2; return 2; }
        ;;
      --follow|-f)
        follow=1
        ;;
      --no-follow)
        follow=0
        ;;
      *)
        echo "[${APP_NAME}] 알 수 없는 logs 옵션: $1" >&2
        return 2
        ;;
    esac
    shift
  done
  touch "${APP_LOG_FILE}"
  if (( follow )); then
    tail -n "${lines}" -f "${APP_LOG_FILE}"
  else
    tail -n "${lines}" "${APP_LOG_FILE}"
  fi
}

cmd="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${cmd}" in
  start) start_cmd ;;
  stop) stop_cmd ;;
  restart) stop_cmd; start_cmd ;;
  status) status_cmd ;;
  logs) logs_cmd "$@" ;;
  -h|--help|help|"") usage ;;
  *) echo "[${APP_NAME}] 알 수 없는 명령: ${cmd}" >&2; usage >&2; exit 2 ;;
esac
