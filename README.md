# Prompt Reader

PNG 이미지에 포함된 생성 프롬프트 메타데이터(ComfyUI, Stable Diffusion WebUI 계열 등)를 추출해서
보여주고, 버튼 클릭 한 번으로 클립보드에 복사해 주는 로컬 웹 앱입니다.

- Python 3.13.x 고정 (`.python-version`, `requires-python = ">=3.13,<3.14"`)
- 패키지 관리: [uv](https://docs.astral.sh/uv/)
- 서버: FastAPI + uvicorn, `http://127.0.0.1:22000`

## 실행 방법

```bash
uv sync                # 가상환경 생성 + 의존성 설치 (.venv)
uv run python main.py  # 서버 실행 (uv run prompt-reader 도 동일)
```

서버가 뜨면 **기본 브라우저가 자동으로 열립니다**. 자동 실행이 안 되는 환경에서는 직접
**http://127.0.0.1:22000** 에 접속하세요.

## 전역 명령으로 설치 (어디서든 `prompt-reader` 실행)

프로젝트 디렉터리에서 한 번만 설치하면 어떤 경로에서든 `prompt-reader` 명령으로 실행됩니다:

```bash
uv tool install .        # 프로젝트 디렉터리(G:\project\python\prompt_reader)에서 실행
```

이후 터미널 어디서든:

```bash
prompt-reader                # 기본 22000 포트로 서버 실행 + 브라우저 자동 오픈
prompt-reader --port 23000   # 포트 지정 실행
prompt-reader --host 0.0.0.0 # LAN의 다른 기기에서 접속 허용
```

`uv run python main.py --port 23000` 처럼 개발 실행에서도 같은 옵션을 씁니다.

- 코드 수정 후 반영(재설치): `uv tool install --force .`
- 제거: `uv tool uninstall prompt-reader`
- 설치된 도구 목록: `uv tool list`
- `prompt-reader`를 찾지 못한다면 uv의 실행 파일 경로가 PATH에 없는 것이므로
  `uv tool update-shell` 실행 후 터미널을 다시 여세요.

프로젝트 디렉터리에서 개발할 때는 기존대로 `uv run python main.py` 또는 `uv run prompt-reader`도 사용할 수 있습니다.

## 기능

- **파일 열기** 버튼 또는 **드래그 앤 드롭**으로 PNG 로드
- 읽어온 이미지를 원본 비율 유지한 채 적절한 크기로 미리보기 + 파일명/해상도/용량/출처 표시
- 긍정 프롬프트 / 부정 프롬프트 / 생성 설정(Steps, Sampler, Seed, Model 등) 추출·표시
- 각 항목의 **복사** 버튼으로 클립보드에 즉시 복사 (localhost는 브라우저에서 보안 컨텍스트로 취급되어
  Clipboard API 동작, 미지원 환경은 폴백 처리)
- 원본 텍스트 청크 전체 확인 (접기/펼치기) 및 전체 복사 — JSON 청크(`prompt`, `workflow` 등)는
  기본적으로 **줄바꿈 정렬 + 유니코드 이스케이프 해석**(`\uc138...` → 한국어)해서 보여주고,
  버튼으로 원문(한 줄) 보기 전환 가능. ComfyUI가 기록하는 Python JSON 확장 표기(`NaN` 등)도
  서버 쪽에서 처리해 정렬 보기가 실패하지 않음
- PNG가 아니거나 메타데이터가 없는 이미지는 안내 메시지 표시

## 지원 포맷

| 생성기 | 사용 청크 | 비고 |
|---|---|---|
| Automatic1111 / SD WebUI / 포크(Forge 등) | `parameters` | 멀티라인 프롬프트·네거티브 지원 |
| ComfyUI | `prompt` (API 그래프) | 샘플러 → CLIPTextEncode 링크 추적으로 positive/negative 구분, ConditioningCombine/ControlNet 등 중간 노드도 통과. `text` 입력이 링크(LLM 프롬프트 생성기 출력 등)면 ShowText 계열 노드에 남은 결과값(`text_0`)으로 최종 프롬프트 복원 |
| ComfyUI (메타데이터 축약본) | `workflow` | workflow 청크만 있는 경우 링크 테이블로 복원 |
| NovelAI | `Title` / `Description` / `Comment` | |
| InvokeAI 구버전 | `sd-metadata` | |

tEXt / zTXt / iTXt 청크와 UTF-8(한국어 프롬프트 포함)를 모두 지원합니다.

## 프로젝트 구조

```
prompt_reader/
├── main.py               # FastAPI 앱 + uvicorn 실행 (포트 22000)
├── pngmeta.py            # PNG 청크 파서 + 프롬프트 추출기 (의존성 없음)
├── static/index.html     # 프론트엔드 (단일 파일)
├── samples/
│   ├── make_samples.py   # 테스트용 샘플 PNG 생성기
│   └── sample_*.png      # 생성된 샘플 (A1111/ComfyUI/NovelAI/메타데이터 없음)
├── test_pngmeta.py       # 단위 테스트
├── pyproject.toml
└── .python-version       # 3.13 고정
```

## 테스트

```bash
uv run python test_pngmeta.py          # 단위 테스트 (14개)
uv run python samples/make_samples.py  # samples/ 아래 샘플 PNG 재생성
```

`samples/`의 샘플 PNG를 페이지에 드롭하면 각 생성기 포맷의 추출 결과를 직접 확인할 수 있습니다.

## 참고

- 포트 기본값은 22000, 호스트 기본값은 127.0.0.1 — `--port` / `--host` 옵션으로 변경
- `--host 0.0.0.0`으로 LAN 접속을 허용해도 자동으로 열리는 브라우저는 루프백(127.0.0.1) 주소로 연다
- 다른 기기에서 http + localhost 가 아닌 주소로 접속하면 브라우저 Clipboard API가 차단될 수 있음
  (폴백 복사는 동작)
