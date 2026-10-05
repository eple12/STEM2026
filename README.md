# STEM2026

STEM 2026 프로젝트 모음. 그중 **FORMULA-AI**(`ai_sw/`)는 직접 운전할 수 있는 3D 레이싱 게임이고,
20대의 AI가 같이 달립니다 (퀄리파잉 / 그랑프리, 서킷 23개).

## FORMULA-AI 실행 (Windows)

**`FORMULA-AI.bat`을 더블클릭하면 끝입니다.**

- 처음 한 번: Python(3.10–3.13)을 찾아 `ai_sw\.venv`를 만들고 필요한 패키지를 설치합니다
  (인터넷 필요, 수 분). Python이 없으면 winget으로 3.12를 설치할지 물어봅니다.
- 그 다음부터: 바로 게임이 열립니다.
- 메뉴를 건너뛰려면 `FORMULA-AI.bat --track Spa --laps 5`, 전체화면은 `--fullscreen`.

그래픽카드 드라이버가 최신이어야 하고, OpenGL 3.2 이상이 필요합니다 (Intel 내장 그래픽으로도 1080p에서
60 fps 안팎).

### 조작

| 키 | 동작 |
|---|---|
| W / S | 가속 / 브레이크 |
| A / D | 조향 |
| SPACE | 핸드브레이크 |
| ESC | 일시정지 |
| H | HUD 숨기기 |
| F11 | 전체화면 |
| **F9** | **화면 녹화 시작 / 종료** |

전체 키 목록과 게임 설명은 [`ai_sw/README.md`](ai_sw/README.md).

### 화면 녹화 (F9)

게임이 직접 인코딩하지 않고 `ffmpeg`로 창을 캡처해 GPU 인코더(Intel Quick Sync, NVENC, AMF)에 넘기므로
게임이 느려지지 않습니다. [ffmpeg](https://ffmpeg.org/download.html)가 PATH에 있거나
`C:\Program Files\ffmpeg\bin`에 있어야 하며(`winget install Gyan.FFmpeg`), 파일은
`내 동영상\FORMULA-AI`에 저장됩니다. 소리는 녹음되지 않습니다.

## 라이선스와 출처

- 서킷 데이터: [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) (MIT).
- 도로변 일부 모델: [Kenney](https://kenney.nl) Racing Kit (CC0) — `ai_sw/assets/models/kenney/LICENSE.txt`.
- 나무 이미지: `ai_sw/assets/foliage/LICENSE.txt`.
- 이 저장소의 나머지 코드와 자산에는 아직 라이선스 파일이 없습니다.
