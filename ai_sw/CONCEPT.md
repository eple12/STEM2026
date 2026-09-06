# FORMULA-AI 3D 레이싱 게임 — 기획서 (v0.1)

> STEM2026 동아리 프로젝트. `neat_racetrack_viz`(2D NEAT 레이싱 학습)의 3D 후속작.
> 작성일 2026-08-29.

---

## 1. 한 줄 요약

**같은 트랙, 같은 센서, 같은 신경망** — 동아리가 2D에서 학습시킨 NEAT 드라이버를
3D 세미 시뮬레이터로 옮겨와, **사람이 직접 핸들을 잡고 AI와 같은 그리드에서 경주**한다.

- 스택: **Python + Ursina** (Panda3D 기반)
- 플레이어: **사람 vs AI** (동시 주행 / 타임어택 / AI끼리 관전)
- 리얼리티: **세미 시뮬레이션** — 자전거 모델 + 간이 타이어 그립, 어시스트 on/off 슬라이더

작업 타이틀 후보: **Neuro Grand Prix**, **NEAT Racer 3D**, **동아리 GP**

---

## 2. 왜 이 방향인가 (기존 자산과의 연결)

`Neural_Network_NEAT-master/new/` 에 이미 있는 것:

| 자산 | 내용 | 3D에서의 재사용 |
|---|---|---|
| `f1tenth_racetracks-main/` | 실제 F1 서킷 23종의 중심선 CSV (`x_m, y_m, w_tr_right_m, w_tr_left_m`) + 레이스라인 + 맵 | **트랙 지오메트리 그대로** → 리본 메시로 압출 |
| `neat_racetrack_viz/models/*` | 세대별 학습된 NEAT 게놈 `.pkl` (Monza/Spa/Silverstone/Nürburgring/Zandvoort/YasMarina) | **AI 드라이버 두뇌**로 직접 로드 |
| `neat_config.ini` | 입력 9 / 출력 2 (steer, throttle) 신경망 정의 | 3D 센서도 동일 9입력으로 맞춤 |
| `main.py` 수학 | `wrap_to_pi`, `normalized_arclength`, 곡률 기반 속도 프로파일, 코너/직선 보상 | 물리·랩타임·AI 진행도 계산에 재사용 |

→ 3D 게임이 단순한 "레이싱 게임"이 아니라 **"2D에서 배운 정책이 3D 물리에서도 통하는가"**
라는 실험 플랫폼이 된다. 발표/시연에서 강력한 스토리.

---

## 3. 게임플레이

### 3.1 모드

1. **타임어택** — 혼자 주행, 베스트 랩 + 고스트 카(자기 기록 재생).
2. **레이스** — 3~5랩, 그리드 스타트, AI 4~6대와 순위 경쟁.
3. **사람 vs AI 매치** — 1:1, 같은 차량·같은 트랙. 학습된 NEAT 게놈 선택 가능.
4. **관전 (AI vs AI)** — 세대가 다른 게놈끼리 붙이기. `neat_racetrack_viz` 리플레이의 3D 버전.
5. (스트레치) **3D 네이티브 학습** — 게임 물리 안에서 NEAT/RL 재학습.

### 3.2 조작

- 키보드 우선: WASD / 방향키 + 스페이스(핸드브레이크) + R(리스폰) + C(카메라 전환).
- 게임패드: Panda3D 입력으로 추가 (트리거 = 스로틀/브레이크 아날로그).
- 카메라: 체이스 / 보닛 / 콕핏 3종 토글.

### 3.3 어시스트 슬라이더 (아케이드 ↔ 세미시뮬)

트랙션 컨트롤 · ABS · 스티어링 어시스트 · 자동 브레이크 각각 on/off.
"세미 시뮬"을 기본값으로, 초보자는 어시스트를 켜고, AI 실험 시엔 전부 끈다.

---

## 4. 차량 물리 (세미 시뮬레이션)

**커스텀 물리** (Panda Bullet 안 씀 — 결정론적이고 NEAT 사이드와 코드 공유해야 함).

- **자전거 모델(bicycle model)**: 상태 = 위치(x,z), 요각, 차체좌표 속도(vx,vy), 요레이트.
- **타이어**: 선형 코너링 강성 + 그립 한계(포화) → 언더/오버스티어, 슬립앵글.
- **종방향**: 엔진력 곡선(RPM→토크) + 브레이크력 + 공기저항 + 구름저항.
- **그립 서클(combined slip)**: 브레이크+조향 동시 = 코너링 그립 감소 (트레일 브레이킹 가능).
- **노면**: 트랙=그립1.0 / 커브=1.05 살짝 튀는 느낌 / 잔디·자갈=0.4 + 저항 증가. 벽 접촉 = 충돌(감속 + 페널티).
- **고정 타임스텝** 100Hz, 렌더와 분리(보간).
- 재사용: `main.py`의 곡률→속도상한 프로파일을 그대로 코너 속도 튜닝 기준으로.

초기엔 파라미터 몇 개(그립, 파워, 질량, 휠베이스)만 노출해서 "느낌" 잡기.

---

## 5. 트랙 시스템

### 5.1 파이프라인

```
f1tenth CSV (중심선 + 좌우폭)
  → 스플라인 리샘플 (등간격 N점, 곡률·법선 계산)   [spline.py, NEAT와 공유]
  → 스케일 업 (F1TENTH은 1:10 축소 → ×N배)
  → 리본 메시: 각 점에서 좌/우 엣지 = center ± normal·width
     · 아스팔트 쿼드 스트립, UV는 호길이 따라 타일링
  → 커브(kerb): 엣지 바깥 얇은 줄 + 살짝 융기, 빨강/흰색 줄무늬
  → 런오프: 커브 바깥 잔디/자갈 평면
  → 배리어: 아웃라인 따라 타이어월/암코 인스턴싱
  → 체크포인트 게이트: 호길이 기반 → 랩타임·섹터·AI 진행도
  → 스타트/피니시 라인, 그리드 슬롯
```

### 5.2 결정 사항

- **고도**: f1tenth 데이터는 2D(평면). v1은 평면으로. v2에서 트랙별 수동 높이 프로파일 or 노이즈.
- **스케일**: 프로젝트 초반에 고정(물리 튜닝 전체가 여기 종속). 잠정 ×10.
- **충돌**: 트랙 경계 폴리곤 vs 레이캐스트(트랙 평면 2D). 3D 메시 충돌 안 씀.

### 5.3 라이선스 주의 (§8 참고)

`f1tenth_racetracks`는 **GPLv3**. 트랙 이름은 실제 서킷 상표.
→ 배포 시 트랙 데이터 동봉하려면 게임도 GPL 호환으로 공개하거나, 데이터 비동봉(사용자가 받게).
→ 공개 배포하면 트랙명을 제네릭으로 ("Temple of Speed" = 몬자 등) 바꾸거나 "inspired by" 표기.
동아리 내부 시연/제출용이면 문제 없음.

---

## 6. AI 드라이버

### 6.1 공통 인터페이스

```python
class Driver:
    def observe(self, car, track) -> np.ndarray:   # shape (9,)
        ...
    def act(self, obs) -> tuple[float, float]:      # (steer ∈ [-1,1], throttle ∈ [-1,1])
        ...
```

### 6.2 센서 (9입력, NEAT config와 일치)

- 레이캐스트 5~7개 (전방 부채꼴) → 트랙 경계까지 거리 (정규화)
- 현재 속도 (정규화)
- 레이스라인 대비 헤딩 오차 (`wrap_to_pi`)
- (필요 시) 횡방향 위치 오차

→ **2D NEAT와 동일한 관측 벡터**를 3D 차량 상태에서 재구성. 이게 정책 전이의 핵심.

### 6.3 백엔드

| 백엔드 | 구현 | 용도 |
|---|---|---|
| `scripted` | pure-pursuit 조향 + 레이스라인 속도 프로파일 추종 | M4 기본 상대 AI, 항상 안정적 |
| `neat` | `neat_racetrack_viz/models/**.pkl` 게놈 + `neat_config.ini` 로드 | M5, "2D 두뇌 3D 주행" |
| `rl` (스트레치) | stable-baselines3 PPO or 커스텀 | 3D 네이티브 학습 |

- 난이도: `scripted` AI의 속도 프로파일 배율 + 실수 확률로 조절.
- 러버밴딩은 기본 off (실험 공정성), 캐주얼 모드에서만 on.

---

## 7. 에셋 전략 ⭐

### 원칙

1. **CC0 우선.** STEM 프로젝트 → 시연·제출·공개 가능성. CC0면 출처 표기 의무 없고 안전.
   (CC-BY도 가능하나 `CREDITS.md` 관리 필요. GPL 음악/모델은 피함.)
2. **2단계 접근**: 먼저 외부 에셋 0개로 플레이 가능하게, 그 다음 큐레이션된 팩으로 폴리시.
3. 저폴리 플랫셰이딩 아트 디렉션 (일관성 + 가벼움 + "풀 시뮬 아님"이라는 톤과 일치).

### 7.1 Phase A — 절차적/프리미티브 (다운로드 0)

| 요소 | 방법 |
|---|---|
| 트랙 | §5 절차적 리본 메시 |
| 차량 | Ursina 프리미티브 (박스 차체 + 실린더 바퀴 4개) 또는 단순 `.obj` 1개 |
| 텍스처 | numpy로 아스팔트·체커·커브 PNG 생성 (`tools/make_procedural_textures.py`) |
| 스카이 | Ursina 기본 그라디언트 스카이 |
| 사운드 | RPM 기반 사인 스윕 합성 (numpy) — 선택 |

→ 이걸로 M1~M5까지 **에셋 의존성 0**으로 완주 가능.

### 7.2 Phase B — 큐레이션 CC0 팩 (폴리시, M6+)

| 소스 | 라이선스 | 가져올 것 |
|---|---|---|
| **Kenney.nl** | CC0 | *Car Kit*(레이싱카 ~15종), *Racing Kit*(콘·배리어·타이어·텐트·깃발·관중석), *Engine Sound Pack*, *UI Pack*, *Music Jingles* |
| **Quaternius** | CC0 | Ultimate Vehicles, 자연물(나무 — 주변 환경) |
| **ambientCG** | CC0 | 타일링 PBR 텍스처: 아스팔트·콘크리트·잔디·자갈 (리얼리티 올릴 때) |
| **Poly Haven** | CC0 | HDRI 스카이(반사·조명 품질) |
| **Google Fonts** | OFL | HUD 폰트 (Orbitron / Rajdhani) — OFL은 임베드 OK |
| **Poly Pizza** | CC0 필터 | 단발성 소품 |

**피하는 것**: incompetech/Kevin MacLeod (CC-BY, 표기 필요), Sketchfab 비-CC0, F1 공식 로고·팀 리버리.

### 7.3 에셋 파이프라인

```
ai_sw/
  assets/
    models/    textures/    audio/    fonts/
    CREDITS.md          ← 모든 소스 + 라이선스 (CC0라도 기록)
  tools/
    fetch_assets.py         ← Kenney zip URL에서 받아 압축 해제, 쓰는 것만 복사
    make_procedural_textures.py
```

- Kenney 팩은 안정 URL에서 스크립트로 페치 (레포 비대화 방지). 작으면 git 벤더링.
- 필요 시 `.glb` 단일 파일로 변환.
- 원본(raw)과 가공본 분리.

---

## 8. 프로젝트 구조

```
ai_sw/
  README.md   CONCEPT.md   requirements.txt   pyproject.toml
  game/
    __main__.py            # python -m game
    app.py                 # Ursina 앱, 씬, 게임 상태 머신
    config.py
    physics/
      vehicle.py           # 자전거 모델
      tires.py
      collision.py         # 경계 레이캐스트
    track/
      loader.py            # f1tenth CSV → Track
      spline.py            # 리샘플·곡률·법선   [NEAT와 공유 후보]
      mesh.py              # 리본/커브/런오프 메시 빌더
    ai/
      base.py              # Driver 인터페이스
      sensors.py           # 9입력 관측         [NEAT와 공유 후보]
      scripted.py          # pure-pursuit
      neat_driver.py       # .pkl 게놈 로드
    entities/
      car.py   hud.py   camera_rig.py
    scenes/
      menu.py   race.py   spectator.py
  assets/   (§7.3)
  tools/    fetch_assets.py   make_procedural_textures.py   export_track_preview.py
  tests/    test_spline.py   test_vehicle.py   test_loader.py
```

**NEAT 사이드와의 코드 공유**: `spline.py` + `sensors.py`는 장기적으로 작은 `trackkit`
패키지로 추출해 `neat_racetrack_viz`와 `ai_sw` 양쪽이 import. **당장은 복사**하고 주석으로
동기화 필요 표시 → 인터페이스 안정되면 추출.

---

## 9. 로드맵 (마일스톤)

| M | 목표 | 산출물 | 대략 |
|---|---|---|---|
| M0 | 셋업 | venv + ursina 설치, "hello cube" 실행, 테스트 골격 | 0.5일 |
| M1 | 트랙 | Monza CSV 로드 → 평면 리본 메시, 절차적 아스팔트, 플라이 카메라 | 1~2일 |
| M2 | 차 + 물리 | 주행 가능 차량, 체이스 카메라, 자전거 모델, 오프트랙 그립↓, 벽 충돌, 리스폰 | 2~3일 |
| M3 | 레이스 루프 | 랩타임·체크포인트·3랩·순위·스타트 라이트·HUD, 타임어택 | 2일 |
| M4 | AI 상대 | scripted pure-pursuit AI 4~6대, 그리드 스타트, 차간 충돌(소프트) | 2~3일 |
| M5 | NEAT 연동 | `models/`에서 게놈 로드, 9센서 관측 배선, 3D에서 NEAT 주행, AI vs AI 관전 | 2~3일 |
| M6 | 폴리시 | Kenney 차·소품, 커브·배리어·관중석, 엔진 사운드, 메뉴 UI, 트랙 셀렉터, 어시스트, 고스트, 리플레이 | 지속 |
| M7 | 스트레치 | 고도, 날씨, 스플릿스크린, RL 학습 모드, 게임패드, VR | — |

**첫 시연 목표 = M5** (사람이 몬자에서 동아리가 학습시킨 AI와 경주).

---

## 10. 리스크 & 확인 필요

| 항목 | 메모 |
|---|---|
| Ursina + Python 3.12 | 설치 검증 필요. 실패 시 panda3d 직접 or 3.11 venv 폴백 |
| f1tenth 스케일 1:10 | 업스케일 배율 초반 고정 (물리 튜닝 전부 종속) |
| 평면 트랙 | 몬자/스파의 고저차 손실 — v1 감수 |
| 차간 충돌 | 데미지 없는 소프트 푸시부터 |
| 2D 정책 전이 실패 가능성 | **버그 아니라 실험 결과** — 시연 포인트. 3D 네이티브 학습 모드로 대응 |
| 성능 | 배리어 인스턴싱, 메시 병합, 관중석 LOD |
| GPLv3 트랙 데이터 | 공개 배포 시 §5.3 |

### 지금 답이 필요한 것

1. 아트 디렉션: **저폴리 플랫셰이딩(Kenney)** vs 텍스처 리얼리즘 → 기획서는 저폴리 가정.
2. 그리드 크기(AI 대수): 4~6 권장(성능).
3. 타이틀 확정.
4. 공개 범위: 동아리 제출용인지 GitHub 공개인지 (→ 라이선스·트랙명 결정).

---

## 11. 다음 단계

1. 이 기획서 리뷰 → 위 4개 결정.
2. M0: `ai_sw/` 스캐폴딩 (`requirements.txt`, `game/` 골격, ursina 설치 검증).
3. M1: 트랙 로더 + 메시 빌더 (가장 재미있고 위험도 낮은 첫 조각).
