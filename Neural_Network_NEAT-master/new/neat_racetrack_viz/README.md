# NEAT F1TENTH Racetrack Visualizer

`f1tenth_racetracks-main`의 데이터 형식을 기준으로, NEAT로 간단한 주행 정책을 학습하고 학습 진행을 실시간으로 시각화하는 프로젝트입니다.

## README 기반 트랙 처리 전략

원본 README의 핵심 포인트를 코드에 반영했습니다.

- `centerline`: `[x_m, y_m, w_tr_right_m, w_tr_left_m]` 포맷(쉼표 구분)
- `raceline`: `[s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2]` 포맷(세미콜론 구분)
- `map.yaml`: 해상도(`resolution`)와 원점(`origin`) 포함
- 일부 트랙은 폴더명과 파일 접두어가 다를 수 있어(`Mexico City` vs `MexicoCity`) 자동 보정 로직 사용

따라서 로더는 아래를 수행합니다.

1. 주석(`#`)과 빈 줄 제거
2. 줄 단위로 구분자(`,` 또는 `;`) 자동 감지
3. 필요한 열만 안정적으로 파싱
4. 트랙 폭 정보(`width_left`, `width_right`)를 보상 및 이탈 판정에 반영

## 모델 입력 및 출력 (Inputs & Outputs)

NEAT 알고리즘이 학습하는 신경망의 구조는 다음과 같습니다.

### 입력값 (9개 관측치 - Observations)
1. **정규화된 횡방향 오차 (Normalized Lateral Error)**: 차량이 기준선(`--follow-line`)에서 얼마나 벗어났는지를 트랙 반폭(half-width)으로 나눈 값입니다.
2. **정규화된 헤딩 오차 (Normalized Heading Error)**: 차량의 현재 진행 방향과 기준선의 목표 방향 사이의 각도 차이를 $\pi$로 나눈 값입니다.
3. **정규화된 속도 (Normalized Speed)**: 현재 속도를 해당 트랙의 최대 허용 속도로 나눈 값입니다.
4. **트랙 진행도 (Track Progress)**: 현재 위치가 전체 트랙의 어느 지점인지를 0~1 사이로 나타낸 값입니다.
5. **현재 속도 편차 (Speed Delta)**: (현재 지점 권장 속도 - 현재 속도) / 최대 속도. 양수면 가속, 음수면 감속이 필요함을 의미합니다.
6. **전방 최소 권장 속도 (Future Min Speed)**: 전방 구간의 가장 낮은 권장 속도입니다. 코너가 가까울수록 낮아집니다.
7. **전방 지점 헤딩 오차 (Lookahead Heading Error)**: 약 15m 전방 지점의 곡률 정보를 미리 파악하여 조향을 준비할 수 있게 합니다.
8. **브레이킹 긴급도 (Brake Urgency)**: 현재 속도가 전방 최소 권장 속도보다 얼마나 높은지 나타냅니다.
9. **전방 곡률 심각도 (Curvature Ahead)**: 앞쪽 코너가 얼마나 강한지 0~1 범위로 나타냅니다.

### 출력값 (2개 행동 - Actions)
1. **조향 (Steering)**: -1.0(왼쪽 최대)에서 1.0(오른쪽 최대) 사이의 값입니다. 실제 적용 가능한 조향률은 속도에 따라 달라지며, 저속에서는 크게 꺾을 수 있고 고속에서는 작게 제한됩니다.
2. **가감속 (Throttle/Brake)**: -1.0(최대 제동)에서 1.0(최대 가속) 사이의 값입니다.

---

## CLI 명령어 전체 정리

아래 명령어는 `new/neat_racetrack_viz` 폴더에서 실행하는 기준입니다.

```bash
cd new/neat_racetrack_viz
python -m pip install -r requirements.txt
```

기본 학습:

```bash
python main.py
```

Austin 트랙에서 40세대 학습하고 시각화까지 켜는 기본 추천 명령:

```bash
python main.py --track-dir ../f1tenth_racetracks-main/Austin --generations 40 --difficulty easy --animate-best --show-net
```

raceline을 기준으로 따라가게 학습하는 명령:

```bash
python main.py --track-dir ../f1tenth_racetracks-main/Austin --follow-line raceline --generations 40 --difficulty easy --animate-best --show-net
```

빠른 확인용 1세대 테스트:

```bash
python main.py --track-dir ../f1tenth_racetracks-main/Austin --generations 1 --difficulty overfit --no-gui
```

다른 트랙 사용:

```bash
python main.py --track-dir "../f1tenth_racetracks-main/Mexico City" --follow-line raceline --generations 50
```

### 전체 옵션

- `--track-dir PATH`: 사용할 트랙 폴더입니다. 폴더 안에 `*_centerline.csv`, `*_raceline.csv`, `*_map.yaml` 파일이 있어야 합니다. 기본값은 `f1tenth_racetracks-main/Austin`입니다.
- `--config PATH`: `neat-python` 설정 파일 경로입니다. 기본값은 `neat_config.ini`입니다.
- `--generations N`: 학습할 세대 수입니다. 기본값은 `40`입니다. 빠른 테스트는 `1`, 긴 학습은 `100` 이상처럼 늘릴 수 있습니다.
- `--difficulty {normal,easy,very-easy,overfit,ideal}`: 학습 난이도 프리셋입니다. 기본값은 `easy`입니다.
- `--follow-line {centerline,raceline}`: 횡방향 오차, 헤딩 오차, 진행도 보상의 기준선을 고릅니다. 기본값은 `centerline`입니다. 레이싱 라인을 따라가게 하고 싶으면 `--follow-line raceline`을 사용하세요.
- `--no-gui`: matplotlib 창 없이 콘솔에서만 학습합니다. 원격 환경, 빠른 테스트, 긴 학습에서는 이 옵션이 안정적입니다.
- `--show-net`: 시각화 창 오른쪽에 신경망 입력/출력 활성도 막대를 표시합니다.
- `--animate-best`: 각 세대의 best genome 궤적을 시작 지점부터 애니메이션으로 보여줍니다.
- `--animate-step-pause SEC`: best 궤적 애니메이션의 프레임 간 지연 시간입니다. 기본값은 `0.01`입니다. 작을수록 빠르게 재생됩니다.
- `--animate-window-m M`: 애니메이션 확대 창의 반경입니다. 기본값은 `18.0`m입니다. 차량 주변을 더 좁게 보고 싶으면 `14`처럼 줄이면 됩니다.
- `--checkpoint-every N`: 센터라인 원본 샘플 기준으로 체크포인트를 배치하는 간격입니다. 기본값은 `45`입니다.
- `--checkpoint-pass-reward VALUE`: 다음 체크포인트를 순서대로 통과했을 때 주는 보상입니다. 기본값은 `72.0`입니다.
- `--checkpoint-miss-penalty VALUE`: 현재 목표가 아닌 체크포인트를 먼저 지나갔을 때 주는 패널티입니다. 기본값은 `48.0`입니다.
- `--max-steps N`: 한 genome의 평가 step 제한입니다. `0`이면 트랙 길이와 난이도에 따라 자동 계산합니다. `dt=0.1`초이므로 `300` step은 시뮬레이션 시간 약 30초입니다.
- `--models-dir PATH`: 세대별 best 모델을 저장할 루트 폴더입니다. 기본값은 `neat_racetrack_viz/models`입니다. 실행할 때마다 현재시각 폴더가 생기고 그 안에 `generation_0000_best.pkl` 형식으로 저장됩니다.

### 속도별 조향 제한

차량은 고정된 최대 조향률을 그대로 쓰지 않고, 현재 속도가 빠를수록 적용 가능한 조향률이 줄어듭니다. 덕분에 저속 코너에서는 크게 꺾을 수 있고, 고속 직선에서는 작은 조향만 허용되어 갑작스러운 회전과 이탈을 줄일 수 있습니다.

관련 상수는 `main.py` 상단에 있습니다.

- `MAX_STEER_RATE`: 저속에서 사용할 최대 조향률입니다.
- `MIN_STEER_RATE_FACTOR`: 최고속 근처에서 남겨둘 최소 조향률 비율입니다. 기본값 `0.35`는 최고속에서 저속 최대 조향률의 35%까지만 허용한다는 뜻입니다.
- `STEER_SPEED_SENSITIVITY`: 속도 증가에 따라 조향 제한이 얼마나 빨리 강해지는지 정합니다. 값이 클수록 저속에서는 조향을 더 많이 허용하고, 고속으로 갈수록 급하게 제한됩니다.

### 자주 쓰는 조합

raceline 기준으로 창 없이 길게 학습:

```bash
python main.py --track-dir ../f1tenth_racetracks-main/Austin --follow-line raceline --generations 100 --difficulty easy --no-gui
```

학습이 너무 오래 걸릴 때 빠른 수렴용:

```bash
python main.py --follow-line raceline --difficulty overfit --generations 20 --no-gui
```

에피소드 시간제한을 늘려서 느린 초반 개체도 더 오래 보게 하기:

```bash
python main.py --follow-line raceline --max-steps 600 --generations 40
```

체크포인트 보상을 더 강하게 주기:

```bash
python main.py --follow-line raceline --checkpoint-every 50 --checkpoint-pass-reward 90 --checkpoint-miss-penalty 60
```

모델 저장 위치를 바꾸기:

```bash
python main.py --follow-line raceline --models-dir ./models_raceline --generations 40
```

저장된 모델을 트랙 경로와 신경망 활성도 애니메이션으로 재생하기:

```bash
python replay_model.py
```

인자 없이 실행하면 `models/` 폴더 아래에서 가장 최근에 저장된 `generation_XXXX_best.pkl`을 자동으로 불러옵니다. 특정 모델을 직접 지정하려면:

```bash
python replay_model.py models/20260505_133815/generation_0000_best.pkl
```

전체 트랙을 고정 화면으로 보고 싶으면:

```bash
python replay_model.py models/20260505_133815/generation_0000_best.pkl --no-zoom
```

재생 속도를 늦추거나 빠르게 조정하려면:

```bash
python replay_model.py models/20260505_133815/generation_0000_best.pkl --pause 0.02 --window-m 14
```

창을 열지 않고 저장 파일이 잘 읽히는지만 확인하려면:

```bash
python replay_model.py --summary-only
```

현재 보이는 replay 화면을 영상 파일로 저장하려면:

```bash
python replay_model.py --save-video replay.gif --no-show
```

기본 환경에서는 matplotlib의 `pillow` writer로 `.gif` 저장이 바로 됩니다. `.mp4`, `.mov`, `.m4v`로 저장하려면 시스템에 `ffmpeg`가 설치되어 있어야 합니다.

```bash
python replay_model.py --save-video replay.mp4 --video-fps 30 --no-show
```

영상 파일이 너무 크거나 저장이 느리면 프레임을 건너뛰어 저장할 수 있습니다. 예를 들어 `--video-stride 2`는 두 step마다 한 프레임만 저장합니다.

```bash
python replay_model.py --save-video replay.gif --video-stride 2 --video-fps 20 --no-show
```

### 난이도 프리셋

- `normal`: 가장 보수적인 기본 난이도입니다.
- `easy`: 기본 추천입니다. 학습이 비교적 빨리 붙도록 완화되어 있습니다.
- `very-easy`: 더 빠른 수렴을 위한 설정입니다.
- `overfit`: 일반화보다 빠른 수렴을 우선하는 설정입니다.
- `ideal`: 속도 보상을 강하게 둔 실험용 설정입니다.

쉬운 모드일수록 이탈 허용폭, 에피소드 길이, 다운샘플링, 체크포인트 보상 계수가 함께 조정됩니다.

### 저장되는 모델 파일

학습을 한 번 실행하면 `models/YYYYMMDD_HHMMSS/` 폴더가 만들어집니다. 각 세대의 best 모델은 아래처럼 저장됩니다.

```text
models/
  20260505_133815/
    generation_0000_best.pkl
    generation_0001_best.pkl
    ...
```

각 `.pkl` 파일에는 `genome`, `config`, `generation`, `fitness`, `track_dir`, `config_path`, `difficulty`, `follow_line`이 들어 있습니다.
새로 저장되는 파일에는 학습 당시의 실제 에피소드 제한인 `max_steps`도 함께 들어가며, `replay_model.py`는 이 값을 사용해서 학습 때와 같은 길이로 재생합니다. 예전 pkl처럼 `max_steps`가 없는 파일은 기본값으로 재계산되므로, 길게 학습한 예전 모델을 끝까지 보고 싶으면 `--max-steps`로 직접 지정하세요.

## 파일 설명

- `main.py`: 트랙 로딩, 환경 시뮬레이션, NEAT 학습, 실시간 플롯
- `replay_model.py`: 저장된 `.pkl` 모델을 불러와 트랙 주행 경로와 신경망 활성도를 애니메이션으로 재생
- `neat_config.ini`: `neat-python` 설정값
- `requirements.txt`: 의존성 목록
