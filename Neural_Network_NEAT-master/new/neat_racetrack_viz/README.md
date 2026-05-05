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

### 입력값 (7개 관측치 - Observations)
1. **정규화된 횡방향 오차 (Normalized Lateral Error)**: 차량이 센터라인에서 얼마나 벗어났는지를 트랙 반폭(half-width)으로 나눈 값입니다.
2. **정규화된 헤딩 오차 (Normalized Heading Error)**: 차량의 현재 진행 방향과 레이싱 라인의 목표 방향 사이의 각도 차이를 $\pi$로 나눈 값입니다.
3. **정규화된 속도 (Normalized Speed)**: 현재 속도를 해당 트랙의 최대 허용 속도로 나눈 값입니다.
4. **트랙 진행도 (Track Progress)**: 현재 위치가 전체 트랙의 어느 지점인지를 0~1 사이로 나타낸 값입니다.
5. **현재 속도 편차 (Speed Delta)**: (현재 지점 권장 속도 - 현재 속도) / 최대 속도. 양수면 가속, 음수면 감속이 필요함을 의미합니다.
6. **전방 속도 편차 (Lookahead Delta)**: (15m 전방 권장 속도 - 현재 속도) / 최대 속도. 코너 진입 전 **미리 감속**해야 할 정도를 나타냅니다.
7. **전방 지점 헤딩 오차 (Lookahead Heading Error)**: 약 15m 전방 지점의 곡률 정보를 미리 파악하여 조향을 준비할 수 있게 합니다.

### 출력값 (2개 행동 - Actions)
1. **조향 (Steering)**: -1.0(왼쪽 최대)에서 1.0(오른쪽 최대) 사이의 값입니다.
2. **가감속 (Throttle/Brake)**: -1.0(최대 제동)에서 1.0(최대 가속) 사이의 값입니다.

---

## 사용자 정의 가능한 상수 (Customizable Constants)

학습 환경과 난이도를 조정하기 위해 다음과 같은 변수들을 커스터마이징 할 수 있습니다.

### 1. 실행 인자 (Command Line Arguments)
`main.py` 실행 시 인자로 전달하여 즉시 변경 가능합니다.
- `--generations`: 진화할 최대 세대 수 (기본값: 40)
- `--difficulty`: 난이도 프리셋 (`normal`, `easy`, `very-easy`, `overfit` / 기본값: `easy`)
- `--show-net`: 애니메이션 모드에서 실시간 신경망 활성도 그래프 표시 (기본값: False)
- `--checkpoint-every`: 체크포인트 설치 간격 (기본값: 45)
- `--checkpoint-pass-reward`: 체크포인트 통과 시 부여되는 보상 (기본값: 72.0)
- `--checkpoint-miss-penalty`: 체크포인트 순서 위반 시 부여되는 패널티 (기본값: 48.0)

### 2. 난이도 프리셋 상세 설정 (`DifficultySettings`)
`main.py` 내부의 `DifficultySettings` 클래스에서 각 프리셋별 세부 수치를 조정할 수 있습니다.
- `width_scale`: 트랙 폭 허용 범위 계수
- `offtrack_scale`: 이탈 판정 기준 계수
- `lane_penalty_weight`: 차선 이탈 패널티 가중치
- `heading_penalty_weight`: 방향 오차 패널티 가중치
- `speed_scale`: 속도 보상 가중치
- `steer_penalty_weight`: 급격한 조향 패널티 가중치
- `start_speed`: 차량의 초기 속도

### 3. NEAT 알고리즘 설정 (`neat_config.ini`)
신경망 구조와 진화 알고리즘의 핵심 파라미터입니다.
- `pop_size`: 한 세대당 개체 수 (기본값: 150)
- `activation_mutate_rate`: 활성화 함수 변이 확률
- `weight_mutate_rate`: 가중치 변이 확률
- `conn_add_prob` / `node_add_prob`: 연결/노드 추가 확률

---

## 학습/시각화 구성

- 보상 구조:
  - **순차 체크포인트**(centerline 기준 간격 게이트) 통과 보상과, 순서 어김 교차 패널티
  - **속도 매칭 보상**: 현재 속도가 해당 지점의 권장 속도(`target_speed_now`)에 가까울수록 보상 부여
  - **전방 예측 패널티**:
    - 현재 속도가 전방의 권장 속도(`target_speed_lookahead`)보다 너무 높으면 패널티 부여 (미리 브레이크 유도)
    - 현재 방향이 전방의 목표 방향과 크게 다르면 패널티 부여 (미리 조향 유도)
  - 차선 이탈, 헤딩 오차, 과도한 조향에 대한 패널티 및 트랙 이탈 시 종료
- 시각화:
  - 좌측: centerline/raceline + 현재까지 최고 개체의 주행 궤적
  - 중앙: 세대별 최고/평균 fitness
  - 우측: (`--show-net` 시) 실시간 신경망 활성도 (7개 입력, 2개 출력)

## 실행 방법

```bash
cd new/neat_racetrack_viz
python -m pip install -r requirements.txt
python main.py --track-dir ../f1tenth_racetracks-main/Austin --generations 40 --difficulty easy --animate-best --show-net
```

빠른 확인(창 없이 1세대 테스트):

```bash
python main.py --track-dir ../f1tenth_racetracks-main/Austin --generations 1 --difficulty overfit --no-gui
```

다른 트랙 예시:

```bash
python main.py --track-dir "../f1tenth_racetracks-main/Mexico City" --generations 50
```

## 난이도(학습 속도) 프리셋

- `normal`: 가장 보수적인 기본 난이도
- `easy`: 기본 추천. 학습이 더 빨리 붙도록 완화
- `very-easy`: 더 빠른 수렴용
- `overfit`: 매우 빠른 학습(일반화보다 수렴 우선)

쉬운 모드일수록 아래가 자동 적용됩니다.

- 이탈 허용폭 확대
- 에피소드 길이 축소(세대당 계산량 감소)
- 트랙 샘플 다운샘플링(평가 속도 향상)
- 체크포인트 통과 보상 계수 증가(난이도 프리셋의 `checkpoint_pass_reward_scale`)

예시:

```bash
python main.py --difficulty overfit --generations 20
```

## 체크포인트 보상 설정

체크포인트는 원본 센터라인을 간격 `--checkpoint-every`로 샘플한 뒤, **센터라인 인덱스가 증가하는 순서**(폐합 polyline 순방향)대로 게이트를 둡니다.  
목표 gate만 지날 때 큰 보상이 나가고, **다른 gate를 먼저 지나면 패널티**입니다. 필요한 중간 gate를 모두 통과한 뒤 다시 시작 gate를 통과하면 라운드가 초기화됩니다.

```bash
python main.py --checkpoint-every 50 --checkpoint-pass-reward 80 --checkpoint-miss-penalty 55
```

## Plot 창 응답 없음 완화

평가 루프 중에도 주기적으로 GUI 이벤트를 처리하도록 수정했습니다.  
그래도 원격 환경이나 매우 무거운 세대 설정에서는 `--no-gui`로 학습 후 결과만 확인하는 방식이 더 안정적입니다.

## 세대별 best 궤적 애니메이션

각 generation의 최고 개체 궤적을 시작 지점 확대 상태에서 실시간으로 재생할 수 있습니다.

```bash
python main.py --animate-best --animate-window-m 14 --animate-step-pause 0.008 --show-net
```

- `--animate-best`: 세대별 best trajectory 애니메이션 활성화
- `--show-net`: 실시간 신경망 활성도 시각화 활성화
- `--animate-window-m`: 확대 창 반경(미터)
- `--animate-step-pause`: 프레임 간 지연(작을수록 빠름)

## 파일 설명

- `main.py`: 트랙 로딩, 환경 시뮬레이션, NEAT 학습, 실시간 플롯
- `neat_config.ini`: `neat-python` 설정값
- `requirements.txt`: 의존성 목록
