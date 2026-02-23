# CUDA IPC GPU Memory Sharing PoC

GPU 메모리에 로드된 모델 가중치를 여러 추론 프로세스가 **zero-copy**로 공유하는 개념 검증(PoC)입니다. CUDA IPC를 통해 하나의 Weight Manager가 GPU 메모리를 소유하고, 다수의 Inference Worker가 복사 없이 같은 메모리를 참조하여 추론합니다.

## 목적

모델 서빙 환경에서 동일한 가중치를 프로세스마다 복제하면 GPU 메모리가 선형으로 증가합니다. 이 PoC는 CUDA IPC(`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)를 활용해 **N개 Worker가 1벌의 가중치만으로 동시 추론**할 수 있음을 검증합니다.

추가로 **Tensor Parallelism**을 구현하여 하나의 모델을 여러 GPU에 분산 배치하고, Worker가 모든 shard를 수집해 추론하는 Multi-WM 아키텍처를 검증합니다.

## 아키텍처

### 단일 GPU (1:N)

```
                  GPU 0
                ┌────────────────┐
                │  Model Weights │  <- 1벌만 존재
                └───────┬────────┘
                        │ CUDA IPC
        ┌───────────────┼───────────────┐
        v               v               v
   Worker 0        Worker 1        Worker 2
   (추론)          (추론)          (추론)
```

Weight Manager가 모델을 GPU에 로드하고, IPC 핸들을 ZMQ REP 소켓으로 서빙합니다. 각 Worker는 ZMQ REQ로 핸들을 받아 `torch.UntypedStorage._new_shared_cuda()`로 같은 GPU 메모리를 열고, `load_state_dict(assign=True)`로 zero-copy 주입합니다.

### Tensor Parallelism (N:1, N:M)

```
  WM-0 (cuda:0)               WM-1 (cuda:1)
  ┌──────────────┐            ┌──────────────┐
  │ rank=0 shard  │            │ rank=1 shard  │
  │ ipc:///wm_0   │            │ ipc:///wm_1   │
  └──────┬───────┘            └──────┬───────┘
         │ ZMQ REQ/REP                │
         └──────────┐    ┌───────────┘
                    v    v
              ┌──────────────┐
              │    Worker     │
              │  TPSimpleMLP  │
              │  forward(x)   │
              └──────────────┘
```

각 WM은 자신의 GPU에 모델의 일부(shard)만 보유합니다. Worker가 모든 WM에 연결하여 shard를 수집하고, TP-aware forward로 추론합니다.

**샤딩 전략 (SimpleMLP: 784 -> 256 -> 10)**:
- `fc1` (Column-parallel): weight `[256/N, 784]`, bias `[256/N]` -- 출력 차원 분할
- `fc2` (Row-parallel): weight `[10, 256/N]` -- 입력 차원 분할
- `fc2.bias`: rank 0만 소유 (all-reduce 후 한 번만 더함)

## 코드 구조

```
src/cuda_ipc_poc/
├── config.py              # ZMQ_ENDPOINT, TP_WORLD_SIZE 등 설정
├── ipc_channel.py         # ZMQ REP/REQ 기반 핸들 서버/클라이언트
├── handle_codec.py        # IPC 핸들 직렬화 (TP rank 메타데이터 포함)
├── weight_manager.py      # 모델 로드 -> IPC 핸들 내보내기 -> 서빙
├── inference_worker.py    # 핸들 수신 -> 텐서 복원 -> 추론 (단일/TP 모드)
├── tensor_parallel.py     # shard_model(), TPSimpleMLP
└── model.py               # SimpleMLP, ResNet18 팩토리

scripts/
├── run_weight_manager.py  # WM 프로세스 진입점
├── run_inference_worker.py # Worker 프로세스 진입점
└── run_verification.py    # 7개 자동 검증 테스트

tests/
├── test_ipc_channel.py    # ZMQ 전송 계층 테스트 (7개)
├── test_handle_codec.py   # 직렬화 + TP 포맷 테스트 (12개)
└── test_tensor_parallel.py # 샤딩/TP forward 테스트 (8개)
```

### 핵심 모듈 요약

| 모듈 | 역할 |
|------|------|
| `ipc_channel.py` | ZMQ REP(서버) / REQ(클라이언트). Poller 기반 1초 타임아웃, LINGER=0, 클라이언트는 RCVTIMEO=5s + 소켓 재생성으로 REQ/REP 상태 머신 리셋 |
| `handle_codec.py` | `_share_cuda_()` 8-tuple + size/stride/dtype를 직렬화. TP 모드에서는 `{"tp_rank", "tp_world_size", "handles"}` envelope 포맷. `tp_rank` 키 유무로 자동 감지 |
| `weight_manager.py` | `tp_world_size > 1`이면 `shard_model()`로 자기 rank의 shard만 IPC 내보내기. `ipc_collect()` 주기 호출로 죽은 worker 메모리 회수 |
| `inference_worker.py` | `endpoint`가 리스트이면 TP 모드: 각 WM에서 rank별 핸들 수신, 각 device에서 텐서 복원, `TPSimpleMLP` 구성 |
| `tensor_parallel.py` | `shard_model()`: Column/Row-parallel 분할 + `.contiguous()`. `TPSimpleMLP.forward()`: fc1 결과를 device_0에 모아 concat, fc2 각 rank에서 부분합, all-reduce + bias |

## 실행 방법

### 설치

```bash
pip install -e ".[dev]"
```

### 유닛 테스트 (GPU 불필요)

```bash
python -m pytest tests/ -v
# 27 tests: IPC 7 + Codec 12 + TP 8
```

### 단일 GPU 수동 실행

```bash
# 터미널 1: Weight Manager
python scripts/run_weight_manager.py --model mlp --device cuda:0

# 터미널 2: Inference Worker
python scripts/run_inference_worker.py --model mlp --device cuda:0
```

### Tensor Parallelism 수동 실행 (GPU 2개)

```bash
# 터미널 1: WM rank 0
python scripts/run_weight_manager.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/wm_0.zmq --tp-rank 0 --tp-world-size 2 --seed 42

# 터미널 2: WM rank 1
python scripts/run_weight_manager.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/wm_1.zmq --tp-rank 1 --tp-world-size 2 --seed 42

# 터미널 3: Worker (TP)
python scripts/run_inference_worker.py --model mlp \
  --endpoints "ipc:///tmp/wm_0.zmq,ipc:///tmp/wm_1.zmq"
```

### 자동 검증 스위트 (GPU 필요)

```bash
python scripts/run_verification.py --model mlp --device cuda:0 -v
```

## 검증 항목과 결과

7개 테스트로 핵심 속성을 자동 검증합니다. Test 7은 GPU 2개 이상일 때 자동 실행됩니다.

| # | 테스트 | 검증 내용 | 결과 |
|---|--------|-----------|------|
| 1 | Memory Sharing | 3개 Worker가 붙어도 GPU 메모리가 선형 증가하지 않음 | PASS |
| 2 | Zero-Copy | WM이 가중치를 수정하면 Worker에서 즉시 관찰됨 | PASS |
| 3 | Worker Lifecycle | 10회 Worker 생성/파괴 후에도 WM이 안정적 | PASS |
| 4 | Inference Accuracy | 같은 seed로 2개 Worker 추론 결과 일치 (max diff 0.00) | PASS |
| 5 | Memory Leak | 10회 warmup 후 40회 사이클에서 메모리 성장 0.0 MB | PASS |
| 6 | Crash Recovery | Worker `kill -9` 후 `ipc_collect()` -> 새 Worker 정상 동작 | PASS |
| 7 | Tensor Parallelism | 2-GPU TP 추론 결과 vs 단일 GPU 참조 모델 (max diff 5.96e-08) | PASS |

### Test 5 설계 노트

`nvidia-smi`는 CUDA 컨텍스트 메모리(프로세스당 ~100-200MB)를 포함합니다. 첫 수 회 worker spawn에서 일회성 증가 후 안정화되므로, 10회 warmup으로 컨텍스트 비용을 흡수한 뒤 baseline을 측정합니다. 이후 40회 사이클에서 성장률이 0에 가까운지 확인합니다.

### Test 7 설계 노트

두 WM 서브프로세스에 동일한 `--seed`를 전달하여 같은 가중치로 초기화합니다. Worker가 양쪽에서 shard를 수집하고 `TPSimpleMLP.forward()`로 추론한 결과를 단일 GPU 참조 모델과 `torch.allclose(atol=1e-5)`로 비교합니다.

## 기술적 주의 사항

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** -- PyTorch 2.2+의 expandable segments는 `cudaMemCreate`/`cudaMemMap`을 사용하여 `cudaIpcGetMemHandle`과 호환되지 않습니다. 반드시 CUDA 연산 전에 설정해야 합니다.

2. **ZMQ REQ/REP 상태 머신** -- `send()` 후 `recv()` 타임아웃 시 소켓이 깨진 상태가 됩니다. 재시도하려면 소켓을 닫고 새로 생성해야 합니다.

3. **TP shard 연속성** -- `torch.chunk()`는 뷰를 반환하므로 `.contiguous()` 호출 필수. CUDA IPC는 독립 할당된 메모리만 공유할 수 있습니다.

4. **다중 GPU `_lazy_init()`** -- Worker가 각 device의 IPC 핸들을 열기 전에 해당 device의 CUDA 컨텍스트를 초기화해야 합니다.

5. **하위 호환성** -- `tp_world_size=1` + 단일 endpoint는 기존 단일 GPU 동작과 동일합니다. handle_codec은 `tp_rank` 키 유무로 TP/legacy 포맷을 자동 감지합니다.

## 의존성

- `torch>=2.1.0` -- CUDA IPC, `_share_cuda_()`, `_new_shared_cuda()`
- `pyzmq>=25.0` -- IPC 핸들 전송
- `psutil` -- 프로세스/메모리 유틸리티
- `numpy` -- 수치 연산
- `pytest>=7.0` (dev) -- 테스트
- `torchvision` (optional) -- ResNet18 모델
