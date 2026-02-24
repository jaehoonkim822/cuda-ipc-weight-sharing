# CUDA IPC GPU Memory Sharing

GPU 메모리에 로드된 모델 가중치를 여러 추론 프로세스가 **zero-copy**로 공유하는 시스템입니다.

CUDA IPC(`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)를 통해 Weight Manager가 GPU 메모리를 소유하고, Inference Worker들이 복사 없이 같은 메모리를 참조합니다. **ModelRegistry**에 모델을 등록하면 로드 → TP 샤딩 → IPC 서빙 → 추론까지 동일한 파이프라인을 재사용할 수 있습니다.

## 아키텍처

### 단일 GPU (1:N) — 메모리 공유

```
                  GPU 0
                ┌────────────────┐
                │  Model Weights │  ← 1벌만 존재
                └───────┬────────┘
                        │ CUDA IPC
        ┌───────────────┼───────────────┐
        v               v               v
   Worker 0        Worker 1        Worker 2
```

Weight Manager가 모델을 GPU에 로드하고, IPC 핸들을 ZMQ로 서빙합니다. Worker는 핸들로 같은 GPU 메모리를 열어 `load_state_dict(assign=True)`로 zero-copy 주입합니다.

### Tensor Parallelism — 다중 GPU

모델이 단일 GPU에 들어가지 않을 때, 파라미터를 여러 GPU에 분할합니다. 두 가지 토폴로지를 지원합니다.

**집중형 (N:1)** — 1 Worker가 모든 WM에서 shard를 수집하여 로컬 TP forward 수행:

```
  WM-0 (cuda:0)               WM-1 (cuda:1)
  ┌──────────────┐            ┌──────────────┐
  │ rank=0 shard  │            │ rank=1 shard  │
  └──────┬───────┘            └──────┬───────┘
         └──────────┐    ┌───────────┘
                    v    v
              ┌──────────────┐
              │    Worker     │
              │  TP forward   │
              └──────────────┘
```

**분산형 (1:1)** — N Worker가 각각 1 WM을 전담하고 NCCL 집합 통신으로 분산 forward 수행:

```
  WM-0 (cuda:0)               WM-1 (cuda:1)
  ┌──────────────┐            ┌──────────────┐
  │ rank=0 shard  │            │ rank=1 shard  │
  └──────┬───────┘            └──────┬───────┘
         v                            v
  ┌──────────────┐            ┌──────────────┐
  │  Worker-0     │◄──NCCL───►│  Worker-1     │
  │  rank=0       │ all_gather │  rank=1       │
  └──────────────┘ all_reduce └──────────────┘
```

### TP 샤딩 전략

`BaseTensorParallelHandler`로 모델별 Column/Row-parallel 패턴을 선언적으로 정의합니다. fnmatch 글로브를 지원하므로 `layers.*.q_proj.weight` 같은 패턴으로 전체 레이어를 한 번에 지정할 수 있습니다.

| 모델 | Column-parallel (dim=0) | Row-parallel (dim=1) | 비고 |
|------|------------------------|---------------------|------|
| SimpleMLP | `fc1.weight`, `fc1.bias` | `fc2.weight` | `fc2.bias`는 rank 0 only |
| TinyTransformer | `q/k/v_proj`, `gate/up_proj` | `o_proj`, `down_proj` | forward에 `all_reduce` 내장 |

## 코드 구조

```
src/cuda_ipc_poc/
├── model_spec.py          # ModelSpec, WeightLoader, ModelRegistry
├── tp_handler.py          # BaseTensorParallelHandler, SimpleMlpTPHandler
├── weight_manager.py      # 모델 로드 → TP 샤딩 → IPC 핸들 서빙
├── inference_worker.py    # 핸들 수신 → 텐서 복원 → 모델 구성/추론
├── ipc_channel.py         # ZMQ REP/REQ 핸들 서버/클라이언트
├── handle_codec.py        # IPC 핸들 직렬화 (TP envelope 자동 감지)
├── tensor_parallel.py     # TPSimpleMLP (로컬 루프), DistributedTPForward (NCCL)
├── model.py               # SimpleMLP, ResNet18 정의 + Registry 등록
├── config.py              # ZMQ_ENDPOINT, DEVICE, CUDA_ALLOC_CONF
└── models/
    └── tiny_llama.py      # TinyTransformer (LLaMA-style) + TPHandler

scripts/
├── run_weight_manager.py  # WM 프로세스 진입점
├── run_inference_worker.py # Worker 프로세스 진입점
└── run_verification.py    # GPU 통합 검증 스위트

tests/                     # 53개 유닛 테스트 (CPU only)
├── test_model_spec.py     # Registry, TPHandler 패턴 매칭 (13)
├── test_tiny_transformer.py # TinyTransformer forward + TP (11)
├── test_tensor_parallel.py # SimpleMLP 샤딩 + TP forward (10)
├── test_handle_codec.py   # 직렬화 + TP 포맷 (12)
└── test_ipc_channel.py    # ZMQ 전송 계층 (7)
```

### 핵심 모듈

| 모듈 | 역할 |
|------|------|
| `model_spec.py` | `ModelSpec` (model_cls, factory, sample_input_fn, tp_handler, weight_loader)과 `ModelRegistry`로 모델 이름 기반 조회. `SafetensorsLoader`로 HF 가중치 로딩 |
| `tp_handler.py` | fnmatch 글로브 패턴으로 Column/Row-parallel 자동 분할. 매칭 안 되는 파라미터는 replicate |
| `weight_manager.py` | `ModelRegistry.get()` → 모델 생성/가중치 로딩, `tp_handler.process_state_dict()`로 샤딩, `_share_cuda_()`로 IPC 핸들 추출 후 ZMQ 서빙 |
| `inference_worker.py` | 3가지 모드 지원: 단일 GPU(`load_state_dict`), 집중형 TP(`TPSimpleMLP`), 분산형 TP(`DistributedTPForward`). TP-aware forward 내장 모델은 sharded state_dict를 직접 주입 |
| `ipc_channel.py` | ZMQ REP/REQ. Poller 기반 타임아웃, 클라이언트는 소켓 재생성으로 REQ/REP 상태 머신 복구 |
| `handle_codec.py` | `_share_cuda_()` 8-tuple + size/stride/dtype 직렬화. TP 모드에서 envelope 포맷 자동 감지 |
| `models/tiny_llama.py` | 2-layer LLaMA-style Transformer (RMSNorm, SwiGLU, multi-head attention). forward에 `dist.all_reduce` 내장 |

## 커스텀 모델 등록

`ModelSpec`을 등록하면 동일한 파이프라인으로 임의의 모델을 서빙할 수 있습니다.

```python
from cuda_ipc_poc.model_spec import ModelRegistry, ModelSpec, SafetensorsLoader
from cuda_ipc_poc.tp_handler import BaseTensorParallelHandler

class MyModelTPHandler(BaseTensorParallelHandler):
    COLUMN_WISE_PARAMS = ["layers.*.q_proj.weight", "layers.*.gate_proj.weight"]
    ROW_WISE_PARAMS = ["layers.*.o_proj.weight", "layers.*.down_proj.weight"]
    HAS_BUILTIN_TP_FORWARD = True  # forward에 dist.all_reduce 포함 시

ModelRegistry.register("my-model", ModelSpec(
    model_cls=MyModel,
    model_factory=lambda: MyModel(hidden=4096, layers=32),
    sample_input_fn=lambda device, batch_size=1: torch.randint(0, 32000, (batch_size, 128), device=device),
    tp_handler=MyModelTPHandler(),
    weight_loader=SafetensorsLoader(),
))
```

**TP-aware forward 내장 모델** (LLaMA 등)은 Worker가 모델 shell을 만들고 sharded state_dict를 직접 주입하면 됩니다. forward에 `all_reduce`가 없는 모델은 `TPSimpleMLP`/`DistributedTPForward` 래퍼를 사용합니다.

## 실행 방법

### 설치

```bash
pip install -e ".[dev]"
```

### 테스트

```bash
# 유닛 테스트 (GPU 불필요, 53개)
python -m pytest tests/ -v

# GPU 통합 검증 (9개, GPU 수에 따라 자동 스케일)
python scripts/run_verification.py --model mlp --device cuda:0 -v
```

### 단일 GPU

```bash
# Weight Manager
python scripts/run_weight_manager.py --model mlp --device cuda:0

# Inference Worker (별도 터미널)
python scripts/run_inference_worker.py --model mlp --device cuda:0

# 외부 가중치 로딩
python scripts/run_weight_manager.py --model my-model --model-path /path/to/weights.safetensors
```

### 집중형 Tensor Parallelism (GPU 2개)

```bash
# WM rank 0, 1
python scripts/run_weight_manager.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/wm_0.zmq --tp-rank 0 --tp-world-size 2 --seed 42
python scripts/run_weight_manager.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/wm_1.zmq --tp-rank 1 --tp-world-size 2 --seed 42

# Worker — 양쪽 shard를 수집
python scripts/run_inference_worker.py --model mlp \
  --endpoints "ipc:///tmp/wm_0.zmq,ipc:///tmp/wm_1.zmq"
```

### 분산형 Tensor Parallelism (GPU 2개)

```bash
# WM rank 0, 1
python scripts/run_weight_manager.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/dtp_wm_0.zmq --tp-rank 0 --tp-world-size 2 --seed 42
python scripts/run_weight_manager.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/dtp_wm_1.zmq --tp-rank 1 --tp-world-size 2 --seed 42

# Worker rank 0, 1 — 각각 자기 WM에만 연결, NCCL로 통신
python scripts/run_inference_worker.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/dtp_wm_0.zmq \
  --distributed-tp --tp-rank 0 --tp-world-size 2
python scripts/run_inference_worker.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/dtp_wm_1.zmq \
  --distributed-tp --tp-rank 1 --tp-world-size 2
```

## 검증

GPU 통합 검증 스위트 (`run_verification.py`)가 9개 테스트를 자동 실행합니다. GPU 수에 따라 TP 테스트가 자동으로 활성화됩니다.

| # | 테스트 | 검증 내용 | GPU |
|---|--------|-----------|-----|
| 1 | Memory Sharing | 3 Worker 연결 시 GPU 메모리 선형 증가 없음 | 1+ |
| 2 | Zero-Copy | WM에서 가중치 수정 → Worker에서 즉시 관찰 | 1+ |
| 3 | Worker Lifecycle | 10회 Worker 생성/파괴 후 WM 안정성 | 1+ |
| 4 | Inference Accuracy | 동일 seed 2 Worker 추론 결과 일치 | 1+ |
| 5 | Memory Leak | warmup 10회 후 40 사이클 메모리 성장 0 MB | 1+ |
| 6 | Crash Recovery | Worker `kill -9` → `ipc_collect()` → 새 Worker 정상 | 1+ |
| 7 | Centralized TP | 1 Worker, 2 WM shard 수집, 참조 모델 대비 atol=1e-5 | 2+ |
| 8 | Distributed TP (2) | 2 WM + 2 Worker, NCCL 통신, 전 rank 출력 일치 | 2+ |
| 9 | Distributed TP (4) | 4 WM + 4 Worker, NCCL 통신, 전 rank 출력 일치 | 4+ |

> **Memory Leak 테스트 설계**: `nvidia-smi`는 CUDA 컨텍스트 메모리(프로세스당 ~100-200MB)를 포함합니다. warmup 10회로 일회성 증가를 흡수한 뒤 baseline을 잡고, 이후 40회 사이클에서 성장률이 0인지 확인합니다.

## 기술적 제약

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** — PyTorch 2.2+의 expandable segments는 `cudaMemCreate`/`cudaMemMap`을 사용하여 `cudaIpcGetMemHandle`과 호환되지 않습니다. 모든 CUDA 연산 이전에 설정해야 합니다.

2. **Linux only** — Windows는 CUDA IPC legacy API를 지원하지 않습니다.

3. **Contiguous memory** — `torch.chunk()`가 반환하는 뷰는 `.contiguous()` 호출 필수. CUDA IPC는 독립 할당된 연속 메모리만 공유할 수 있습니다.

4. **다중 GPU CUDA 컨텍스트** — Worker가 IPC 핸들을 열기 전에 해당 device의 CUDA 컨텍스트를 `_lazy_init()`으로 초기화해야 합니다.

5. **분산 TP 환경변수** — `DistributedTPForward` 사용 시 `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`가 필요합니다. CLI의 `--distributed-tp` 플래그가 자동 설정합니다.

6. **PyTorch 내부 API** — `_share_cuda_()`, `_new_shared_cuda_()`는 공식 API가 아닙니다. PyTorch 버전 업그레이드 시 호환성 확인이 필요합니다.

## 의존성

| 패키지 | 용도 |
|--------|------|
| `torch>=2.1.0` | CUDA IPC, `_share_cuda_()`, `_new_shared_cuda_()` |
| `pyzmq>=25.0` | IPC 핸들 전송 |
| `psutil` | 프로세스/메모리 유틸리티 |
| `numpy` | 수치 연산 |
| `pytest>=7.0` (dev) | 테스트 |
| `torchvision` (optional) | ResNet18 모델 |
| `safetensors` (optional) | HuggingFace 가중치 로딩 |
