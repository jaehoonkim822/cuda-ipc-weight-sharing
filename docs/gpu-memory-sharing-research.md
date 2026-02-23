# NVIDIA GPU Inter-Process Memory Sharing: Research Document

> **Date**: 2026-02-23
> **Scope**: CUDA IPC mechanisms, PyTorch integration, weight manager/inference worker architecture

---

## Table of Contents

1. [Overview](#1-overview)
2. [CUDA IPC Mechanisms](#2-cuda-ipc-mechanisms)
3. [PyTorch Integration](#3-pytorch-integration)
4. [Architecture: Weight Manager + Inference Worker](#4-architecture-weight-manager--inference-worker)
5. [Feasibility Analysis](#5-feasibility-analysis)
6. [Proof-of-Concept Design](#6-proof-of-concept-design)
7. [Related Work in Production Systems](#7-related-work-in-production-systems)
8. [Conclusions and Recommendations](#8-conclusions-and-recommendations)
9. [References](#9-references)

---

## 1. Overview

이 문서는 NVIDIA GPU에서 서로 다른 프로세스 간에 GPU 메모리를 공유하는 방법을 조사하고, 특히 **수명(lifetime)이 다른 두 프로세스** - 하나는 모델 가중치를 관리하고, 다른 하나는 해당 가중치를 사용해 추론하는 구조 - 에서의 메모리 공유 가능성을 검증한다.

### Problem Statement

```
+--------------------+          +--------------------+
|  Weight Manager    |  shared  | Inference Worker   |
|  (long-lived)      |<-------->| (short-lived)      |
|                    |   GPU    |                    |
| - Load weights     |  memory  | - Read weights     |
| - Update weights   |          | - Run forward()    |
| - Manage versions  |          | - Return results   |
+--------------------+          +--------------------+
        |                              |
        |  lifetime: persistent        |  lifetime: per-request
        |  or until model swap         |  or per-batch
```

핵심 질문:
- GPU 메모리를 프로세스 간에 복사 없이 공유할 수 있는가?
- Producer(Weight Manager)가 살아있는 동안 Consumer(Inference Worker)가 자유롭게 생성/소멸할 수 있는가?
- PyTorch에서 이를 어떻게 구현할 수 있는가?

---

## 2. CUDA IPC Mechanisms

NVIDIA CUDA는 프로세스 간 GPU 메모리 공유를 위해 세 가지 주요 메커니즘을 제공한다.

### 2.1 Legacy CUDA IPC API

가장 기본적인 방식. `cudaMalloc`으로 할당한 메모리의 핸들을 다른 프로세스에 전달한다.

```
Producer Process:
  cudaMalloc(&devPtr, size)
  cudaIpcGetMemHandle(&handle, devPtr)
  --- handle을 OS IPC (pipe, socket, shared memory 등)로 전달 --->

Consumer Process:
  cudaIpcOpenMemHandle(&devPtr, handle, flags)
  // devPtr로 GPU 메모리 접근 가능
  cudaIpcCloseMemHandle(devPtr)
```

**제약 사항:**
- **Linux 전용** (Windows에서는 VMM API 사용 필요)
- `cudaMallocManaged` (Unified Memory)로 할당한 메모리는 지원하지 않음
- Sub-allocation 문제: `cudaMalloc`이 내부적으로 큰 블록에서 sub-allocate하면, IPC는 전체 블록을 공유하게 됨
  - **권장**: 2MiB 정렬된 크기의 할당만 공유
- 하나의 IPC 핸들은 다른 프로세스의 하나의 context에서만 열 수 있음
- **Producer가 살아있어야 함**: Producer가 메모리를 free하거나 종료하면 Consumer 측에서 undefined behavior 발생

### 2.2 Virtual Memory Management (VMM) API

CUDA 10.2+에서 도입된 저수준 가상 메모리 관리 API. OS의 파일 디스크립터(fd)를 통해 메모리를 공유한다.

```
Producer Process:
  cuMemCreate(&allocHandle, size, &prop, 0)    // 물리 메모리 할당
  cuMemExportToShareableHandle(&shHandle, allocHandle, handleType, 0)
  --- shHandle (fd)를 OS IPC로 전달 --->

Consumer Process:
  cuMemImportFromShareableHandle(&allocHandle, shHandle, handleType)
  cuMemAddressReserve(&ptr, size, 0, 0, 0)     // 가상 주소 공간 예약
  cuMemMap(ptr, size, 0, allocHandle, 0)        // 매핑
  cuMemSetAccess(ptr, size, &accessDesc, 1)     // 접근 권한 설정
```

**장점:**
- Linux + Windows 지원
- per-allocation 단위 IPC 제어 가능
- Fabric handle을 통한 multi-node IPC 지원
- **중요**: `CUmemAllocationProp::requestedHandleTypes`에 IPC 핸들 타입 명시 필요

**단점:**
- API가 복잡 (5단계 이상 필요)
- PyTorch에서 직접 노출되지 않음 (C/C++ extension 필요)

### 2.3 Stream-Ordered Memory Pool IPC

CUDA 11.2+에서 도입. Memory Pool을 통한 IPC 공유.

```
Producer Process:
  cudaMemPoolCreate(&pool, &poolProps)    // IPC 지원 풀 생성
  cudaMallocFromPoolAsync(&ptr, size, pool, stream)
  cudaMemPoolExportToShareableHandle(&shHandle, pool, handleType, 0)
  cudaMemPoolExportPointer(&exportData, ptr)
  --- shHandle + exportData 전달 --->

Consumer Process:
  cudaMemPoolImportFromShareableHandle(&pool, shHandle, handleType, 0)
  cudaMemPoolImportPointer(&ptr, pool, &exportData)
```

**특징:**
- 기본 메모리 풀은 IPC를 지원하지 않음 -> 명시적으로 `cudaMemPoolCreate` 사용
- VMM과 유사한 보안 이점
- 풀 수준에서 접근 권한 관리

### 2.4 Mechanism Comparison

| Feature | Legacy IPC | VMM IPC | Pool IPC |
|---|---|---|---|
| CUDA Version | 4.0+ | 10.2+ | 11.2+ |
| Platform | Linux only | Linux + Windows | Linux + Windows |
| Granularity | Allocation | Per-allocation | Pool + Pointer |
| Complexity | Low | High | Medium |
| Producer Lifetime | Must stay alive | fd-based (more flexible) | Pool must exist |
| PyTorch Support | Built-in | None (C ext needed) | None (C ext needed) |
| Multi-node | No | Yes (fabric handles) | No |

---

## 3. PyTorch Integration

### 3.1 Built-in CUDA Tensor Sharing (torch.multiprocessing)

PyTorch는 `torch.multiprocessing`을 통해 Legacy CUDA IPC를 내부적으로 사용한다.

**동작 방식:**
```python
# 내부적으로 사용되는 핵심 메서드
storage = tensor.untyped_storage()
metadata = storage._share_cuda_()
# metadata: (device, handle, storage_size_bytes, storage_offset_bytes,
#            ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)

# Consumer 측에서 복원
tensor = torch._utils._rebuild_tensor(
    torch.Storage._new_shared_cuda(
        *metadata
    ),
    storage_offset, size, stride
)
```

**torch.multiprocessing.Queue를 통한 공유:**
```python
import torch
import torch.multiprocessing as mp

def producer(queue):
    tensor = torch.randn(1000, 1000, device='cuda')
    queue.put(tensor)  # IPC 핸들이 자동 생성됨
    # 주의: consumer가 사용을 마칠 때까지 이 프로세스가 살아있어야 함

def consumer(queue):
    tensor = queue.get()  # IPC 핸들로부터 복원
    result = tensor.sum()
    return result

if __name__ == '__main__':
    mp.set_start_method('spawn')  # CUDA는 fork 지원 안함
    q = mp.Queue()
    p = mp.Process(target=consumer, args=(q,))
    p.start()
    producer(q)
    p.join()
```

**Reference Counting 메커니즘:**
- `CudaIPCSentData`: Producer 측에서 DataPtr를 감싸는 구조체. 즉시 해제하지 않고 참조 카운트 확인
- `CudaIPCReceivedData`: Consumer 측 구조체. 소멸 시 참조 카운트 감소
- `CudaIPCSentDataLimbo`: Producer가 더 이상 사용하지 않지만 Consumer가 아직 참조 중인 데이터를 보관
- `torch.cuda.ipc_collect()`: Limbo 리스트를 스캔하여 해제 가능한 블록 정리

### 3.2 핵심 제약 사항

1. **Producer 수명 제약**: CUDA 텐서를 공유한 Producer 프로세스는 모든 Consumer가 텐서 참조를 해제할 때까지 살아있어야 한다
2. **share_memory_() 는 CUDA에서 no-op**: CPU 텐서용 공유 메모리 메커니즘이며, CUDA 텐서에는 효과 없음
3. **독립 프로세스 간 공유 제한**: share_memory_()는 parent-child 관계에서만 작동. 독립적으로 시작된 프로세스 간에는 Queue나 직접 IPC 핸들 전달 필요
4. **Start method**: `spawn` 또는 `forkserver` 필수 (`fork`는 CUDA와 호환 불가)
5. **비정상 종료**: Consumer가 fatal signal로 종료되면 참조 카운트 정리 불가 -> 메모리 누수

### 3.3 저수준 접근: _share_cuda_() 직접 사용

독립 프로세스 간 공유를 위해 IPC 핸들을 직접 추출하여 전달할 수 있다:

```python
import torch

# === Producer ===
model = load_model()
model.cuda()

# 각 파라미터의 IPC 핸들 추출
handles = {}
for name, param in model.named_parameters():
    storage = param.data.untyped_storage()
    metadata = storage._share_cuda_()
    handles[name] = {
        'metadata': metadata,
        'size': param.size(),
        'stride': param.stride(),
        'dtype': param.dtype,
    }

# socket, shared file, redis 등으로 handles 전달
send_via_ipc(handles)

# === Consumer ===
handles = receive_via_ipc()

state_dict = {}
for name, info in handles.items():
    storage = torch.UntypedStorage._new_shared_cuda(*info['metadata'])
    tensor = torch.empty(0, dtype=info['dtype'], device='cuda')
    tensor.set_(storage, 0, info['size'], info['stride'])
    state_dict[name] = tensor

model = create_model_skeleton()
model.load_state_dict(state_dict, assign=True)
```

### 3.4 TorchStore (RFC, 미구현)

PyTorch RFC #64932에서 제안된 key-value 형태의 공유 메모리 텐서 저장소.

- CPU + CUDA 텐서 지원
- Daemon mode로 독립 프로세스 간 공유 가능
- 참조 카운팅으로 안전한 메모리 관리
- **현재 상태**: 2021년 제안 이후 아직 메인 리포지토리에 미통합 (2025년 기준 open)

---

## 4. Architecture: Weight Manager + Inference Worker

### 4.1 Design Goals

```
요구사항:
1. Weight Manager (WM): 모델 가중치를 GPU에 로드하고 관리하는 장수명 프로세스
2. Inference Worker (IW): 추론 요청을 처리하는 단수명 프로세스 (요청/배치 단위)
3. WM이 살아있는 동안 IW는 자유롭게 생성/소멸
4. GPU 메모리 복사 없이 가중치 공유
5. WM이 가중치를 업데이트(hot-swap)할 수 있어야 함
```

### 4.2 Proposed Architecture

```
                    +------------------------------+
                    |      Coordinator / Router     |
                    |  (요청 분배, worker 관리)       |
                    +------+-----------+-----------+
                           |           |
              +------------v---+  +----v------------+
              | Weight Manager |  | Inference Worker |
              |                |  |   (Pool: N개)    |
              |                |  |                  |
              | 1. Load model  |  | 1. Receive IPC   |
              | 2. Allocate on |  |    handles       |
              |    GPU         |  | 2. Map to local  |
              | 3. Export IPC  |  |    address space  |
              |    handles     |  | 3. Run inference  |
              | 4. Serve       |  | 4. Return result  |
              |    handles to  |  | 5. Close handle   |
              |    workers     |  | 6. Exit           |
              | 5. Manage      |  |                   |
              |    versions    |  |                   |
              +----------------+  +-------------------+
                     |                    ^
                     |   GPU Memory       |
                     v   (shared)         |
              +----------------------------+
              |  GPU Device Memory         |
              |  +--------------------+    |
              |  | Model Weights v1   |<---|-- IPC Handle
              |  +--------------------+    |
              |  +--------------------+    |
              |  | Model Weights v2   |    |  (hot-swap 용)
              |  +--------------------+    |
              +----------------------------+
```

### 4.3 Lifetime Management Strategy

**핵심 원칙**: Weight Manager가 항상 Inference Worker보다 오래 살아야 한다.

```
Timeline:
  WM  ====================================================  (persistent)
  IW1      ====                                              (request 1)
  IW2           ======                                       (request 2)
  IW3                    ========                            (request 3)
  IW4                              ====                      (request 4)
                    ^
                    | Model v1 -> v2 hot-swap
                    | (WM이 v2를 로드한 후, 이후 IW에게 v2 핸들 제공)
```

이 구조에서 Legacy CUDA IPC의 "Producer must stay alive" 제약은 **자연스럽게 충족**된다:
- Weight Manager = Producer (장수명) -> 항상 살아있음
- Inference Worker = Consumer (단수명) -> 자유롭게 생성/소멸

### 4.4 Version Management (Hot-Swap)

```python
class WeightManager:
    def __init__(self):
        self.versions = {}  # version -> {name: ipc_handles}
        self.current_version = None

    def load_model(self, model_path, version_tag):
        model = load_model(model_path)
        model.cuda()

        handles = {}
        for name, param in model.named_parameters():
            storage = param.data.untyped_storage()
            handles[name] = {
                'metadata': storage._share_cuda_(),
                'size': param.size(),
                'stride': param.stride(),
                'dtype': param.dtype,
            }

        self.versions[version_tag] = {
            'model': model,    # 참조 유지 (GC 방지)
            'handles': handles,
        }
        self.current_version = version_tag

    def get_current_handles(self):
        return self.versions[self.current_version]['handles']

    def retire_version(self, version_tag):
        # 해당 버전을 사용하는 worker가 없을 때만 호출
        del self.versions[version_tag]
        torch.cuda.ipc_collect()
```

---

## 5. Feasibility Analysis

### 5.1 Legacy CUDA IPC 기반 (권장)

| 항목 | 평가 |
|---|---|
| 가능성 | **높음** - WM이 장수명이므로 producer lifetime 제약 자연 충족 |
| PyTorch 지원 | **내장** - `_share_cuda_()` + `_new_shared_cuda()` |
| 복잡도 | **중간** - IPC 핸들 전달 메커니즘 (socket/pipe/redis) 구현 필요 |
| 성능 | **우수** - zero-copy, GPU 메모리 복사 없음 |
| 안정성 리스크 | IW 비정상 종료 시 메모리 누수 가능 -> `ipc_collect()` 주기적 호출로 완화 |
| 플랫폼 | Linux 전용 |

### 5.2 VMM API 기반 (고급)

| 항목 | 평가 |
|---|---|
| 가능성 | **높음** - 가장 유연한 lifetime 관리 |
| PyTorch 지원 | **없음** - C extension 직접 작성 필요 |
| 복잡도 | **높음** - 5단계 이상의 CUDA Driver API 호출 |
| 성능 | **우수** - zero-copy |
| 장점 | fd 기반으로 producer 종료 후에도 consumer가 fd를 보유하면 메모리 유지 가능할 수 있음 |
| 플랫폼 | Linux + Windows |

### 5.3 NVIDIA MPS (보완적)

MPS는 메모리 "공유"가 아닌 GPU 컴퓨트 리소스의 다중 프로세스 공유에 가까움.
하지만 다수의 Inference Worker가 동시에 GPU를 사용할 때 컨텍스트 스위칭 오버헤드를 줄여주므로 보완적으로 사용 가치 있음.

### 5.4 결론

**Legacy CUDA IPC가 이 사용 사례에 가장 적합하다.** 이유:
1. Weight Manager가 장수명이므로 producer lifetime 제약이 자연스럽게 충족됨
2. PyTorch가 내장 지원하므로 별도 C extension 불필요
3. Zero-copy로 최적 성능
4. 구현 복잡도가 합리적

---

## 6. Proof-of-Concept Design

### 6.1 Component Overview

```
poc/
  weight_manager.py      # 장수명 프로세스: 모델 로드 + IPC 핸들 서빙
  inference_worker.py    # 단수명 프로세스: IPC 핸들로 추론 실행
  ipc_channel.py         # IPC 핸들 전달 메커니즘 (socket 기반)
  run_poc.py             # 전체 PoC 실행 스크립트
```

### 6.2 IPC Channel (Handle Transfer)

IPC 핸들 자체는 바이트열이므로, 프로세스 간 전달에는 어떤 OS IPC든 사용 가능:
- Unix Domain Socket (가장 간단)
- Shared Memory File
- Redis/ZMQ (분산 환경)
- Named Pipe

```python
# ipc_channel.py (Unix Domain Socket 기반 예시)
import socket
import json
import os
import struct

SOCKET_PATH = '/tmp/weight_manager.sock'

class HandleServer:
    """Weight Manager 측에서 IPC 핸들을 서빙"""
    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        self.handles_bytes = None

    def start(self, serialized_handles):
        self.handles_bytes = serialized_handles
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(16)

    def serve_forever(self):
        while True:
            conn, _ = self.sock.accept()
            length = len(self.handles_bytes)
            conn.sendall(struct.pack('>Q', length))
            conn.sendall(self.handles_bytes)
            conn.close()

class HandleClient:
    """Inference Worker 측에서 IPC 핸들을 수신"""
    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path

    def get_handles(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        length = struct.unpack('>Q', sock.recv(8))[0]
        data = b''
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                break
            data += chunk
        sock.close()
        return data  # caller deserializes
```

### 6.3 Weight Manager

```python
# weight_manager.py
import torch

class WeightManager:
    def __init__(self, model_cls, model_path, device='cuda:0'):
        self.device = device
        self.model = model_cls()
        self.model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.model.to(device)
        self.model.requires_grad_(False)
        self._export_handles()

    def _export_handles(self):
        self.handles = {}
        for name, param in self.model.named_parameters():
            storage = param.data.untyped_storage()
            self.handles[name] = {
                'metadata': storage._share_cuda_(),
                'size': list(param.size()),
                'stride': list(param.stride()),
                'dtype': str(param.dtype),
                'storage_offset': param.storage_offset(),
            }
        # buffer도 공유 (BatchNorm running_mean/var 등)
        for name, buf in self.model.named_buffers():
            storage = buf.data.untyped_storage()
            self.handles['__buffer__' + name] = {
                'metadata': storage._share_cuda_(),
                'size': list(buf.size()),
                'stride': list(buf.stride()),
                'dtype': str(buf.dtype),
                'storage_offset': buf.storage_offset(),
            }

    def get_handles(self):
        return self.handles

    def run_server(self, socket_path='/tmp/weight_manager.sock'):
        from ipc_channel import HandleServer
        server = HandleServer(socket_path)
        # Serialize handles appropriately for IPC transfer
        import io
        buf = io.BytesIO()
        torch.save(self.handles, buf)
        server.start(buf.getvalue())
        print("Weight Manager serving on " + socket_path)
        server.serve_forever()
```

### 6.4 Inference Worker

```python
# inference_worker.py
import torch
import io

class InferenceWorker:
    def __init__(self, model_cls, socket_path='/tmp/weight_manager.sock'):
        from ipc_channel import HandleClient
        client = HandleClient(socket_path)
        raw_bytes = client.get_handles()
        raw_handles = torch.load(io.BytesIO(raw_bytes), weights_only=False)

        # IPC 핸들에서 텐서 복원
        params = {}
        buffers = {}
        for name, info in raw_handles.items():
            storage = torch.UntypedStorage._new_shared_cuda(*info['metadata'])
            dtype = getattr(torch, info['dtype'].replace('torch.', ''))
            tensor = torch.empty([], dtype=dtype, device='cuda')
            tensor.set_(
                storage,
                info['storage_offset'],
                info['size'],
                info['stride'],
            )
            if name.startswith('__buffer__'):
                buffers[name.replace('__buffer__', '', 1)] = tensor
            else:
                params[name] = tensor

        # 모델 skeleton에 공유 가중치 주입
        self.model = model_cls()
        self.model.to('cuda')
        combined = {}
        combined.update(params)
        combined.update(buffers)
        self.model.load_state_dict(combined, strict=False, assign=True)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def infer(self, input_tensor):
        return self.model(input_tensor)
```

### 6.5 검증 항목

| 검증 항목 | 방법 |
|---|---|
| 메모리 공유 확인 | `nvidia-smi`로 GPU 메모리 사용량 확인. N개 worker가 있어도 모델 1개분의 메모리만 사용 |
| Zero-copy 확인 | 공유된 텐서의 data_ptr 비교 (같은 GPU 주소여야 함) |
| Worker 생성/소멸 | Worker를 반복 생성/종료하면서 WM 안정성 확인 |
| 추론 정확도 | 공유 가중치로 추론한 결과와 단일 프로세스 추론 결과 비교 |
| 메모리 누수 | 장시간 Worker 생성/소멸 반복 후 GPU 메모리 증가 여부 확인 |
| 비정상 종료 | Worker를 kill -9로 강제 종료한 후 WM 측 ipc_collect() 호출로 정리 가능한지 확인 |

---

## 7. Related Work in Production Systems

### 7.1 vLLM

- 프론트엔드, 코디네이터, 추론 워커가 별도 프로세스로 실행
- 텐서 병렬리즘에서 NCCL로 GPU 간 통신
- Shared Memory IPC Caching으로 멀티모달 입력 전달 최적화
- KV 캐시 공유에 `/dev/shm` 활용

### 7.2 NVIDIA Dynamo + NIXL

- NIXL: GPU HBM, CPU DRAM, SSD, 네트워크 스토리지 간 통합 데이터 이동 API
- Disaggregated serving (prefill/decode 분리)에서 KV 캐시 전달에 사용
- NVLink, InfiniBand, Ethernet 지원

### 7.3 TorchServe + NVIDIA MPS

- TorchServe with MPS로 다수 워커가 GPU를 동시 활용
- MPS는 컴퓨트 리소스 공유이지 메모리 공유는 아님
- 각 워커가 별도 모델 복사본을 보유하는 기존 패턴

---

## 8. Conclusions and Recommendations

### 8.1 핵심 결론

1. **컨셉은 유효하다.** Weight Manager(장수명) + Inference Worker(단수명) 구조에서 Legacy CUDA IPC를 사용한 GPU 메모리 공유는 기술적으로 가능하며, CUDA IPC의 "producer must stay alive" 제약과 자연스럽게 부합한다.

2. **PyTorch에서 구현 가능하다.** `_share_cuda_()` + `_new_shared_cuda()` API를 사용하면 별도의 C extension 없이도 프로세스 간 GPU 텐서 공유가 가능하다. 단, 이들은 internal API이므로 PyTorch 버전 간 호환성 주의 필요.

3. **주의 사항이 존재한다.**
   - Worker 비정상 종료 시 메모리 누수 가능 -> `ipc_collect()` 주기적 호출 필요
   - `_share_cuda_()` 메타데이터의 프로세스 간 전달 채널 구현 필요
   - sub-allocation 이슈로 예상보다 많은 메모리가 공유될 수 있음 (2MiB 정렬 권장)

### 8.2 권장 다음 단계

1. **PoC 구현**: 위 6장의 설계를 기반으로 간단한 모델(ResNet-18 등)로 PoC 구현
2. **메모리 사용량 벤치마크**: 공유 vs 비공유 환경의 GPU 메모리 사용량 비교
3. **안정성 테스트**: Worker 반복 생성/소멸, 비정상 종료 시나리오 테스트
4. **VMM API 탐색** (선택): 더 유연한 lifetime 관리가 필요하면 CUDA VMM API 기반 C extension 개발 고려

---

## 9. References

### NVIDIA Official Documentation
- [CUDA IPC - Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
- [CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [CUDA Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [NVIDIA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html)
- [NVIDIA MPS - When to Use](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
- [Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
- [Boost GPU Memory with CUDA MPS](https://developer.nvidia.com/blog/boost-gpu-memory-performance-with-no-code-changes-using-nvidia-cuda-mps/)

### PyTorch Documentation and Source
- [PyTorch Multiprocessing](https://docs.pytorch.org/docs/stable/multiprocessing.html)
- [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [PyTorch CUDA Multiprocessing Internals](https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/cuda_multiprocessing.md)
- [PyTorch multiprocessing/reductions.py](https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/reductions.py)
- [TorchStore RFC #64932](https://github.com/pytorch/pytorch/issues/64932)
- [torch.cuda.ipc_collect](https://docs.pytorch.org/docs/stable/generated/torch.cuda.ipc_collect.html)

### PyTorch Forums and Issues
- [Using CUDA IPC memory handles in pytorch](https://discuss.pytorch.org/t/using-cuda-ipc-memory-handles-in-pytorch/17548)
- [Inter-process sharing CUDA Tensor](https://discuss.pytorch.org/t/inter-process-sharing-cuda-tensor/224560)
- [How to share a CUDA tensor between processes](https://discuss.pytorch.org/t/how-to-share-a-cuda-tensor-between-processes/221703)
- [Issue #149187: Shared CUDA Tensor Reference Counting](https://github.com/pytorch/pytorch/issues/149187)

### Production Systems
- [vLLM Shared Memory IPC Caching](https://blog.vllm.ai/2025/11/13/shm-ipc-cache.html)
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [NVIDIA Dynamo Blog Post](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NIXL - NVIDIA Inference Xfer Library](https://github.com/ai-dynamo/nixl)
- [TorchServe with NVIDIA MPS](https://docs.pytorch.org/serve/hardware_support/nvidia_mps.html)

### NVIDIA Developer Forums
- [CUDA IPC vs NVSHMEM](https://forums.developer.nvidia.com/t/cuda-ipc-vs-nvshmem-for-shared-memory-between-applications/237071)
- [How to access GPU memory between processes](https://forums.developer.nvidia.com/t/how-to-access-gpu-memory-between-processes/259593)
- [Share CUDA memory between different system processes](https://forums.developer.nvidia.com/t/share-cuda-memory-between-different-system-processes/191906)
