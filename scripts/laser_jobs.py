"""
Job registry en memoria para procesamiento asincrónico con progreso por SSE.

Diseño:
- Cada job tiene un `JobState` mutable que el worker thread actualiza periódicamente.
- El endpoint SSE consulta el estado y lo emite al cliente cada N segundos.
- Los jobs se limpian automáticamente tras `JOB_TTL_SECONDS` o cuando hay más
  de `MAX_JOBS_IN_MEMORY` (FIFO).

NO es production-grade (no persiste a disco, no escala a múltiples replicas). Pensado
para uso local single-process. Para una versión distribuida usaríamos Redis/Celery.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

JOB_TTL_SECONDS = 3600  # Jobs viejos se limpian tras 1 hora
MAX_JOBS_IN_MEMORY = 100  # Si se llena, se borra el job más viejo (FIFO).
MAX_SCORE_HISTORY = 60   # Cap del sparkline (puntos enviados al cliente)
MAX_LOG_LINES = 80       # Cap del log de eventos del worker
JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


@dataclass
class JobState:
    """Estado mutable de un job asincrónico. El worker actualiza; el SSE lo lee."""

    job_id: str
    status: JobStatus = "queued"
    current: int = 0
    total: int = 0
    best_score: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    stage: str = ""  # "preprocess", "refine", "encode", etc.
    error_message: str | None = None
    image_bytes: bytes | None = None
    image_headers: dict[str, str] = field(default_factory=dict)
    sim_image_bytes: bytes | None = None  # opcional: simulación adjunta
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Telemetría adicional (v1.4.1+)
    memory_mb: float | None = None
    cpu_pct: float | None = None
    seconds_per_candidate: float | None = None  # promedio: elapsed / current
    last_candidate_seconds: float | None = None  # tiempo entre los dos últimos updates
    score_history: list[float] = field(default_factory=list)  # best_score acumulado por iter
    log_lines: list[dict] = field(default_factory=list)  # {t: float, msg: str, kind: 'info'|'warn'|'error'}
    _cancel_requested: bool = False

    def to_progress_dict(self) -> dict:
        """Snapshot serializable a JSON para SSE (sin bytes binarios)."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "best_score": self.best_score,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "stage": self.stage,
            "error_message": self.error_message,
            "progress_pct": round(100.0 * self.current / self.total, 1) if self.total > 0 else 0.0,
            "memory_mb": round(self.memory_mb, 1) if self.memory_mb is not None else None,
            "cpu_pct": round(self.cpu_pct, 1) if self.cpu_pct is not None else None,
            "seconds_per_candidate": (
                round(self.seconds_per_candidate, 2) if self.seconds_per_candidate is not None else None
            ),
            "last_candidate_seconds": (
                round(self.last_candidate_seconds, 2) if self.last_candidate_seconds is not None else None
            ),
            "score_history": [round(float(s), 4) for s in self.score_history[-MAX_SCORE_HISTORY:]],
            "log_lines": list(self.log_lines[-MAX_LOG_LINES:]),
        }

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def touch(self) -> None:
        self.updated_at = time.time()

    def push_score(self, score: float) -> None:
        """Append un best_score al historial (con cap para no crecer indefinidamente)."""
        self.score_history.append(float(score))
        if len(self.score_history) > MAX_SCORE_HISTORY * 2:
            self.score_history = self.score_history[-MAX_SCORE_HISTORY:]

    def log(self, msg: str, kind: str = "info") -> None:
        """Append una línea al log del job (con timestamp relativo al created_at)."""
        self.log_lines.append({
            "t": round(time.time() - self.created_at, 2),
            "msg": str(msg),
            "kind": kind,
        })
        if len(self.log_lines) > MAX_LOG_LINES * 2:
            self.log_lines = self.log_lines[-MAX_LOG_LINES:]


class JobRegistry:
    """Registro thread-safe de jobs en memoria con cleanup automático."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()

    def create(self, total: int = 0) -> JobState:
        with self._lock:
            self._cleanup_locked()
            job_id = uuid.uuid4().hex[:12]
            job = JobState(job_id=job_id, total=total)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.touch()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def _cleanup_locked(self) -> None:
        """Borra jobs expirados; si supera el límite total, borra los más viejos."""
        now = time.time()
        expired = [jid for jid, j in self._jobs.items() if now - j.updated_at > JOB_TTL_SECONDS]
        for jid in expired:
            self._jobs.pop(jid, None)
        if len(self._jobs) > MAX_JOBS_IN_MEMORY:
            # Borrar los más viejos hasta quedar en el límite
            sorted_jobs = sorted(self._jobs.items(), key=lambda kv: kv[1].created_at)
            excess = len(self._jobs) - MAX_JOBS_IN_MEMORY
            for jid, _ in sorted_jobs[:excess]:
                self._jobs.pop(jid, None)


# Singleton global del registry (uso local single-process)
REGISTRY = JobRegistry()
