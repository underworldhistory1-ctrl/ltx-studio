"""Job Queue, persistent execution loop, and History manager."""
import asyncio, copy, json, logging, os, time
from pathlib import Path
import aiohttp

from common import DIRS, atomic_write_json, read_json, safe_id, new_id
from engine import Engine, JobCancelled

logger = logging.getLogger("ltx.jobs")


class JobManager:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.queue = []
        self.active_job = None
        self.jobs = {}
        self.listeners = []
        self._lock = asyncio.Lock()
        self._worker_task = None
        self._load_history()

    def _load_history(self):
        """Loads persistent job documents from disk on startup."""
        for f in DIRS["jobs"].glob("ltx_*.json"):
            data = read_json(f)
            if data and "id" in data:
                # Mark interrupted jobs as failed on reload
                if data.get("status") in ("running", "queued"):
                    data["status"] = "failed"
                    data["error"] = "Interrupted by server restart"
                    atomic_write_json(f, data)
                self.jobs[data["id"]] = data

    def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    def subscribe(self, callback):
        self.listeners.append(callback)

    def unsubscribe(self, callback):
        if callback in self.listeners:
            self.listeners.remove(callback)

    def _notify(self, job):
        atomic_write_json(DIRS["jobs"] / f"{job['id']}.json", job)
        for cb in list(self.listeners):
            try:
                cb(job)
            except Exception:
                pass

    async def _check_h3_busy(self) -> bool:
        """Checks whether the shared GPU is currently executing, decoding, or encoding a MiniMax H3 render."""
        return await self.engine.check_h3_busy()

    async def submit_job(self, input_video_path: str, settings: dict, name: str = None) -> dict:
        job_id = new_id("ltx")
        
        async with self._lock:
            h3_busy = await self._check_h3_busy()
            is_busy = bool(self.active_job or self.queue or h3_busy)
            pos = len(self.queue) + (1 if self.active_job or h3_busy else 0)

            job = {
                "id": job_id,
                "name": name or Path(input_video_path).name,
                "input_video_path": str(input_video_path),
                "input_video_url": f"/inputs/{Path(input_video_path).name}",
                "settings": settings,
                "created_at": time.time(),
                "status": "queued",
                "queue_position": pos if is_busy else 0,
                "stage": f"Waiting in queue (Position #{pos})..." if pos > 1 else ("Waiting in queue: GPU is busy with H3 render. Will start automatically..." if h3_busy else "Queued..."),
                "progress": 0.0,
                "eta_text": "Queued",
                "error": None
            }
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self._notify(job)
        return job

    async def get_job(self, job_id: str) -> dict:
        return self.jobs.get(job_id)

    async def list_jobs(self, limit: int = 100) -> list:
        # Return sorted by created_at descending
        res = list(self.jobs.values())
        res.sort(key=lambda j: j.get("created_at", 0), reverse=True)
        return res[:limit]

    async def get_queue_summary(self) -> dict:
        async with self._lock:
            h3_busy = await self._check_h3_busy()
            q_list = []
            for i, qid in enumerate(self.queue):
                j = self.jobs.get(qid)
                if j:
                    q_list.append({
                        "id": j["id"],
                        "name": j.get("name"),
                        "position": i + 1,
                        "created_at": j.get("created_at"),
                        "stage": j.get("stage")
                    })
            return {
                "active_job": self.active_job["id"] if self.active_job else None,
                "queued_count": len(self.queue),
                "queued_jobs": q_list,
                "h3_busy": h3_busy
            }

    async def cancel_job(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self.queue:
                self.queue.remove(job_id)
                # Re-calculate positions for remaining jobs
                for i, qid in enumerate(self.queue):
                    qj = self.jobs.get(qid)
                    if qj:
                        qj["queue_position"] = i + 1
                        qj["stage"] = f"Waiting in queue (Position #{i + 1})..."
                        self._notify(qj)
            job = self.jobs.get(job_id)
            if job:
                job["status"] = "cancelled"
                job["stage"] = "Cancelled by user"
                job["eta_text"] = "Cancelled"
                self._notify(job)
                return True
        return False

    async def delete_job(self, job_id: str) -> bool:
        # cancel_job acquires the same non-reentrant queue lock itself.
        await self.cancel_job(job_id)
        async with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
            f = DIRS["jobs"] / f"{job_id}.json"
            if f.exists():
                f.unlink(missing_ok=True)
            return True

    async def restart_job(self, job_id: str) -> dict:
        old = await self.get_job(job_id)
        if not old:
            raise KeyError("Job not found")
        return await self.submit_job(old["input_video_path"], copy.deepcopy(old["settings"]), name=old.get("name"))

    async def _worker_loop(self):
        while True:
            job_to_run = None
            async with self._lock:
                if self.queue:
                    # Look at next job in queue without popping yet
                    next_id = self.queue[0]
                    job = self.jobs.get(next_id)
                    if not job or job.get("status") == "cancelled":
                        self.queue.pop(0)
                        continue

                    # Release only a verified idle H3 cache before the admission guard.
                    # An active queue is never asked to unload.
                    await self.engine.free_h3_cache()
                    # Check if shared GPU is currently busy with H3 render
                    h3_busy = await self._check_h3_busy()
                    if h3_busy:
                        job["status"] = "queued"
                        job["stage"] = "Waiting in queue: GPU is currently busy with H3 render. Will start automatically..."
                        job["eta_text"] = "Waiting for H3..."
                        self._notify(job)
                        await asyncio.sleep(3.0)
                        continue

                    # GPU is free - take this job from queue
                    self.queue.pop(0)
                    job_to_run = job
                    job_to_run["queue_position"] = 0
                    
                    # Update queue positions for remaining queued jobs
                    for i, qid in enumerate(self.queue):
                        qj = self.jobs.get(qid)
                        if qj and qj.get("status") == "queued":
                            qj["queue_position"] = i + 1
                            qj["stage"] = f"Waiting in queue (Position #{i + 1})..."
                            self._notify(qj)

            if job_to_run:
                self.active_job = job_to_run
                try:
                    await self.engine.execute(job_to_run, self._notify)
                except JobCancelled:
                    job_to_run["status"] = "cancelled"
                    job_to_run["stage"] = "Cancelled"
                    self._notify(job_to_run)
                except Exception as e:
                    logger.exception(f"Job {job_to_run['id']} execution error: {e}")
                    job_to_run["status"] = "failed"
                    job_to_run["error"] = str(e)
                    job_to_run["stage"] = f"Failed: {str(e)[:200]}"
                    self._notify(job_to_run)
                finally:
                    self.active_job = None

            await asyncio.sleep(1.0)
