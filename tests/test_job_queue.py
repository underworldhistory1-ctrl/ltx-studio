"""Exercise the actual queue class without importing GPU inference dependencies."""
import ast
import asyncio
import copy
import itertools
import json
import logging
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'studio/jobs.py').read_text())
        queue_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'JobManager')
        counter = itertools.count()

        def write_json(path, value):
            path.write_text(json.dumps(value))

        namespace = dict(asyncio=asyncio, copy=copy, json=json, logging=logging, time=time, Path=Path,
                         Engine=object, JobCancelled=RuntimeError, DIRS={'jobs': self.folder},
                         atomic_write_json=write_json, read_json=lambda path: json.loads(path.read_text()),
                         new_id=lambda prefix: f'{prefix}_test{next(counter)}', safe_id=lambda value: value,
                         logger=logging.getLogger(__name__))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[queue_class], type_ignores=[])), 'jobs.py', 'exec'), namespace)
        self.engine = types.SimpleNamespace(check_h3_busy=AsyncMock(return_value=True))
        self.manager = namespace['JobManager'](self.engine)

    async def submit(self, name='Source.mp4'):
        return await asyncio.wait_for(self.manager.submit_job('/inputs/source.mp4', {'mode': 'refine'}, name=name), 0.5)

    async def test_delete_completed_job_then_accept_new_submission(self):
        job = await self.submit()
        job['status'] = 'completed'
        self.manager.queue.clear()
        await asyncio.wait_for(self.manager.delete_job(job['id']), 0.5)
        self.assertNotIn(job['id'], self.manager.jobs)
        self.assertFalse((self.folder / f"{job['id']}.json").exists())
        replacement = await self.submit('Next.mp4')
        self.assertIn(replacement['id'], self.manager.queue)
        self.assertFalse(self.manager._lock.locked())

    async def test_delete_queued_job_preserves_other_jobs_and_queue_positions(self):
        first = await self.submit('First.mp4')
        second = await self.submit('Second.mp4')
        third = await self.submit('Third.mp4')
        await asyncio.wait_for(self.manager.delete_job(second['id']), 0.5)
        self.assertEqual(self.manager.queue, [first['id'], third['id']])
        self.assertEqual(third['queue_position'], 2)
        summary = await asyncio.wait_for(self.manager.get_queue_summary(), 0.5)
        self.assertEqual(len(summary['queued_jobs']), 2)
        self.assertNotIn(second['id'], self.manager.jobs)

    async def test_concurrent_delete_and_submit_both_finish(self):
        old = await self.submit()
        await asyncio.wait_for(asyncio.gather(self.manager.delete_job(old['id']), self.submit('New.mp4')), 0.5)
        self.assertEqual(len(self.manager.jobs), 1)
        self.assertEqual(len(self.manager.queue), 1)

    async def test_deleting_missing_job_does_not_lock_future_submission(self):
        self.assertTrue(await asyncio.wait_for(self.manager.delete_job('ltx_missing'), 0.5))
        await self.submit()


if __name__ == '__main__':
    unittest.main()
