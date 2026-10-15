"""True CPU quadruped/production controls; RGB is an explicit stub."""
import copy
import io
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'disable')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_live_shadow as s


class LiveShadowTest(unittest.TestCase):
  def test_import_is_lazy_and_framing_rejects_truncation_and_oversize(self):
    result = subprocess.run([sys.executable, '-c',
        "import os,sys; b=dict(os.environ); import scripts.dl_live_shadow; "
        "assert dict(os.environ)==b; assert 'jax' not in sys.modules; "
        "assert not any(x.startswith('dm_control') for x in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stderr)
    stream = io.BytesIO()
    s.write_frame(stream, {'actual': np.float64(.1)})
    stream.seek(0)
    self.assertEqual(s.read_frame(stream)['actual'].tobytes(), np.float64(.1).tobytes())
    self.assertIsNone(s.read_frame(stream))
    for damaged in (b'x', struct.pack('>Q', s.MAX_FRAME+1), struct.pack('>Q', 3)+b'x'):
      with self.assertRaises(ValueError): s.read_frame(io.BytesIO(damaged))

  def test_actual_cpu_shadow_preserves_live_and_matches_qq_first_f64(self):
    from dm_control.mujoco.engine import Physics
    from scripts import dl_diagnostic as d
    from scripts.dl_live_policy import unit_plan
    from scripts.dl_feedback_physics import FeedbackEnvironment
    def stub_rgb(self, height=64, width=64, camera_id=2, **kwargs):
      return np.full((height, width, 3), int(self.data.time*1000) % 256, np.uint8)
    with mock.patch.object(Physics, 'render', stub_rgb):
      live = FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk', 8192))
      try:
        live.reset()
        for _ in range(unit_plan(0)['t']): live.step(np.zeros(12, np.float32))
        before = live.snapshot()
        raw = np.linspace(-1.4, 1.4, 12, dtype=np.float32)
        request = s.make_request(0, before, s.static_model_identity(live.suite.physics.model),
            raw, bundle_sha256='a'*64, request_id='unit0')
        original = pickle.dumps(request, protocol=5)
        # Production wrappers import JAX. A shadow in that same process is rejected.
        with self.assertRaisesRegex(ValueError, 'JAX/model process'): s.shadow_step(request)
        packet = io.BytesIO(); s.write_frame(packet, request)
        result = subprocess.run([sys.executable, str(ROOT/'scripts/dl_live_shadow.py'), '--serve'],
            input=packet.getvalue()*2, cwd=ROOT, capture_output=True, timeout=60,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable'))
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        responses = io.BytesIO(result.stdout)
        receipt, replay = s.read_frame(responses), s.read_frame(responses)
        self.assertIsNone(s.read_frame(responses))
        self.assertEqual(pickle.dumps(request, protocol=5), original)
        self.assertEqual(s.snapshot_sha256(live.snapshot()), s.snapshot_sha256(before))
        self.assertEqual(receipt['response_sha256'], s.response_sha256(receipt))
        next_obs, applied = live.step(raw)
        ts = live.recorder.last_timestep
        self.assertEqual(applied.tobytes(), receipt['applied'].tobytes())
        self.assertEqual(np.asarray(ts.reward).tobytes(), receipt['reward_f64'].tobytes())
        self.assertEqual(next_obs['reward'].tobytes(), receipt['reward_f32'].tobytes())
        self.assertEqual(s.snapshot_sha256(live.snapshot()), receipt['next_snapshot_sha256'])
        # Two requests in one genuine exec process each build a fresh official suite.
        self.assertEqual(replay['response_sha256'], receipt['response_sha256'])
        for kind in ('source', 'model', 'raw', 'request'):
          broken = copy.deepcopy(request)
          if kind == 'source': broken['actual_full_snapshot']['suite']['integration'][0] += 1
          elif kind == 'model': broken['static_mjb']['mjb_sha256'] = '0'*64
          elif kind == 'raw': broken['raw_q'] = raw.astype(np.float64)
          else: broken['request_sha256'] = '0'*64
          if kind == 'request':
            self.assertNotEqual(broken['request_sha256'], s.request_sha256(broken))
          elif kind != 'model':
            with self.assertRaises(ValueError): s.request_sha256(broken)
          else:
            broken['request_sha256'] = s.request_sha256(broken)
            failed_packet = io.BytesIO(); s.write_frame(failed_packet, broken)
            failed = subprocess.run([sys.executable,str(ROOT/'scripts/dl_live_shadow.py'),'--serve'],
                input=failed_packet.getvalue(),cwd=ROOT,capture_output=True,timeout=60,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable'))
            self.assertNotEqual(failed.returncode,0)
            self.assertIn(b'MJB identity differs', failed.stderr)
        changed = copy.deepcopy(receipt)
        changed['reward_f64'][...] = np.nextafter(changed['reward_f64'], np.inf)
        self.assertNotEqual(s.response_sha256(changed), receipt['response_sha256'])
      finally:
        live.close()


if __name__ == '__main__': unittest.main()
