"""Regression tests against legacy transform; no training or user data writes."""
import runpy
import sys
import unittest
import uuid
import json
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Code.encry import photo_encry_checked as checked


class CheckedTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = runpy.run_path(str(ROOT/'Code/encry/photo_encry'),run_name='test_legacy')

    def test_legacy_cipher_compatibility(self):
        rng = np.random.default_rng(123)
        for h,w in [(8,8),(16,16),(17,25),(32,32),(64,64)]:
            for _ in range(4):
                image = rng.integers(0,256,(h,w,3),dtype=np.uint8)
                rules = tuple(rng.integers(1,9,checked.block_count(image.shape),dtype=np.uint8) for _ in range(4))
                cipher,meta = checked.encrypt_image(image,rules,'test-only')
                old_cipher,_ = self.old['encrypt'](rules,image,'test-only')
                np.testing.assert_array_equal(cipher,old_cipher)
                np.testing.assert_array_equal(image,checked.decrypt_image(cipher,rules,meta,'test-only'))
                self.assertNotIn('initial_value',meta)

    def test_block_size_is_consistent(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0,256,(17,25,3),dtype=np.uint8)
        for p in [1,4,8,16,32]:
            rules = tuple(rng.integers(1,9,checked.block_count(image.shape,p),dtype=np.uint8) for _ in range(4))
            c,m = checked.encrypt_image(image,rules,'test-only',p)
            np.testing.assert_array_equal(image,checked.decrypt_image(c,rules,m,'test-only'))

    def test_quantizer_compatibility(self):
        rng = np.random.default_rng(3)
        raw = tuple(rng.normal(0,3,64) for _ in range(4))
        actual = checked.legacy_quantize(raw)
        for a,b in zip(raw,actual):
            expected = (np.round(a)%8+1).astype(np.uint8)
            for _ in range(15):
                expected = (self.old['process_array_with_kalman'](expected)%8+1).astype(np.uint8)
            np.testing.assert_array_equal(expected,b)

    def test_default_quantizer_does_not_filter(self):
        a = np.array([-1.5,0.49,0.51,2.5,3.5,8.0])
        streams = [a]*4
        expected = (np.round(a)%8+1).astype(np.uint8)
        for q in checked.quantize_streams(streams):
            np.testing.assert_array_equal(q,expected)
        for passes in [1,3,5,15]:
            q = expected.copy()
            for _ in range(passes):
                q = (self.old['process_array_with_kalman'](q)%8+1).astype(np.uint8)
            np.testing.assert_array_equal(q,checked.quantize_streams(streams,passes)[0])
        for invalid in [-1,16,1.5]:
            with self.assertRaises(ValueError):
                checked.quantize_streams(streams,invalid)

    def test_reject_bad_streams(self):
        for streams in [(),(np.ones(1,dtype=int),),tuple(np.zeros(1,dtype=int) for _ in range(4)),tuple(np.ones(1) for _ in range(4))]:
            with self.assertRaises(ValueError):
                checked.validate_rules(streams,1)
        with self.assertRaises(ValueError):
            checked.legacy_quantize([np.array([np.nan])]*4)

    def test_wrong_password_is_not_success(self):
        image = np.arange(192,dtype=np.uint8).reshape(8,8,3)
        rules = tuple(np.ones(1,dtype=np.uint8) for _ in range(4))
        c,m = checked.encrypt_image(image,rules,'test-only')
        candidate = checked.decrypt_image(c,rules,m,'wrong-test-only')
        self.assertIsInstance(candidate,np.ndarray)
        self.assertFalse(np.array_equal(image,candidate))

    def test_degeneracy_is_reported(self):
        rules = tuple(np.ones(2,dtype=np.uint8) for _ in range(4))
        d = checked.rule_diagnostics(rules,rules,2)
        self.assertEqual(d['master_identity_block_fraction'],1.0)
        self.assertEqual(d['master_unique_per_stream'],[1]*4)

    def test_cli_reports_failure_and_preserves_outputs(self):
        scratch = ROOT/'tmp'
        scratch.mkdir(exist_ok=True)
        directory = scratch/('encryption_test_'+uuid.uuid4().hex)
        directory.mkdir()
        # Keep isolated test artifacts for inspection; avoid Windows temp ACLs.
        with contextlib.nullcontext():
            folder = Path(directory)
            image = np.arange(192,dtype=np.uint8).reshape(8,8,3)
            checked.cv2.imwrite(str(folder/'input.png'),image)
            # Mock checkpoint loading; this tests orchestration, not a policy.
            model = folder/'node2.zip'
            model.touch()
            raw = tuple(np.zeros(1) for _ in range(4))
            argv = ['checked','--image',str(folder/'input.png'),'--output',str(folder/'run'),'--node','2','--model',str(model),'--burn-in','0']
            with patch.object(sys,'argv',argv),patch.object(checked.getpass,'getpass',return_value='test-only'),patch.object(checked,'generate_streams',return_value=(raw,raw)),patch.object(checked,'decrypt_image',return_value=np.zeros_like(image)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checked.main(),1)
                report = json.loads((folder/'run/report.json').read_text())
                self.assertEqual(report['status'],'recovery_failed')
                self.assertFalse(report['exact_recovery'])
                self.assertEqual(report['attempts'],1)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    checked.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)
