import json
import os
import struct
import tempfile
import unittest


def create_synthetic_safetensors(path: str):
    # Create simple tensor
    tensor_data = bytes([0x12] * 512)
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [128, 1],
            "data_offsets": [0, 512],
        },
        "__metadata__": {"format": "pt"},
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(tensor_data)


class TestLlmZipPy(unittest.TestCase):
    def test_version_and_imports(self):
        try:
            import llmzip
        except ImportError:
            self.skipTest("llmzip module not compiled into Python site-packages")

        self.assertEqual(llmzip.version(), "0.2.0")

    def test_inspect_and_quantize_with_progress(self):
        try:
            import llmzip
        except ImportError:
            self.skipTest("llmzip module not compiled into Python site-packages")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "model.safetensors")
            output_file = os.path.join(tmpdir, "model-q4km.gguf")
            create_synthetic_safetensors(input_file)

            # Test inspect
            info = llmzip.inspect(input_file)
            self.assertIn("total_file_size", info)
            self.assertIn("predicted_sizes", info)
            self.assertGreaterEqual(info["tensor_count"], 1)

            # Test quantize_stream with callback
            events = []

            def callback(event: llmzip.ProgressEvent):
                events.append(event)

            res = llmzip.quantize_stream(
                input_file,
                output_file,
                preset="q4_k_m",
                progress_callback=callback,
            )
            self.assertIn("committed_tensors", res)
            self.assertTrue(os.path.exists(output_file))


if __name__ == "__main__":
    unittest.main()
