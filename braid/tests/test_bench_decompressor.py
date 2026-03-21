"""
Benchmarks comparing C extension vs Python reference decompressor.

Run with: pixi run test-bench
"""

import gzip
import time
from pathlib import Path

import numpy as np
import pytest

from braid.decompressor import DVIDDecompressor

TEST_DATA = Path(__file__).parent / "test_data"


@pytest.fixture
def decompressor():
    return DVIDDecompressor()


@pytest.fixture
def real_shard_data():
    """Load the real Go-produced shard for macro-benchmark."""
    from braid import ShardReader
    arrow = TEST_DATA / "30720_24576_28672.arrow"
    csv = TEST_DATA / "30720_24576_28672.csv"
    if not arrow.exists():
        pytest.skip("Real shard test data not available")
    return ShardReader(str(arrow), str(csv))


@pytest.fixture
def fib19_block():
    """Load the real DVID compressed block for micro-benchmark."""
    block_file = TEST_DATA / "fib19-64x64x64-sample1-block.dat.gz"
    if not block_file.exists():
        pytest.skip("fib19 block test data not available")
    import zstandard as zstd
    raw_dvid = gzip.decompress(block_file.read_bytes())
    # Wrap in zstd for the decompress_block API
    cctx = zstd.ZstdCompressor()
    return cctx.compress(raw_dvid)


class TestBenchmarkMicro:
    """Micro-benchmark: single chunk decompression."""

    def test_c_extension_loaded(self, decompressor):
        """Verify the C extension is available."""
        assert decompressor._c_lib is not None, \
            "C extension not loaded — run 'pixi run build-braid-c' first"

    def test_single_chunk_c(self, decompressor, fib19_block):
        """Benchmark C extension on a single real DVID block."""
        # Warmup
        decompressor.decompress_block(fib19_block)

        n_iterations = 1000
        start = time.perf_counter()
        for _ in range(n_iterations):
            decompressor.decompress_block(fib19_block)
        elapsed = time.perf_counter() - start

        ms_per_chunk = elapsed / n_iterations * 1000
        chunks_per_sec = n_iterations / elapsed
        print(f"\n  C extension: {ms_per_chunk:.3f} ms/chunk ({chunks_per_sec:.0f} chunks/sec)")

    def test_single_chunk_python(self, decompressor, fib19_block):
        """Benchmark Python reference on a single real DVID block."""
        # Force Python path by temporarily disabling C lib
        saved = decompressor._c_lib
        decompressor._c_lib = None

        # Warmup
        decompressor.decompress_block(fib19_block)

        n_iterations = 5  # Python is slow, don't do many
        start = time.perf_counter()
        for _ in range(n_iterations):
            decompressor.decompress_block(fib19_block)
        elapsed = time.perf_counter() - start

        decompressor._c_lib = saved

        ms_per_chunk = elapsed / n_iterations * 1000
        chunks_per_sec = n_iterations / elapsed
        print(f"\n  Python ref:  {ms_per_chunk:.1f} ms/chunk ({chunks_per_sec:.1f} chunks/sec)")

    def test_c_vs_python_match(self, decompressor, fib19_block):
        """Verify C and Python produce identical output."""
        result_c = decompressor.decompress_block(fib19_block)

        saved = decompressor._c_lib
        decompressor._c_lib = None
        result_py = decompressor.decompress_block(fib19_block)
        decompressor._c_lib = saved

        np.testing.assert_array_equal(result_c, result_py,
                                       err_msg="C and Python outputs differ")


class TestBenchmarkShard:
    """Macro-benchmark: full shard decompression."""

    def test_shard_c(self, decompressor, real_shard_data):
        """Benchmark C extension on full 258-chunk shard."""
        reader = real_shard_data
        from braid import LabelType

        start = time.perf_counter()
        for cx, cy, cz in reader.available_chunks:
            reader.read_chunk(cx, cy, cz, label_type=LabelType.LABELS)
        elapsed = time.perf_counter() - start

        chunks_per_sec = reader.chunk_count / elapsed
        print(f"\n  C extension shard: {elapsed:.2f}s for {reader.chunk_count} chunks "
              f"({chunks_per_sec:.0f} chunks/sec)")

    def test_speedup_assertion(self, decompressor, fib19_block):
        """Assert C extension is at least 100x faster than Python reference."""
        # Time C
        n_c = 500
        start = time.perf_counter()
        for _ in range(n_c):
            decompressor.decompress_block(fib19_block)
        c_elapsed = time.perf_counter() - start
        c_per_chunk = c_elapsed / n_c

        # Time Python
        saved = decompressor._c_lib
        decompressor._c_lib = None
        n_py = 3
        start = time.perf_counter()
        for _ in range(n_py):
            decompressor.decompress_block(fib19_block)
        py_elapsed = time.perf_counter() - start
        py_per_chunk = py_elapsed / n_py
        decompressor._c_lib = saved

        speedup = py_per_chunk / c_per_chunk
        print(f"\n  Speedup: {speedup:.0f}x (C: {c_per_chunk*1000:.3f}ms, Python: {py_per_chunk*1000:.1f}ms)")
        assert speedup >= 100, f"Expected >=100x speedup, got {speedup:.0f}x"
