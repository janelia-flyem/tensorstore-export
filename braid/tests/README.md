# BRAID Test Suite

This directory contains the comprehensive test suite for the BRAID library, which provides efficient access to sharded Arrow files containing DVID compressed blocks.

## Testing Strategy

The BRAID test suite follows a multi-layered testing approach that validates each component individually and then tests their integration:

### 1. Unit Testing
- **Component isolation**: Each major component (decompressor, reader, helper functions) is tested independently
- **Edge case validation**: Tests cover boundary conditions, error states, and invalid inputs
- **Algorithm verification**: Core algorithms like bit manipulation and compression are rigorously tested

### 2. Integration Testing
- **Pipeline validation**: Tests the complete flow from Arrow/CSV files through decompression
- **Data format compliance**: Ensures proper handling of the two-layer compression architecture
- **Real-world scenarios**: Uses actual DVID test data when available

### 3. Performance Testing
- **Compression efficiency**: Validates that the two-layer compression provides meaningful benefits
- **Memory usage**: Ensures decompression works with 64×64×64 blocks without excessive memory overhead
- **Error handling**: Tests graceful degradation under various failure conditions

## Test Coverage

### Core Components

#### `test_decompressor.py` - DVID Block Decompressor
**Purpose**: Tests the core DVID segmentation decompression functionality

**Coverage**:
- **Helper Functions**:
  - `bits_for()`: Validates bit calculation for packed values
  - `get_packed_value()`: Tests bit extraction from byte arrays
- **Solid Block Decompression**: Tests single-label blocks (most common case)
- **Multi-label Block Decompression**: Tests complex blocks with sub-block structure
- **Label Mapping**: Validates supervoxel → agglomerated label mapping
- **Error Handling**: Tests invalid data, malformed headers, size mismatches
- **Real Data Integration**: Uses actual DVID test files when available

**Key Test Cases**:
```python
def test_solid_block_decompression()     # Basic functionality
def test_label_mapping()                 # Supervoxel mapping
def test_invalid_data_handling()         # Error conditions
def test_real_dvid_data()                # Real-world data
```

#### `test_compression_layers.py` - Two-Layer Compression Architecture
**Purpose**: Tests the interaction between zstd (outer) and DVID (inner) compression layers

**Coverage**:
- **Layer Separation**: Validates that zstd and DVID compression work independently
- **Compression Efficiency**: Measures compression ratios with real data
- **Round-trip Validation**: Ensures data integrity through both compression layers
- **Error Propagation**: Tests error handling in each layer
- **Performance Metrics**: Validates compression provides meaningful benefits

**Key Test Cases**:
```python
def test_compression_layer_separation()   # Layer independence
def test_compression_efficiency()         # Performance metrics
def test_label_mapping_through_layers()   # End-to-end mapping
def test_error_handling_in_layers()       # Error isolation
```

#### `test_integration.py` - Full Pipeline Integration
**Purpose**: Tests the complete BRAID workflow from Arrow files to decompressed arrays

**Coverage**:
- **File Format Compliance**: Tests proper Arrow IPC and CSV index format
- **ShardReader Integration**: Validates the reader with real compressed data
- **Label Type Handling**: Tests both `LABELS` and `SUPERVOXELS` modes
- **Chunk Management**: Tests chunk discovery, indexing, and retrieval
- **Error Scenarios**: Tests missing chunks, invalid coordinates, file errors

**Key Test Cases**:
```python
def test_full_pipeline_labels()         # Agglomerated labels workflow
def test_full_pipeline_supervoxels()    # Supervoxel labels workflow
def test_chunk_not_found()              # Error handling
def test_chunk_info()                   # Metadata extraction
```

#### `test_reader.py` - ShardReader
**Purpose**: Tests the Arrow/CSV file reading and indexing functionality

**Coverage**:
- Arrow IPC file loading and validation
- CSV index parsing and coordinate mapping
- Schema validation and error handling
- Chunk existence checking and metadata extraction
- Label type selection (LABELS vs SUPERVOXELS)

**Key Test Cases**:
```python
def test_reader_initialization()          # File loading and schema check
def test_read_chunk_with_labels()         # Agglomerated label mapping
def test_read_chunk_with_supervoxels()    # Raw supervoxel output
def test_read_chunk_raw()                 # Raw data without decompression
```

#### `test_e2e_precomputed.py` - End-to-End Precomputed Roundtrip
**Purpose**: Tests the full pipeline from DVID shard to neuroglancer precomputed volume and back

**Coverage**:
- Creates synthetic DVID shard files with known label patterns
- Writes to neuroglancer precomputed volume via TensorStore
- Reads back and verifies voxel-level correctness
- Tests both supervoxel and agglomerated label modes
- Verifies multi-chunk volume integrity

**Key Test Cases**:
```python
def test_solid_block_supervoxels()            # Single-label block roundtrip
def test_two_label_block_labels()             # Agglomerated mapping roundtrip
def test_full_volume_integrity()              # All chunks correct after write
def test_agglomerated_labels_roundtrip()      # Label mapping preserved
```

#### `test_real_data.py` - Ground Truth Verification
**Purpose**: Verifies bit-identical output to Go's `MakeLabelVolume()` using real DVID test data

**Coverage**:
- Loads compressed blocks and raw volumes from `test_data/`
- Voxel-exact comparison against Go decompressor output
- Tests FIB-19 and CX datasets
- Validates block header parsing and label extraction

**Key Test Cases**:
```python
def test_fib19_sample1_voxel_exact()    # 262,144 voxels match Go output
def test_fib19_sample1_label_set()      # Correct unique labels extracted
def test_fib19_block_header()           # Block header fields parsed correctly
def test_corner_voxels()                # Spot-check boundary voxels
```

#### `test_bench_decompressor.py` - C Extension Benchmarks
**Purpose**: Measures C vs Python decompressor performance and verifies parity

**Coverage**:
- Single-chunk micro-benchmarks (C and Python)
- Full-shard macro-benchmark (258 chunks)
- Voxel-exact C vs Python comparison
- C extension loading verification

**Key Test Cases**:
```python
def test_c_extension_loaded()          # C library available
def test_single_chunk_c()              # C decompression timing
def test_single_chunk_python()         # Python reference timing
def test_c_vs_python_match()           # Voxel-exact parity
def test_shard_c()                     # Full shard throughput
def test_speedup_assertion()           # C at least 100x faster
```

#### `test_cseg_encode.py` - Compressed Segmentation Encoder
**Purpose**: Verifies BRAID's C encoder against TensorStore's reference implementation

**Coverage**:
- Byte-exact match against TensorStore C++ test vectors
- TensorStore decode roundtrip at every encoding bit width (0–16 bits)
- Fused DVID-to-cseg pipeline with real test data
- Label mapping (supervoxel → agglomerated) in C
- Gzip output compression
- Table deduplication

**Key Test Cases**:
```python
def test_solid_roundtrip()                 # 0-bit encoding
def test_many_labels_roundtrip()           # Multi-label TensorStore roundtrip
def test_all_chunks_in_real_shard()        # 258 chunks from mCNS shard
def test_fib19_fused_roundtrip()           # Full DVID→cseg pipeline
def test_label_mapping()                   # Supervoxel→agglomerated in C
```

#### `test_range_reader.py` - ShardRangeReader
**Purpose**: Verifies ShardRangeReader produces identical output to ShardReader

**Coverage**:
- Chunk-by-chunk parity with full-load ShardReader
- Byte-range read mechanics and batch caching
- New-format CSV index parsing
- Both label types (LABELS and SUPERVOXELS)

**Key Test Cases**:
```python
def test_all_chunks_labels()             # Full parity, agglomerated labels
def test_all_chunks_supervoxels()        # Full parity, supervoxel labels
def test_read_chunk_raw_parity()         # Raw data matches
def test_batch_size_gt1_cache_reuse()    # Batch cache reduces fetches
```

#### `test_go_produced_shard.py` - Go→Python Cross-Language Compatibility
**Purpose**: Tests reading Arrow IPC shards written by DVID's Go `export-shards` command

**Coverage**:
- Arrow schema compatibility between Go writer and Python reader
- CSV index coordinates match Arrow record fields
- All 258 chunks decompress to 64×64×64 uint64 arrays
- Supervoxel values are subsets of each chunk's supervoxel list
- Agglomerated label mapping produces valid values

**Key Test Cases**:
```python
def test_reader_opens()                      # Go-written Arrow file loads
def test_schema_fields()                     # Schema matches EXPECTED_SCHEMA
def test_labels_supervoxels_equal_length()   # Mapping arrays aligned
def test_supervoxels_match_label_list()      # Voxel values ⊆ supervoxel list
def test_label_mapping()                     # Agglomerated labels from mapping
```

### Test Data and Fixtures

#### `conftest.py` - Shared Test Infrastructure
**Purpose**: Provides reusable test fixtures and configuration

**Fixtures**:
- `zstd_compressor`: Pre-configured zstd compressor for tests
- `solid_dvid_block`: Factory for creating test DVID blocks
- `sample_arrow_data`: Sample Arrow table with compressed blocks
- `temp_shard_files`: Temporary Arrow/CSV files for integration tests

#### Real Data Integration
The test suite integrates with real DVID test data when available:
- Uses `fib19-64x64x64-sample1-block.dat.gz` for realistic compression testing
- Validates decompression produces expected multi-label segmentation
- Measures actual compression efficiency (typically 4-5x with zstd)

## Running Tests

The test suite has 112 tests across 10 test modules.

### Primary: pixi tasks (from tensorstore-export root)
```bash
pixi run test-braid    # unit + integration tests
pixi run test-bench    # C vs Python benchmarks
pixi run test-e2e      # end-to-end precomputed roundtrip
pixi run test-all      # everything (112 tests)
```

### Direct pytest (from braid/ directory)
```bash
pytest tests/                           # All tests
pytest tests/test_decompressor.py       # Specific module
pytest tests/ -v                        # Verbose output
pytest tests/ --cov=braid               # With coverage report
```

### Legacy runner
```bash
cd braid
python run_tests.py                     # unittest-based runner
python run_tests.py decompressor        # Specific module
```

## Test Data Requirements

### Required Test Files
- Real DVID test data is loaded from `tests/test_data/` (compressed blocks and raw volumes)
- Go-produced Arrow shard files in `tests/test_data/` for cross-language tests

### Generated Test Data
- Most tests use programmatically generated DVID blocks
- Test data includes both solid blocks and multi-label blocks
- Compression testing uses both minimal and realistic data sizes

## Expected Test Results

### Performance Benchmarks
When running with real data, expect:
- **Compression Ratio**: 4-5x compression with zstd over DVID blocks
- **Decompression Speed**: Sub-second for 64×64×64 blocks
- **Memory Usage**: ~2MB peak for single block decompression

### Test Coverage Metrics
The test suite provides comprehensive coverage:
- **Line Coverage**: >95% of decompressor and reader code
- **Branch Coverage**: >90% including error paths
- **Integration Coverage**: Complete pipeline from Arrow files to numpy arrays

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
pip install numpy pyarrow zstandard pytest  # Core dependencies
```

#### Missing Test Data
If real DVID data tests fail:
```python
# Tests automatically skip with message:
"Real DVID test data not available"
```

#### Import Errors
Ensure you're running from the braid directory:
```bash
cd braid
export PYTHONPATH=$PYTHONPATH:src
python run_tests.py
```

## Contributing

When adding new functionality to BRAID:

1. **Add unit tests** for new functions/methods
2. **Update integration tests** if changing the pipeline
3. **Add performance tests** if affecting compression/decompression
4. **Update this README** if changing the testing strategy

### Test Naming Conventions
- Test files: `test_<component>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<specific_functionality>`
- Fixtures: `<data_type>_<purpose>` (e.g., `solid_dvid_block`)

### Test Organization Principles
- **One test file per major component**
- **Group related tests in classes**
- **Use descriptive test method names**
- **Isolate external dependencies with fixtures**
- **Test both success and failure paths**