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

#### `test_reader.py` - ShardReader (Existing)
**Purpose**: Tests the Arrow/CSV file reading and indexing functionality

**Coverage**:
- Arrow IPC file loading and validation
- CSV index parsing and coordinate mapping
- Schema validation and error handling
- Chunk existence checking and metadata extraction

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

### Run All Tests
```bash
cd braid
python run_tests.py
```

### Run Specific Test Modules
```bash
python run_tests.py decompressor        # Core decompression tests
python run_tests.py compression_layers  # Two-layer compression tests
python run_tests.py integration         # Full pipeline tests
python run_tests.py reader              # ShardReader tests
```

### Using pytest (if available)
```bash
cd braid
pytest tests/                           # Run all tests
pytest tests/test_decompressor.py       # Run specific test file
pytest tests/ -v                        # Verbose output
pytest tests/ --cov=braid               # With coverage report
```

### Using unittest directly
```bash
cd braid
python -m unittest tests.test_decompressor        # Specific module
python -m unittest discover tests/                # All tests
```

## Test Data Requirements

### Required Test Files
- Real DVID test data is loaded from `../research/fib19-64x64x64-sample1-block.dat.gz`
- If this file is not available, real data tests are skipped automatically

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