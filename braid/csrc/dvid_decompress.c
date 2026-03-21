/*
 * DVID compressed segmentation block decompressor.
 *
 * Direct port of Go's MakeLabelVolume() from
 * dvid/datatype/common/labels/compressed.go (lines 1568-1636).
 *
 * Takes pre-zstd-decompressed DVID block data plus a pre-mapped label
 * array (supervoxel -> agglomerated mapping already applied by Python).
 * Writes decompressed uint64 volume to a pre-allocated output buffer
 * in ZYX order (Z changes slowest, X changes fastest — C row-major
 * with shape [gz*8][gy*8][gx*8]).
 *
 * Build: gcc -O2 -shared -fPIC -o libdvid_decompress.so dvid_decompress.c
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define SUB_BLOCK_SIZE 8
#define SUB_BLOCK_VOXELS (SUB_BLOCK_SIZE * SUB_BLOCK_SIZE * SUB_BLOCK_SIZE)

/* Left bit masks for each bit position in a byte (same as Go's leftBitMask). */
static const uint8_t LEFT_BIT_MASK[8] = {
    0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01
};

/*
 * Calculate minimum bits needed to represent n-1 values (0 to n-1).
 * Equivalent to Go's bitsFor().
 */
static uint32_t bits_for(uint16_t n) {
    uint32_t bits = 0;
    if (n < 2) return 0;
    n--;
    while (n > 0) {
        bits++;
        n >>= 1;
    }
    return bits;
}

/*
 * Extract a packed value of 'bits' bits starting at 'bit_head' bits
 * into the byte array.  Values cannot straddle more than 2 bytes.
 * Equivalent to Go's getPackedValue().
 */
static inline uint16_t get_packed_value(const uint8_t *b, uint32_t bit_head, uint32_t bits) {
    uint32_t byte_pos = bit_head >> 3;
    uint32_t bit_pos = bit_head & 7;
    uint16_t index;

    if (bit_pos + bits <= 8) {
        /* Index entirely within this byte. */
        uint32_t right_shift = 8 - bit_pos - bits;
        index = (uint16_t)((b[byte_pos] & LEFT_BIT_MASK[bit_pos]) >> right_shift);
    } else {
        /* Index spans two bytes. */
        index = (uint16_t)(b[byte_pos] & LEFT_BIT_MASK[bit_pos]) << 8;
        index |= (uint16_t)b[byte_pos + 1];
        index >>= (16 - bit_pos - bits);
    }
    return index;
}

/*
 * Decompress a DVID compressed segmentation block.
 *
 * Parameters:
 *   data           - DVID compressed block bytes (after zstd decompression)
 *   data_len       - length of data in bytes
 *   mapped_labels  - pre-mapped label array (supervoxel IDs already replaced
 *                    with agglomerated labels by the caller)
 *   num_labels     - number of entries in mapped_labels
 *   output         - pre-allocated output buffer, size gx*8 * gy*8 * gz*8 uint64s
 *   gx, gy, gz     - number of sub-blocks in each dimension
 *
 * Returns:
 *   0 on success, -1 on error (data too short, label count mismatch, etc.)
 *
 * The output is written in ZYX order: output[z * (gy*8 * gx*8) + y * (gx*8) + x]
 * which matches numpy's C-order array with shape (gz*8, gy*8, gx*8).
 */
int dvid_decompress_block(
    const uint8_t *data, size_t data_len,
    const uint64_t *mapped_labels, size_t num_labels,
    uint64_t *output,
    int gx, int gy, int gz)
{
    size_t pos;
    int num_sub_blocks;
    const uint16_t *num_sb_labels;
    const uint32_t *sb_indices;
    const uint8_t *sb_values;
    uint32_t index_pos, bit_pos;
    int sub_block_num;
    int sx, sy, sz;
    int size_x, size_y;

    /* Validate header: need at least 16 bytes for gx, gy, gz, N. */
    if (data_len < 16) return -1;

    /* Skip header — gx, gy, gz, N are passed in and validated by Python.
     * The header is: gx(u32) gy(u32) gz(u32) N(u32) in little-endian.
     * N should match num_labels. */
    uint32_t header_n;
    memcpy(&header_n, data + 12, 4);  /* little-endian uint32 */
    if ((size_t)header_n != num_labels) return -1;

    pos = 16;

    /* Skip the labels array in the data — we use mapped_labels instead. */
    pos += num_labels * 8;
    if (pos > data_len) return -1;

    size_x = gx * SUB_BLOCK_SIZE;
    size_y = gy * SUB_BLOCK_SIZE;
    num_sub_blocks = gx * gy * gz;

    /* Solid block: 0 or 1 labels — fill entire output. */
    if (num_labels < 2) {
        uint64_t label = (num_labels == 1) ? mapped_labels[0] : 0;
        size_t total = (size_t)size_x * size_y * gz * SUB_BLOCK_SIZE;
        for (size_t i = 0; i < total; i++) {
            output[i] = label;
        }
        return 0;
    }

    /* Parse NumSBLabels: uint16 per sub-block. */
    if (pos + (size_t)num_sub_blocks * 2 > data_len) return -1;
    num_sb_labels = (const uint16_t *)(data + pos);
    pos += (size_t)num_sub_blocks * 2;

    /* Calculate total SBIndices count. */
    uint32_t total_sb_indices = 0;
    for (int i = 0; i < num_sub_blocks; i++) {
        total_sb_indices += num_sb_labels[i];
    }

    /* Parse SBIndices: uint32 per index. */
    if (pos + (size_t)total_sb_indices * 4 > data_len) return -1;
    sb_indices = (const uint32_t *)(data + pos);
    pos += (size_t)total_sb_indices * 4;

    /* Remaining bytes are the bit-packed sub-block values. */
    sb_values = data + pos;

    /* Decompress each sub-block. */
    index_pos = 0;
    bit_pos = 0;
    sub_block_num = 0;

    for (sz = 0; sz < gz; sz++) {
        for (sy = 0; sy < gy; sy++) {
            for (sx = 0; sx < gx; sx++) {
                uint16_t n_labels = num_sb_labels[sub_block_num];
                uint32_t bits = bits_for(n_labels);

                /* Gather this sub-block's label palette from mapped_labels. */
                /* Using stack array — max sub-block labels observed is ~481. */
                uint64_t sb_labels[512];
                for (uint16_t i = 0; i < n_labels && i < 512; i++) {
                    uint32_t idx = sb_indices[index_pos];
                    if (idx < num_labels) {
                        sb_labels[i] = mapped_labels[idx];
                    }
                    index_pos++;
                }

                /* Calculate output position for this sub-block. */
                int base_z = sz * SUB_BLOCK_SIZE;
                int base_y = sy * SUB_BLOCK_SIZE;
                int base_x = sx * SUB_BLOCK_SIZE;

                /* Fill sub-block voxels. */
                for (int z = 0; z < SUB_BLOCK_SIZE; z++) {
                    for (int y = 0; y < SUB_BLOCK_SIZE; y++) {
                        int lbl_pos = (base_z + z) * size_y * size_x
                                    + (base_y + y) * size_x
                                    + base_x;
                        for (int x = 0; x < SUB_BLOCK_SIZE; x++) {
                            uint64_t label;
                            if (n_labels == 0) {
                                label = 0;
                            } else if (n_labels == 1) {
                                label = sb_labels[0];
                            } else {
                                uint16_t index = get_packed_value(sb_values, bit_pos, bits);
                                label = sb_labels[index];
                                bit_pos += bits;
                            }
                            output[lbl_pos + x] = label;
                        }
                    }
                }

                /* Pad to byte boundary. */
                if (bit_pos % 8 != 0) {
                    bit_pos += 8 - (bit_pos % 8);
                }

                sub_block_num++;
            }
        }
    }

    return 0;
}
