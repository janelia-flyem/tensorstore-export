/*
 * Neuroglancer compressed_segmentation encoder and DVID-to-cseg converter.
 *
 * Encoding algorithm ported from TensorStore (Apache 2.0):
 *   tensorstore/internal/compression/neuroglancer_compressed_segmentation.cc
 *
 * Produces single-channel uint64 compressed_segmentation output matching
 * TensorStore's EncodeChannels<uint64_t>() for num_channels=1.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o libbraid_codec.so dvid_decompress.c cseg_encode.c \
 *       -lzstd -lz
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>
#include <zlib.h>

/* Flags for dvid_to_cseg(). */
#define BRAID_INPUT_ZSTD   1
#define BRAID_OUTPUT_GZIP  2

/* External: DVID decompressor (from dvid_decompress.c). */
extern int dvid_decompress_block(
    const uint8_t *data, size_t data_len,
    const uint64_t *mapped_labels, size_t num_labels,
    uint64_t *output,
    int gx, int gy, int gz);


/* ---- Little-endian helpers ---- */

static inline uint32_t le_load32(const void *p) {
    const uint8_t *b = (const uint8_t *)p;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static inline void le_store32(void *p, uint32_t v) {
    uint8_t *b = (uint8_t *)p;
    b[0] = (uint8_t)(v);
    b[1] = (uint8_t)(v >> 8);
    b[2] = (uint8_t)(v >> 16);
    b[3] = (uint8_t)(v >> 24);
}

static inline void le_store64(void *p, uint64_t v) {
    le_store32(p, (uint32_t)v);
    le_store32((uint8_t *)p + 4, (uint32_t)(v >> 32));
}


/* ---- Table deduplication cache ---- */

#define MAX_DEDUP_LABELS  512   /* max labels per cache entry */
#define MAX_DEDUP_ENTRIES 1024

typedef struct {
    uint64_t labels[MAX_DEDUP_LABELS];
    uint32_t count;
    uint32_t offset;  /* table offset in uint32 units from channel start */
} dedup_entry_t;

typedef struct {
    dedup_entry_t *entries;
    int count;
    int cap;
} dedup_cache_t;

static void dedup_init(dedup_cache_t *c) {
    c->entries = (dedup_entry_t *)calloc(MAX_DEDUP_ENTRIES, sizeof(dedup_entry_t));
    c->count = 0;
    c->cap = MAX_DEDUP_ENTRIES;
}

static void dedup_free(dedup_cache_t *c) {
    free(c->entries);
    c->entries = NULL;
    c->count = 0;
}

/* Returns existing table offset or -1 if not found. */
static int dedup_lookup(const dedup_cache_t *c,
                        const uint64_t *sorted_labels, uint32_t n) {
    for (int i = 0; i < c->count; i++) {
        if (c->entries[i].count == n &&
            memcmp(c->entries[i].labels, sorted_labels,
                   (size_t)n * sizeof(uint64_t)) == 0) {
            return (int)c->entries[i].offset;
        }
    }
    return -1;
}

static void dedup_add(dedup_cache_t *c,
                      const uint64_t *sorted_labels, uint32_t n,
                      uint32_t offset) {
    if (c->count >= c->cap || n > MAX_DEDUP_LABELS) return;
    dedup_entry_t *e = &c->entries[c->count++];
    memcpy(e->labels, sorted_labels, (size_t)n * sizeof(uint64_t));
    e->count = n;
    e->offset = offset;
}


/* ---- Sorting / searching ---- */

static int cmp_uint64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

/* Binary search for val in sorted array. Returns index. */
static size_t bsearch_uint64(const uint64_t *arr, size_t n, uint64_t val) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


/* ---- Encoding bits ---- */

static uint32_t cseg_encoding_bits(size_t num_unique) {
    if (num_unique <= 1) return 0;
    uint32_t bits = 1;
    while ((size_t)(1u << bits) < num_unique) bits *= 2;
    return bits;
}


/*
 * cseg_max_encoded_size — conservative upper bound on compressed_segmentation
 * output size in bytes (before any gzip).
 */
size_t cseg_max_encoded_size(int nz, int ny, int nx,
                             int bz, int by, int bx)
{
    size_t gz = ((size_t)nz + bz - 1) / bz;
    size_t gy = ((size_t)ny + by - 1) / by;
    size_t gx = ((size_t)nx + bx - 1) / bx;
    size_t num_blocks = gz * gy * gx;
    size_t block_voxels = (size_t)bz * by * bx;

    /* Channel header + block headers */
    size_t header = 4 + num_blocks * 8;
    /* Per-block worst case: 16-bit encoding + full table (all voxels unique) */
    size_t max_enc_words = (16 * block_voxels + 31) / 32;
    size_t max_table_words = block_voxels * 2;  /* 2 uint32 per uint64 */
    size_t per_block = (max_enc_words + max_table_words) * 4;

    return header + num_blocks * per_block;
}


/*
 * cseg_encode_chunk — encode a uint64 ZYX volume as neuroglancer
 * compressed_segmentation (single channel).
 *
 * The volume must be C-contiguous in ZYX order: shape (nz, ny, nx),
 * Z slowest, X fastest.  This matches DVID's decompressor output.
 *
 * Returns 0 on success, -1 on error.
 */
int cseg_encode_chunk(
    const uint64_t *input,
    int nz, int ny, int nx,
    int bz, int by, int bx,
    uint8_t *output, size_t output_cap,
    size_t *output_size)
{
    if (!input || !output || !output_size) return -1;
    if (nz <= 0 || ny <= 0 || nx <= 0) return -1;
    if (bz <= 0 || by <= 0 || bx <= 0) return -1;

    int gz = (nz + bz - 1) / bz;
    int gy = (ny + by - 1) / by;
    int gx = (nx + bx - 1) / bx;
    int num_blocks = gz * gy * gx;

    size_t block_voxels = (size_t)bz * by * bx;

    /* Channel header (4 bytes) + block headers (num_blocks * 8 bytes) */
    size_t header_bytes = 4 + (size_t)num_blocks * 8;
    if (header_bytes > output_cap) return -1;

    /* Write channel header: channel 0 starts at uint32 offset 1 */
    le_store32(output, 1);

    /* base_offset: byte offset where channel data starts (after channel header) */
    size_t base_offset = 4;

    /* Current write position (after all headers) */
    size_t pos = header_bytes;

    /* Temp buffer for collecting block voxels (for unique-finding) */
    uint64_t *vox_buf = (uint64_t *)malloc(block_voxels * sizeof(uint64_t));
    if (!vox_buf) return -1;

    /* Table deduplication cache */
    dedup_cache_t cache;
    dedup_init(&cache);

    int result = 0;

    for (int ibz = 0; ibz < gz; ibz++) {
        for (int iby = 0; iby < gy; iby++) {
            for (int ibx = 0; ibx < gx; ibx++) {
                int block_idx = ibx + gx * (iby + gy * ibz);

                /* Block origin in voxels */
                int z0 = ibz * bz;
                int y0 = iby * by;
                int x0 = ibx * bx;

                /* Actual block shape (clipped to volume bounds) */
                int bsz = (z0 + bz <= nz) ? bz : nz - z0;
                int bsy = (y0 + by <= ny) ? by : ny - y0;
                int bsx = (x0 + bx <= nx) ? bx : nx - x0;

                /* Copy block voxels to temp buffer and find unique labels */
                size_t n_vox = 0;
                for (int z = 0; z < bsz; z++) {
                    for (int y = 0; y < bsy; y++) {
                        const uint64_t *row = input
                            + (size_t)(z0 + z) * ny * nx
                            + (size_t)(y0 + y) * nx
                            + x0;
                        for (int x = 0; x < bsx; x++) {
                            vox_buf[n_vox++] = row[x];
                        }
                    }
                }

                /* Sort and deduplicate to get unique labels */
                qsort(vox_buf, n_vox, sizeof(uint64_t), cmp_uint64);
                size_t num_unique = (n_vox > 0) ? 1 : 0;
                for (size_t i = 1; i < n_vox; i++) {
                    if (vox_buf[i] != vox_buf[num_unique - 1]) {
                        vox_buf[num_unique++] = vox_buf[i];
                    }
                }
                /* vox_buf[0..num_unique) is now the sorted unique label set */

                uint32_t enc_bits = cseg_encoding_bits(num_unique);
                size_t enc_words = (enc_bits * block_voxels + 31) / 32;
                size_t table_words = num_unique * 2;  /* 2 uint32 per uint64 */

                /* Check table dedup */
                int cached_offset = dedup_lookup(&cache, vox_buf,
                                                 (uint32_t)num_unique);
                int write_table = (cached_offset < 0);

                size_t data_words = enc_words + (write_table ? table_words : 0);
                size_t data_bytes = data_words * 4;

                if (pos + data_bytes > output_cap) {
                    result = -1;
                    goto done;
                }

                /* Encoded value offset (uint32 units from channel start) */
                uint32_t enc_val_off = (uint32_t)((pos - base_offset) / 4);

                /* Zero the encoded values area */
                memset(output + pos, 0, enc_words * 4);

                /* Bit-pack voxel indices */
                if (enc_bits > 0) {
                    uint32_t enc_mask = (1u << enc_bits) - 1;
                    (void)enc_mask;
                    for (int z = 0; z < bsz; z++) {
                        for (int y = 0; y < bsy; y++) {
                            const uint64_t *row = input
                                + (size_t)(z0 + z) * ny * nx
                                + (size_t)(y0 + y) * nx
                                + x0;
                            for (int x = 0; x < bsx; x++) {
                                uint64_t val = row[x];
                                uint32_t idx = (uint32_t)bsearch_uint64(
                                    vox_buf, num_unique, val);

                                /* Voxel offset within full block (x-fastest) */
                                size_t voff = (size_t)x
                                    + (size_t)bx * ((size_t)y + (size_t)by * (size_t)z);
                                size_t bit_off = voff * enc_bits;
                                size_t word_idx = bit_off / 32;
                                uint32_t bit_pos = (uint32_t)(bit_off % 32);

                                uint8_t *wp = output + pos + word_idx * 4;
                                uint32_t w = le_load32(wp);
                                w |= (idx << bit_pos);
                                le_store32(wp, w);
                            }
                        }
                    }
                }

                /* Table offset */
                uint32_t tbl_off;
                if (write_table) {
                    tbl_off = (uint32_t)((pos - base_offset) / 4 + enc_words);
                    /* Write label table */
                    uint8_t *tp = output + pos + enc_words * 4;
                    for (size_t i = 0; i < num_unique; i++) {
                        le_store64(tp + i * 8, vox_buf[i]);
                    }
                    dedup_add(&cache, vox_buf, (uint32_t)num_unique, tbl_off);
                } else {
                    tbl_off = (uint32_t)cached_offset;
                }

                /* Write block header */
                uint8_t *hdr = output + base_offset + (size_t)block_idx * 8;
                le_store32(hdr, tbl_off | (enc_bits << 24));
                le_store32(hdr + 4, enc_val_off);

                pos += data_bytes;
            }
        }
    }

    *output_size = pos;

done:
    free(vox_buf);
    dedup_free(&cache);
    return result;
}


/*
 * _apply_label_mapping — map DVID block labels from supervoxel IDs to
 * agglomerated label IDs using a key/value lookup table.
 *
 * For each of the N block labels, searches sv_keys for a match and
 * replaces it with the corresponding sv_vals entry.  Labels not found
 * in the mapping are kept as-is (identity).
 *
 * Both sv_keys and sv_vals must have num_map_entries elements.
 * The output mapped_labels array must be pre-allocated with at least
 * num_block_labels elements.
 */
static void _apply_label_mapping(
    const uint64_t *block_labels, size_t num_block_labels,
    const uint64_t *sv_keys, const uint64_t *sv_vals, size_t num_map_entries,
    uint64_t *mapped_labels)
{
    for (size_t i = 0; i < num_block_labels; i++) {
        uint64_t sv = block_labels[i];
        uint64_t mapped = sv;  /* default: identity */
        for (size_t j = 0; j < num_map_entries; j++) {
            if (sv_keys[j] == sv) {
                mapped = sv_vals[j];
                break;
            }
        }
        mapped_labels[i] = mapped;
    }
}


/*
 * _gzip_compress — gzip-compress src into dst using zlib.
 *
 * Returns 0 on success, -1 on error.
 * *dst_len is set to the actual compressed size.
 */
static int _gzip_compress(const uint8_t *src, size_t src_len,
                           uint8_t *dst, size_t dst_cap, size_t *dst_len)
{
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    /* windowBits = 15 + 16 for gzip format */
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return -1;

    strm.next_in = (Bytef *)src;
    strm.avail_in = (uInt)src_len;
    strm.next_out = dst;
    strm.avail_out = (uInt)dst_cap;

    int ret = deflate(&strm, Z_FINISH);
    *dst_len = strm.total_out;
    deflateEnd(&strm);
    return (ret == Z_STREAM_END) ? 0 : -1;
}


/*
 * dvid_to_cseg — full pipeline from (optionally zstd-compressed) DVID block
 * to (optionally gzip-compressed) neuroglancer compressed_segmentation.
 *
 * The DVID block header (gx, gy, gz, N, labels) is parsed internally.
 * The supervoxel→agglomerated label mapping is applied to the block's
 * internal label array before decompression.
 *
 * Parameters:
 *   dvid_data, dvid_len  — DVID block bytes (optionally zstd-compressed)
 *   sv_map_keys           — supervoxel IDs (mapping keys), or NULL for identity
 *   sv_map_vals           — agglomerated label IDs (mapping values), or NULL
 *   num_map_entries       — number of entries in the mapping (0 if NULL)
 *   bz, by, bx            — compressed_segmentation block size
 *   flags                 — BRAID_INPUT_ZSTD | BRAID_OUTPUT_GZIP
 *   output, output_cap    — pre-allocated output buffer
 *   output_size            — [out] actual bytes written
 *
 * Returns 0 on success, -1 on error.
 */
int dvid_to_cseg(
    const uint8_t *dvid_data, size_t dvid_len,
    const uint64_t *sv_map_keys, const uint64_t *sv_map_vals,
    size_t num_map_entries,
    int bz, int by, int bx,
    int flags,
    uint8_t *output, size_t output_cap,
    size_t *output_size)
{
    if (!dvid_data || !output || !output_size) return -1;

    const uint8_t *raw_dvid = dvid_data;
    size_t raw_dvid_len = dvid_len;
    uint8_t *zstd_buf = NULL;
    uint64_t *mapped_labels = NULL;
    uint64_t *volume = NULL;
    uint8_t *cseg_buf = NULL;
    int result = -1;

    /* Step 1: Optional zstd decompression */
    if (flags & BRAID_INPUT_ZSTD) {
        unsigned long long dec_size = ZSTD_getFrameContentSize(
            dvid_data, dvid_len);
        if (dec_size == ZSTD_CONTENTSIZE_UNKNOWN ||
            dec_size == ZSTD_CONTENTSIZE_ERROR)
            dec_size = dvid_len * 10;  /* fallback */

        zstd_buf = (uint8_t *)malloc((size_t)dec_size);
        if (!zstd_buf) goto cleanup;

        size_t zr = ZSTD_decompress(zstd_buf, (size_t)dec_size,
                                    dvid_data, dvid_len);
        if (ZSTD_isError(zr)) goto cleanup;
        raw_dvid = zstd_buf;
        raw_dvid_len = zr;
    }

    /* Step 2: Parse DVID header and apply label mapping */
    if (raw_dvid_len < 16) goto cleanup;

    uint32_t gx, gy, gz, num_labels;
    memcpy(&gx, raw_dvid + 0, 4);
    memcpy(&gy, raw_dvid + 4, 4);
    memcpy(&gz, raw_dvid + 8, 4);
    memcpy(&num_labels, raw_dvid + 12, 4);

    if (gx == 0 || gy == 0 || gz == 0) goto cleanup;
    if (16 + (size_t)num_labels * 8 > raw_dvid_len) goto cleanup;

    /* Read block's internal label array (supervoxel IDs) */
    const uint64_t *block_labels = (const uint64_t *)(raw_dvid + 16);

    /* Apply supervoxel → agglomerated mapping */
    mapped_labels = (uint64_t *)malloc((size_t)num_labels * sizeof(uint64_t));
    if (!mapped_labels) goto cleanup;

    if (sv_map_keys && sv_map_vals && num_map_entries > 0) {
        _apply_label_mapping(block_labels, num_labels,
                             sv_map_keys, sv_map_vals, num_map_entries,
                             mapped_labels);
    } else {
        /* Identity: copy block labels as-is */
        memcpy(mapped_labels, block_labels, (size_t)num_labels * sizeof(uint64_t));
    }

    /* Step 3: Decompress DVID block to uint64 volume */
    {
        int nx = (int)gx * 8;
        int ny = (int)gy * 8;
        int nz = (int)gz * 8;
        size_t vol_count = (size_t)nx * ny * nz;

        volume = (uint64_t *)malloc(vol_count * sizeof(uint64_t));
        if (!volume) goto cleanup;

        int ret = dvid_decompress_block(raw_dvid, raw_dvid_len,
                                        mapped_labels, num_labels,
                                        volume, (int)gx, (int)gy, (int)gz);
        if (ret != 0) goto cleanup;

        /* Step 4: Encode as compressed_segmentation */
        size_t cseg_size = 0;

        if (flags & BRAID_OUTPUT_GZIP) {
            size_t max_cseg = cseg_max_encoded_size(nz, ny, nx, bz, by, bx);
            cseg_buf = (uint8_t *)malloc(max_cseg);
            if (!cseg_buf) goto cleanup;

            ret = cseg_encode_chunk(volume, nz, ny, nx, bz, by, bx,
                                    cseg_buf, max_cseg, &cseg_size);
            if (ret != 0) goto cleanup;

            /* Step 5: Gzip compress */
            ret = _gzip_compress(cseg_buf, cseg_size,
                                 output, output_cap, output_size);
            if (ret != 0) goto cleanup;
        } else {
            ret = cseg_encode_chunk(volume, nz, ny, nx, bz, by, bx,
                                    output, output_cap, &cseg_size);
            if (ret != 0) goto cleanup;
            *output_size = cseg_size;
        }
    }

    result = 0;

cleanup:
    free(cseg_buf);
    free(volume);
    free(mapped_labels);
    free(zstd_buf);
    return result;
}
