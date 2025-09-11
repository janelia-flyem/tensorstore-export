I have a question in optimizing the neuroglancer precomputed volume sharding specification for use with tensorstore (https://github.com/google/tensorstore) in creating large-scale neuroglancer precomputed volumes.  Read the following web pages documenting how tensorstore handles chunks (typically 64 x 64 x 64 voxels) by partitioning them into shards in morton space.

https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md

And in addition the above documents from the creators of tensorstore is a 3rd party analysis of how to best design a shard specification:
https://github.com/seung-lab/cloud-volume/wiki/Sharding:-Reducing-Load-on-the-Filesystem#designing-a-shardingspecification

Our "mcns" dataset has a very large 3D image volume (each voxel is `uint8`) that was segmented, producing a very large 3D segmentation volume (each voxel is `uint64`). Our Google colleagues wrote the initial image sharding specification, and then my colleague created a different sharding specification for our segmentation. I want to know what are the pros/cons of the two approaches. Basically my colleague reused the same `preshift_bits`, `minishard_bits`, and `shard_bits` across all scales while the Google-created spec decreases those bits as scale increases and the volume is further decreased in resolution by 2x each scale increase.

My question is if the reused bits across all scales has any pros/cons vs the Google strategy of decreasing bits allocated?  Are there inefficiencies in tensorstore code if far more bits are used than the down-res data needs?

Here is the original Google specification for the image:

{
    "@type": "neuroglancer_multiscale_volume",
    "data_type": "uint8",
    "num_channels": 1,
    "scales": [
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "8x8x8",
            "resolution": [
                8,
                8,
                8
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 19
            },
            "size": [
                94088,
                78317,
                134576
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "16x16x16",
            "resolution": [
                16,
                16,
                16
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 16
            },
            "size": [
                47044,
                39159,
                67288
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "32x32x32",
            "resolution": [
                32,
                32,
                32
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 13
            },
            "size": [
                23522,
                19580,
                33644
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "64x64x64",
            "resolution": [
                64,
                64,
                64
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 10
            },
            "size": [
                11761,
                9790,
                16822
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "128x128x128",
            "resolution": [
                128,
                128,
                128
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 7
            },
            "size": [
                5881,
                4895,
                8411
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "256x256x256",
            "resolution": [
                256,
                256,
                256
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 4
            },
            "size": [
                2941,
                2448,
                4206
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "512x512x512",
            "resolution": [
                512,
                512,
                512
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 1
            },
            "size": [
                1471,
                1224,
                2103
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "1024x1024x1024",
            "resolution": [
                1024,
                1024,
                1024
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 4,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 0
            },
            "size": [
                736,
                612,
                1052
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "2048x2048x2048",
            "resolution": [
                2048,
                2048,
                2048
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 1,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 9,
                "shard_bits": 0
            },
            "size": [
                368,
                306,
                526
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "4096x4096x4096",
            "resolution": [
                4096,
                4096,
                4096
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 0,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 7,
                "shard_bits": 0
            },
            "size": [
                184,
                153,
                263
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "encoding": "jpeg",
            "key": "8192x8192x8192",
            "resolution": [
                8192,
                8192,
                8192
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": 0,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 4,
                "shard_bits": 0
            },
            "size": [
                92,
                77,
                132
            ]
        }
    ],
    "type": "image"
}

Here is my colleague's modified sharding specification for our segmentation:

{
    "@type": "neuroglancer_multiscale_volume",
    "data_type": "uint64",
    "num_channels": 1,
    "scales": [
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "8_8_8",
            "resolution": [
                8.0,
                8.0,
                8.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                94088,
                77248,
                134592
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "16_16_16",
            "resolution": [
                16.0,
                16.0,
                16.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                47044,
                38624,
                67296
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "32_32_32",
            "resolution": [
                32.0,
                32.0,
                32.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                23522,
                19312,
                33648
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "64_64_64",
            "resolution": [
                64.0,
                64.0,
                64.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                11761,
                9656,
                16824
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "128_128_128",
            "resolution": [
                128.0,
                128.0,
                128.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                5880,
                4828,
                8412
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "256_256_256",
            "resolution": [
                256.0,
                256.0,
                256.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                2940,
                2414,
                4206
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "512_512_512",
            "resolution": [
                512.0,
                512.0,
                512.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                1470,
                1207,
                2103
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "1024_1024_1024",
            "resolution": [
                1024.0,
                1024.0,
                1024.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                735,
                603,
                1051
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "2048_2048_2048",
            "resolution": [
                2048.0,
                2048.0,
                2048.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                367,
                301,
                525
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "4096_4096_4096",
            "resolution": [
                4096.0,
                4096.0,
                4096.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                183,
                150,
                262
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        },
        {
            "chunk_sizes": [
                [
                    64,
                    64,
                    64
                ]
            ],
            "compressed_segmentation_block_size": [
                8,
                8,
                8
            ],
            "encoding": "compressed_segmentation",
            "key": "8192_8192_8192",
            "resolution": [
                8192.0,
                8192.0,
                8192.0
            ],
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "data_encoding": "gzip",
                "hash": "identity",
                "minishard_bits": 6,
                "minishard_index_encoding": "gzip",
                "preshift_bits": 6,
                "shard_bits": 21
            },
            "size": [
                91,
                75,
                131
            ],
            "voxel_offset": [
                0,
                0,
                0
            ]
        }
    ],
    "type": "segmentation"
}