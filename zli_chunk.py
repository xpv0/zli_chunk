#!/usr/bin/env python3    
# -*- coding: utf-8 -*-    
    
"""zli_chunk.py    
    
GitHub ref: d177e9f32bd5ac037d54a2af5f7588b841aeb47d    
of Zli/OpenZL    

compress dataset using https://github.com/facebook/openzl    
zli is alpha software, first released on October 1st 2025    
the following issues with zli are bypassed:    
  * zli supposedly only works on files up to 2GB (not 2GiB),    
    see:    
    https://github.com/facebook/openzl/blob/d177e9f32bd5ac037d54a2af5f7588b841aeb47d/cli/commands/cmd_compress.cpp#L150    
    actually, 1GiB files work    
  * zli offers profiles which improve compression, such as le-i64,    
    which may fail, in which case we fallback on le-i32    

Uses a pool of processes call several times Zli from OpenZL at once.    

TODO:    
  * use argparse to select input file    
  * allow more failure modes:    
    - keep chunk even if compression failed    
    - exit on failure    
  * implement decompression and merging with fallocate    
    
Tested on one dataset of ~100 GB at around ~400 MB/s throuput with    
a compression factor close to 7.    
"""

import os    
import sys    
import time    
import argparse as ap    
import contextlib as cl    
import subprocess as sp    
import collections as clt    
import multiprocessing as mp    


FILE_EXTENSION = "npy"    
ZLI_EXTENSION = "zli"
ZLI_ALGO_I8 = "le-i64"
ZLI_ALGO_I4 = "le-i32"


class BaseZliChunkException(Exception):
    """BaseZliChunkException"""


class ProcessCountError(BaseZliChunkException, ValueError):
    """ProcessCountError"""


class ZliCompressFailure(BaseZliChunkException, RuntimeError):
    """ZliCompressFailure"""


log = print  # logger
Instruction = clt.namedtuple("Instruction", "idx offset chunk fname zli_bin")


@cl.contextmanager
def timeit(label: str = ""):
    """timeit"""
    elapsed = -time.time()
    try:
        yield
    finally:
        elapsed += time.time()
        log(f"{label} took {elapsed = } s")


def _zli_compress(fname: int | str, zli_bin: str, algo: str) -> None:
    """_zli_compress"""
    comp = 8 if algo == ZLI_ALGO_I8 else 4
    _output = sp.check_output(
        [
            zli_bin,
            "compress",
            "--profile",
            algo,
            f"{fname}.{FILE_EXTENSION}",
            "--output",
            f"{fname}.{FILE_EXTENSION}.{comp}.{ZLI_EXTENSION}",
        ]
    )


def _zli_compress_retry(fname: int | str, zli_bin: str) -> None:
    """_zli_compress_retry"""
    try:
        _zli_compress(fname, zli_bin, ZLI_ALGO_I8)
        return
    except sp.CalledProcessError as exc:
        log(f"Failed {fname} using {ZLI_ALGO_I8}, {exc = }")

    try:
        _zli_compress(fname, zli_bin, ZLI_ALGO_I4)
        return
    except sp.CalledProcessError as exc:
        log(f"Failed {fname} using {ZLI_ALGO_I4}, {exc = }")
        raise ZliCompressFailure(f"Could not compress {fname}") from exc

def zli_compress_routine(instruction: Instruction) -> None:
    """zli_compress_routine"""
    with open(instruction.fname, mode="rb") as frb:
        frb.seek(instruction.offset)
        data = frb.read(instruction.chunk)
    with open(f"{instruction.idx}.{FILE_EXTENSION}", mode="wb") as fwb:
        fwb.write(data)
    _zli_compress_retry(instruction.idx, instruction.zli_bin)
    os.unlink(f"{instruction.idx}.{FILE_EXTENSION}")


def main():
    """main"""
    parser = ap.ArgumentParser(
        prog="Zli chunk",
        description="Compress numpy-like huge files using newly released OpenZL / Zli",
        epilog="This is **alpha software**, since OpenZL is too.",
    )
    parser.add_argument("-f", "--fname", help="name of the file to compress")
    parser.add_argument(
        "-b", "--bin", type=str, default="zli", help="location of the zli binary"
    )
    parser.add_argument("-p", "--process", type=int, default=4, help="process count")
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=2**30,
        help="chunk size before compression",
    )
    arguments = parser.parse_args()

    if not arguments.fname:
        log("No file provided, exiting.")
        return 0
    if arguments.process > 1024:
        raise ProcessCountError(f"Process count likely too high {arguments.process = }")

    b = os.path.getsize(arguments.fname)
    instructions = [
        Instruction(
            idx=i,
            offset=offset,
            chunk=arguments.chunk_size,
            fname=arguments.fname,
            zli_bin=arguments.bin,
        )
        for i, offset in enumerate(range(0, b, arguments.chunk_size))
    ]
    processes = min(arguments.process, len(instructions))
    with timeit("main"), mp.Pool(processes=processes) as pool:
        pool.map(zli_compress_routine, instructions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
