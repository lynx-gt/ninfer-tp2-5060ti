"""Repack the QUASAR NVFP4 ``.ninfer`` artifact onto this fork's W4A4 contract.

The QUASAR artifact
(`MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer`) already stores every tensor
the W4A4 tier needs, byte-for-byte — but it declares ``identity.weights_id =
nvfp4``, whose registered contract does not match its endpoint descriptors,
and it keeps the 48 GDN projections fused as ``gdn/a_b_projection`` where the
W4A4 contract binds separate ``a_projection`` / ``b_projection`` objects.

This repack is strictly lossless and touches no payload byte:

* each fused ``gdn/a_b_projection`` (shape ``[96, 5120]`` BF16, 983,040 B) is
  replaced by two directory views over the same payload: ``a_projection``
  (first half) and ``b_projection`` (second half) — no copy, no requantization;
* ``identity.weights_id`` is rewritten ``nvfp4`` -> ``nvfp4-w4a4`` so the
  engine selects the W4A4 tier;
* the file is rewritten as ``prefix + header length + header JSON + zero
  padding up to align_up(16 + header, 4096) + payload`` (the layout
  ``src/artifact/reader.cpp`` derives), with the object directory sorted by
  ascending, non-overlapping offsets as the reader requires.

Invocation::

    python3 -m tools.convert.qwen3_8_27b.repack_quasar \
      --src /path/to/qwen3_8_27b_nvfp4.ninfer \
      --out models/qwen3_8_27b_quasar_w4a4.ninfer
"""

from __future__ import annotations

import argparse
import json
import os
import struct

ALIGN = 4096
PREFIX = 16
MAGIC = b"NINFER\x00\x02"
PARENT_SUFFIX = "/gdn/a_b_projection"
HALF_BYTES = 491520


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="QUASAR .ninfer artifact (identity nvfp4)")
    parser.add_argument("--out", required=True, help="output path for the repacked artifact")
    args = parser.parse_args()

    with open(args.src, "rb") as f:
        prefix = f.read(PREFIX)
        if prefix[:8] != MAGIC:
            raise SystemExit("bad magic: not a version-2 .ninfer container")
        jlen = struct.unpack("<Q", prefix[8:16])[0]
        meta = json.loads(f.read(jlen).decode("utf-8"))
        old_payload = ((PREFIX + jlen + ALIGN - 1) // ALIGN) * ALIGN

    objects = meta["objects"]
    names = {o["name"] for o in objects}

    parents = {}
    added = []
    for o in objects:
        n = o["name"]
        if not n.endswith(PARENT_SUFFIX):
            continue
        if o["format"] != "BF16" or o["layout"] != "contiguous-le-v1":
            raise SystemExit(f"unexpected parent format: {n}")
        if o["shape"] != [96, 5120] or o["bytes"] != 2 * HALF_BYTES:
            raise SystemExit(f"unexpected parent shape: {n}")
        parents[n] = o
        base = n[: -len("a_b_projection")]
        added.append({"name": base + "a_projection", "kind": "tensor", "shape": [48, 5120],
                      "format": "BF16", "layout": "contiguous-le-v1",
                      "offset": o["offset"], "bytes": HALF_BYTES})
        added.append({"name": base + "b_projection", "kind": "tensor", "shape": [48, 5120],
                      "format": "BF16", "layout": "contiguous-le-v1",
                      "offset": o["offset"] + HALF_BYTES, "bytes": HALF_BYTES})
    if len(parents) != 48:
        raise SystemExit(f"expected 48 fused gdn/a_b_projection parents, found {len(parents)}")
    for a in added:
        if a["name"] in names:
            raise SystemExit(f"name collision: {a['name']}")

    kept = [o for o in objects if o["name"] not in parents]
    merged = kept + added
    merged.sort(key=lambda o: o["offset"])
    meta["objects"] = merged

    old_wid = meta["identity"]["weights_id"]
    if old_wid != "nvfp4":
        raise SystemExit(f"expected identity.weights_id 'nvfp4', found {old_wid!r}")
    meta["identity"]["weights_id"] = "nvfp4-w4a4"

    new_js = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    new_payload = ((PREFIX + len(new_js) + ALIGN - 1) // ALIGN) * ALIGN
    if new_payload < old_payload:
        raise SystemExit("header shrank the payload offset; this layout shift is not supported")

    tmp = args.out + ".part"
    with open(tmp, "wb") as out, open(args.src, "rb") as src:
        out.write(MAGIC + struct.pack("<Q", len(new_js)))
        out.write(new_js)
        out.write(b"\x00" * (new_payload - PREFIX - len(new_js)))
        src.seek(old_payload, os.SEEK_SET)
        copied = 0
        while True:
            chunk = src.read(64 << 20)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
    os.replace(tmp, args.out)

    payload_size = os.path.getsize(args.src) - old_payload
    if copied != payload_size:
        raise SystemExit(f"payload copy mismatch: {copied} != {payload_size}")

    with open(args.out, "rb") as f:
        pre = f.read(PREFIX)
        jl = struct.unpack("<Q", pre[8:16])[0]
        check = json.loads(f.read(jl).decode("utf-8"))
    by_name = {o["name"]: o for o in check["objects"]}
    probe = "text/layers/0/gdn/"
    if by_name[probe + "a_projection"]["offset"] != parents[probe + "a_b_projection"]["offset"]:
        raise SystemExit("self-check failed: a_projection view offset")
    if by_name[probe + "b_projection"]["offset"] != parents[probe + "a_b_projection"]["offset"] + HALF_BYTES:
        raise SystemExit("self-check failed: b_projection view offset")
    if check["identity"]["weights_id"] != "nvfp4-w4a4":
        raise SystemExit("self-check failed: identity")
    print(f"repacked {len(kept)} objects + {len(added)} views, "
          f"payload {copied} bytes -> {args.out}")


if __name__ == "__main__":
    main()
