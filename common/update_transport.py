"""Binary transport for tensor updates sent by REST clients."""

import hashlib
import json
import math
import struct

import numpy as np

from common.model import decode_weights


UPDATE_CONTENT_TYPE = "application/vnd.fedsec.tensor-update"
UPDATE_TENSOR_ENCODING = "raw_float32"
_HEADER_LENGTH = struct.Struct("!I")
_DIGEST_LENGTH = hashlib.sha256().digest_size
_TENSOR_FIELDS = (("local_model", "weights"), ("update", "values"))
_FLOAT32 = np.dtype("<f4")


def encode_update(payload: dict) -> bytes:
    header = {**payload, "tensor_encoding": UPDATE_TENSOR_ENCODING}
    tensors = []
    for section, field in _TENSOR_FIELDS:
        shapes = {}
        for name, tensor in decode_weights(payload[section][field]).items():
            shapes[name] = list(tensor.shape)
            tensors.append(tensor.numpy().astype(_FLOAT32, copy=False).tobytes(order="C"))
        header[section] = {**payload[section], field: shapes}

    metadata = json.dumps(header, separators=(",", ":")).encode("utf-8")
    chunks = [_HEADER_LENGTH.pack(len(metadata)), metadata, *tensors]
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return b"".join([digest.digest(), *chunks])


def decode_update(body: bytes) -> dict:
    data = memoryview(body)[_DIGEST_LENGTH:]
    if len(data) < _HEADER_LENGTH.size:
        raise ValueError("Truncated tensor update")
    if hashlib.sha256(data).digest() != body[:_DIGEST_LENGTH]:
        raise ValueError("Tensor update checksum mismatch")

    (header_length,) = _HEADER_LENGTH.unpack_from(data)
    offset = _HEADER_LENGTH.size + header_length
    if offset > len(data):
        raise ValueError("Truncated tensor update metadata")
    try:
        payload = json.loads(data[_HEADER_LENGTH.size:offset].tobytes())
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid tensor update metadata") from error
    if not isinstance(payload, dict) or payload.get("tensor_encoding") != UPDATE_TENSOR_ENCODING:
        raise ValueError("Unsupported tensor update format")

    for section, field in _TENSOR_FIELDS:
        entries = payload.get(section)
        shapes = entries.get(field) if isinstance(entries, dict) else None
        if not isinstance(shapes, dict) or not shapes:
            raise ValueError(f"Missing tensor update {section}.{field}")
        weights = {}
        for name, shape in shapes.items():
            if not isinstance(shape, list) or any(type(size) is not int or size < 1 for size in shape):
                raise ValueError("Invalid update tensor shape")
            size = math.prod(shape)
            end = offset + size * _FLOAT32.itemsize
            if end > len(data):
                raise ValueError("Truncated update tensor")
            try:
                weights[name] = np.frombuffer(data, dtype=_FLOAT32, count=size, offset=offset).reshape(shape).astype(np.float32, copy=True)
            except ValueError as error:
                raise ValueError("Invalid update tensor shape") from error
            offset = end
        entries[field] = weights
    if offset != len(data):
        raise ValueError("Unexpected trailing data in tensor update")
    return payload
