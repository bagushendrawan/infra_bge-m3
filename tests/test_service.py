import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.encoding.config import Settings
from src.encoding.exceptions import EncodingError
from src.encoding.service import Encoder, serialize_outputs


@pytest.mark.parametrize("device,fp16", [("cpu", False), ("cuda:0", True)])
def test_device(monkeypatch, device, fp16):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=factory))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    Encoder(Settings(device=device))
    assert factory.call_args.kwargs["devices"] == device
    assert factory.call_args.kwargs["use_fp16"] is fp16


@pytest.mark.parametrize(
    "flags", [(True, False, False), (False, True, False), (False, False, True), (True, True, True)]
)
def test_serialization(flags):
    output = {
        "dense_vecs": [Mock(tolist=lambda: [1.0]), Mock(tolist=lambda: [2.0])],
        "lexical_weights": [{1: 0.5}, {2: 0.75}],
        "colbert_vecs": [Mock(tolist=lambda: [[1.0]]), Mock(tolist=lambda: [[2.0]])],
    }
    payload = dict(zip(["return_dense", "return_sparse", "return_colbert_vecs"], flags))
    payload["texts"] = ["first", "second"]
    result = serialize_outputs(output, payload, "test")
    assert len(result) == 1 + sum(flags)
    if flags[0]:
        assert result["dense_vecs"] == [[1.0], [2.0]]
    if flags[1]:
        assert result["lexical_weights"] == [{"1": 0.5}, {"2": 0.75}]
    if flags[2]:
        assert result["colbert_vecs"] == [[[1.0]], [[2.0]]]


def test_output_limit(monkeypatch):
    class OOM(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM))
    )
    encoder = object.__new__(Encoder)
    encoder.settings = Settings(max_result_bytes=1)
    encoder.model = Mock()
    encoder.model.encode.return_value = {"lexical_weights": [{"1": 0.5}]}
    with pytest.raises(EncodingError) as error:
        encoder.encode(
            {
                "texts": ["a"],
                "return_dense": False,
                "return_sparse": True,
                "return_colbert_vecs": False,
                "max_length": 512,
            }
        )
    assert error.value.status == 413
