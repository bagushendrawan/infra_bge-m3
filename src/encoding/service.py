import json

from src.encoding.exceptions import EncodingError


def serialize_outputs(output, payload, model):
    result = {"model": model}
    for flag, name in [
        ("return_dense", "dense_vecs"),
        ("return_sparse", "lexical_weights"),
        ("return_colbert_vecs", "colbert_vecs"),
    ]:
        if payload[flag]:
            values = output[name]
            if len(values) != len(payload["texts"]):
                raise RuntimeError("Model output count mismatch")
            result[name] = (
                [{str(k): float(v) for k, v in item.items()} for item in values]
                if name == "lexical_weights"
                else [item.tolist() for item in values]
            )
    return result


class Encoder:
    def __init__(self, settings):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        self.settings = settings
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; configure BGE_M3_DEVICE=cpu for CPU inference")
        self.model = BGEM3FlagModel(
            settings.model,
            devices=settings.device,
            use_fp16=settings.device != "cpu",
            normalize_embeddings=True,
        )

    def encode(self, payload):
        import torch

        try:
            output = self.model.encode(
                **{
                    "sentences": payload["texts"],
                    "batch_size": self.settings.batch_size,
                    **{k: v for k, v in payload.items() if k != "texts"},
                }
            )
            result = serialize_outputs(output, payload, self.settings.model)
            if len(json.dumps(result, allow_nan=False).encode()) > self.settings.max_result_bytes:
                raise EncodingError(413, "Encoded result exceeds size limit")
            return result
        except torch.cuda.OutOfMemoryError as exc:
            raise EncodingError(503, "GPU out of memory") from exc
