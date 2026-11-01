from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator


class EncodeRequest(BaseModel):
    texts: list[Annotated[str, StringConstraints(min_length=1)]] = Field(min_length=1)
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert_vecs: bool = False
    max_length: int = Field(default=512, ge=2, le=8192)

    @model_validator(mode="after")
    def validate_request(self):
        if any(not text.strip() for text in self.texts):
            raise ValueError("Texts must not be blank")
        if not (self.return_dense or self.return_sparse or self.return_colbert_vecs):
            raise ValueError("Enable at least one output")
        return self


class EncodeResponse(BaseModel):
    model: str
    dense_vecs: list[list[float]] | None = None
    lexical_weights: list[dict[str, float]] | None = None
    colbert_vecs: list[list[list[float]]] | None = None
