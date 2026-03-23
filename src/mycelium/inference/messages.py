"""Inference request and response message types."""

from __future__ import annotations

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """A request to run inference through the pipeline."""

    request_id: str
    model_name: str
    prompt: str


class InferenceResponse(BaseModel):
    """Response from the inference pipeline."""

    request_id: str
    token_ids: list[int] | None = None
    text: str | None = None
    error: str | None = None
