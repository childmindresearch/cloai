"""Unit tests for the test_llm module."""

import json

import pydantic

from cloai.llm import llm


class ModelPrimitive(pydantic.BaseModel):
    """Model containing basic Python types."""

    var1: int = 1
    var2: float = 2.0
    var3: str = "a"
    var4: list[str] = ("a",)  # type: ignore[assignment]


class ModelRecursive(pydantic.BaseModel):
    """Model containing a model."""

    primitive: ModelPrimitive = ModelPrimitive()


def test_recursive_pydantic_model_dump_primitive() -> None:
    """Test dumping a model containing primitives."""
    model = ModelPrimitive()
    expected = model.model_dump()

    actual = llm._recursive_pydantic_model_dump(model)

    assert actual == expected


def test_recursive_pydantic_model_dump_recursive() -> None:
    """Test dumping a model containing a model."""
    model = ModelRecursive()
    expected = model.model_dump()
    expected["primitive"] = ModelPrimitive().model_dump()

    actual = llm._recursive_pydantic_model_dump(model)

    assert actual == expected


def test_model_to_string_model() -> None:
    """Test converting a model to a string."""
    model = ModelPrimitive()
    expected = json.dumps(model.model_dump())

    actual = llm._model_to_string(model)

    assert actual == expected
