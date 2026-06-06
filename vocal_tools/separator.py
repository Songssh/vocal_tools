from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

from vocal_tools.catalog import ModelCatalog
from vocal_tools.core.errors import InvalidOptionError, UnsupportedEngineError
from vocal_tools.core.progress import ProgressCallback
from vocal_tools.core.types import OptionSpec, SeparationRequest, SeparationResult
from vocal_tools.engines import DemucsEngine, MDXCEngine, MDXOnnxEngine, VRArchEngine
from vocal_tools.engines.base import SeparationEngine


class Separator:
    def __init__(
        self,
        catalog: Optional[ModelCatalog] = None,
        model_dir=None,
        engines: Optional[Mapping[str, SeparationEngine]] = None,
    ):
        if catalog is None:
            if model_dir is None:
                raise ValueError("Either catalog or model_dir must be provided.")
            catalog = ModelCatalog(model_dir=model_dir)

        self.catalog = catalog
        self.engines: Dict[str, SeparationEngine] = {
            "mdx_onnx": MDXOnnxEngine(),
            "demucs": DemucsEngine(),
            "vr_arch": VRArchEngine(),
            "mdxc": MDXCEngine(),
        }
        if engines is not None:
            self.engines.update(dict(engines))

    def separate(
        self,
        input_audio,
        output_dir,
        model_id: str,
        options: Optional[Dict[str, object]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SeparationResult:
        input_audio = Path(input_audio)
        output_dir = Path(output_dir)
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio}")

        model = self.catalog.get_model(model_id)
        engine = self._get_engine(model.arch)
        option_schema = engine.get_option_schema(model)
        options_used = self._merge_options(model.defaults, option_schema, options or {})
        self._validate_options(model.defaults, option_schema, options_used, options or {})

        model_path = self.catalog.store.ensure_model(model, progress_callback=progress_callback)
        request = SeparationRequest(
            input_audio=input_audio,
            output_dir=output_dir,
            model=model,
            model_path=model_path,
            options=options_used,
            option_overrides=dict(options or {}),
        )
        return engine.separate(request, progress_callback=progress_callback)

    def _get_engine(self, arch: str) -> SeparationEngine:
        try:
            return self.engines[arch]
        except KeyError as exc:
            raise UnsupportedEngineError(f"Unsupported separation engine: {arch}") from exc

    def _merge_options(
        self,
        defaults: Dict[str, object],
        option_schema: list[OptionSpec],
        overrides: Dict[str, object],
    ) -> Dict[str, object]:
        merged = dict(defaults)
        for option in option_schema:
            if option.name not in merged:
                merged[option.name] = option.default
        merged.update(overrides)
        return merged

    def _validate_options(
        self,
        defaults: Dict[str, object],
        option_schema: list[OptionSpec],
        options_used: Dict[str, object],
        overrides: Dict[str, object],
    ) -> None:
        specs = {option.name: option for option in option_schema}
        allowed = set(defaults) | set(specs)
        unknown = sorted(name for name in overrides if name not in allowed)
        if unknown:
            raise InvalidOptionError(f"Unknown option(s): {', '.join(unknown)}")

        for name, spec in specs.items():
            if name in options_used:
                self._validate_option_value(spec, options_used[name])

    def _validate_option_value(self, spec: OptionSpec, value: object) -> None:
        if value is None:
            return

        if spec.type == "bool":
            valid = isinstance(value, bool)
        elif spec.type == "int":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif spec.type == "float":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif spec.type == "string":
            valid = isinstance(value, str)
        elif spec.type == "enum":
            valid = spec.choices is None or value in spec.choices
        elif spec.type == "dict":
            valid = isinstance(value, dict)
        elif spec.type == "list":
            valid = isinstance(value, list)
        else:
            raise InvalidOptionError(f"Unsupported option type for {spec.name!r}: {spec.type!r}")

        if not valid:
            raise InvalidOptionError(
                f"Invalid value for option {spec.name!r}: expected {spec.type}, got {type(value).__name__}."
            )

        if spec.choices is not None and value not in spec.choices:
            raise InvalidOptionError(f"Invalid value for option {spec.name!r}: {value!r}")
        if spec.min is not None and value < spec.min:
            raise InvalidOptionError(f"Option {spec.name!r} must be >= {spec.min}.")
        if spec.max is not None and value > spec.max:
            raise InvalidOptionError(f"Option {spec.name!r} must be <= {spec.max}.")
