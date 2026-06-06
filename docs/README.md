# vocal_tools Documentation

`vocal_tools` is a headless Python audio engine library. It is designed to be imported by host applications that need model catalogs, source separation, RVC conversion, audio mixing, and cover-pipeline helpers.

This project intentionally does not contain a FastAPI server, web UI, database, or job queue. Those belong in the host product. The package also avoids hardcoding SHS GPU Node or SHS AI Tools integration points.

## Documents

| File | Purpose |
| --- | --- |
| [project_summary.md](project_summary.md) | Final project summary for downstream reuse |
| [current_status.md](current_status.md) | Current implementation and known gaps |
| [architecture.md](architecture.md) | Library structure and engine responsibilities |
| [installation.md](installation.md) | Installation and optional extras |
| [public_api.md](public_api.md) | Stable import surface grouped by category |
| [cli.md](cli.md) | `vocal-tools` command-line usage |
| [testing.md](testing.md) | Lightweight smoke tests and inference regression guidance |
| [release_checklist.md](release_checklist.md) | Packaging and release-readiness checklist |
| [model_layout.md](model_layout.md) | Model directory and manifest layout |
| [model_manifest_format.md](model_manifest_format.md) | Builtin, user, remote, and voice manifest shape |
| [usage_mdx.md](usage_mdx.md) | MDX ONNX usage |
| [usage_demucs.md](usage_demucs.md) | Demucs usage and two-stems behavior |
| [usage_rvc.md](usage_rvc.md) | RVC conversion usage |
| [usage_rvc_voice_catalog.md](usage_rvc_voice_catalog.md) | RVC voice model catalog usage |
| [usage_cover_pipeline.md](usage_cover_pipeline.md) | MDX -> RVC -> Mix cover pipeline |
| [usage_catalog.md](usage_catalog.md) | Builtin, remote, and user catalog usage |
| [task_payload_helpers.md](task_payload_helpers.md) | JSON-safe task helper and validation functions |
| [shs_gpu_node_integration.md](shs_gpu_node_integration.md) | How SHS GPU Node can call the library |
| [examples.md](examples.md) | Example file guide |
| [roadmap.md](roadmap.md) | Forward roadmap |

## Quick Example

```python
from vocal_tools import ModelCatalog, Separator

catalog = ModelCatalog(model_dir="test_models/mdx")
separator = Separator(catalog=catalog)

result = separator.separate(
    input_audio="input.wav",
    output_dir="output/mdx",
    model_id="kim_vocal_2",
    options={"denoise": False},
)

print(result.to_dict())
```

## CLI Example

After `python -m pip install -e .`, use the console script:

```powershell
vocal-tools models list --models-dir test_models
vocal-tools voices list --models-dir test_models
```

During development, the module entrypoint also works:

```powershell
python -m vocal_tools.cli --help
```

## Current Scope

Implemented:

- MDX ONNX separation through `ModelCatalog`, `Separator`, and `MDXOnnxEngine`
- Demucs separation through `DemucsEngine`
- RVC conversion through `RVCConverter`
- Audio mixing through `mix_vocal_with_instrumental()`
- Cover pipeline through `run_cover_pipeline()`
- JSON-safe task helper functions under `vocal_tools.tasks`
- Thin `vocal-tools` CLI wrapper around task helpers
- Standard-library smoke tests under `tests/`
- Builtin, user, and remote manifest loading
- Model download status APIs through `ModelStore`
- RVC voice model scanning through `RVCVoiceCatalog`

Not implemented yet:

- VR Arch runtime
- MDXC runtime
- Service layer, API server, web UI, database, or queue

## Import Boundary

`import vocal_tools` is intentionally lightweight. Heavy MDX, Demucs, and RVC runtime packages are imported when the corresponding engine or conversion path is executed.
