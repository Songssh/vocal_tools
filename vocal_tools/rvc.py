from multiprocessing import cpu_count
from pathlib import Path
from dataclasses import dataclass, replace
from contextlib import contextmanager
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("RVC conversion requires PyTorch. Install the RVC runtime dependencies first.") from exc
    return torch


@dataclass(frozen=True)
class RVCVoiceModelPaths:
    model_path: Path
    index_path: Optional[Path] = None


@dataclass(frozen=True)
class RVCConversionOptions:
    pitch_change: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.5
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    crepe_hop_length: int = 128
    sid: int = 0
    resample_sr: int = 0
    f0_file: Optional[object] = None


@dataclass
class RVCLoadedModel:
    paths: RVCVoiceModelPaths
    cpt: dict
    version: str
    net_g: object
    tgt_sr: int
    vc: object


def resolve_device(device="auto"):
    torch = _import_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return str(device)


def resolve_half(is_half, device):
    if is_half in (None, "auto"):
        return str(device).startswith("cuda")
    return bool(is_half)


def load_audio(file, sr):
    try:
        import ffmpeg
        import numpy as np

        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


class Config:
    def __init__(self, device="auto", is_half="auto", rmvpe_path=None):
        self._torch = _import_torch()
        self.device = resolve_device(device)
        self.is_half = resolve_half(is_half, self.device)
        self.rmvpe_path = str(rmvpe_path) if rmvpe_path else None
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        torch = self._torch
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif (
            self.device == "mps"
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            print("Use MPS for inference")
            self.is_half = False
        else:
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device, is_half, model_path):
    from fairseq import checkpoint_utils

    with _fairseq_torch_load_compat():
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='', )
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


@contextmanager
def _fairseq_torch_load_compat():
    torch = _import_torch()
    original_load = torch.load

    def load_with_legacy_checkpoint_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_checkpoint_default
    try:
        yield
    finally:
        torch.load = original_load


def get_vc(device, is_half, config, model_path):
    torch = _import_torch()
    from .infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )
    from .vc_infer_pipeline import VC

    cpt = torch.load(model_path, map_location='cpu')
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError(f"Unsupported RVC model version: {version}")

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    vc.rmvpe_path = getattr(config, "rmvpe_path", None)
    return cpt, version, net_g, tgt_sr, vc


def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, sid=0, resample_sr=0, f0_file=None, rmvpe_path=None):
    from scipy.io import wavfile

    if rmvpe_path:
        vc.rmvpe_path = str(rmvpe_path)

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(hubert_model, net_g, sid, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate, version, protect, crepe_hop_length, f0_file=f0_file)
    output_sr = resample_sr if resample_sr >= 16000 and tgt_sr != resample_sr else tgt_sr
    wavfile.write(output_path, output_sr, audio_opt)
    return output_path


class RVCConverter:
    def __init__(self, models_dir, hubert_path=None, rmvpe_path=None, device="auto", is_half="auto"):
        self.models_dir = Path(models_dir)
        self.hubert_path = Path(hubert_path) if hubert_path else self.models_dir / "hubert_base.pt"
        self.rmvpe_path = Path(rmvpe_path) if rmvpe_path else self.models_dir / "rmvpe.pt"
        self.config = Config(device=device, is_half=is_half, rmvpe_path=self.rmvpe_path)
        self.device = self.config.device
        self.is_half = self.config.is_half
        self._hubert_model = None
        self._model_cache: Dict[str, RVCLoadedModel] = {}
        self._active_model: Optional[RVCLoadedModel] = None

    def find_voice_model(self, voice_model):
        candidate = Path(voice_model)
        if not candidate.is_absolute():
            candidate = self.models_dir / candidate

        if candidate.is_file():
            if candidate.suffix.lower() != ".pth":
                raise ValueError(f"RVC voice model must be a .pth file: {candidate}")
            model_path = candidate
            search_dir = candidate.parent
        elif candidate.is_dir():
            pth_files = sorted(candidate.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No RVC .pth model found in: {candidate}")
            model_path = pth_files[0]
            search_dir = candidate
        else:
            raise FileNotFoundError(f"RVC voice model not found: {candidate}")

        index_files = sorted(search_dir.glob("*.index"))
        index_path = index_files[0] if index_files else None
        return RVCVoiceModelPaths(model_path=model_path, index_path=index_path)

    def _ensure_hubert(self):
        if self._hubert_model is None:
            if not self.hubert_path.exists():
                raise FileNotFoundError(f"Hubert model not found: {self.hubert_path}")
            self._hubert_model = load_hubert(self.device, self.is_half, str(self.hubert_path))
        return self._hubert_model

    def load_voice_model(self, voice_model):
        paths = self.find_voice_model(voice_model)
        cache_key = str(paths.model_path.resolve())
        if cache_key not in self._model_cache:
            cpt, version, net_g, tgt_sr, vc = get_vc(
                self.device,
                self.is_half,
                self.config,
                str(paths.model_path),
            )
            vc.rmvpe_path = str(self.rmvpe_path)
            self._model_cache[cache_key] = RVCLoadedModel(
                paths=paths,
                cpt=cpt,
                version=version,
                net_g=net_g,
                tgt_sr=tgt_sr,
                vc=vc,
            )

        self._active_model = self._model_cache[cache_key]
        return self._active_model

    def convert(self, input_audio, output_audio, voice_model=None, options=None, **option_overrides):
        options = options or RVCConversionOptions()
        if option_overrides:
            options = replace(options, **option_overrides)

        loaded_model = self.load_voice_model(voice_model) if voice_model else self._active_model
        if loaded_model is None:
            raise ValueError("voice_model is required before a model has been loaded.")

        output_audio = Path(output_audio)
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        index_path = str(loaded_model.paths.index_path) if loaded_model.paths.index_path else ""

        return Path(
            rvc_infer(
                index_path,
                options.index_rate,
                str(input_audio),
                str(output_audio),
                options.pitch_change,
                options.f0_method,
                loaded_model.cpt,
                loaded_model.version,
                loaded_model.net_g,
                options.filter_radius,
                loaded_model.tgt_sr,
                options.rms_mix_rate,
                options.protect,
                options.crepe_hop_length,
                loaded_model.vc,
                self._ensure_hubert(),
                sid=options.sid,
                resample_sr=options.resample_sr,
                f0_file=options.f0_file,
                rmvpe_path=self.rmvpe_path,
            )
        )
