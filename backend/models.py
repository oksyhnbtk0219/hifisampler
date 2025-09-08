import logging
from pathlib import Path

from config import CONFIG
from util.nsf_hifigan import NsfHifiGAN
from util.wav2mel import PitchAdjustableMelSpectrogram
from util.hnsep import load_sep_model

vocoder = None
ort_session = None
hnsep_model = None
hnsep_ort_session = None
mel_analyzer = None

logging.basicConfig(format='%(message)s', level=logging.INFO)

def initialize_models():
    global vocoder, ort_session, hnsep_model, mel_analyzer
    
    logging.info("Initializing models...")

    # 1. 加载 HifiGAN Vocoder
    vocoder_path = Path(CONFIG.vocoder_path)
    onnx_default_path = Path(
                r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.onnx")
    ckpt_default_path = Path(
        r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt")

    # Determine actual vocoder path
    actual_vocoder_path = None
    if vocoder_path.exists():
        actual_vocoder_path = vocoder_path
    elif ckpt_default_path.exists():
        actual_vocoder_path = ckpt_default_path
        logging.info(
            f"Configured vocoder path not found, using default: {ckpt_default_path}")
    elif onnx_default_path.exists():
        actual_vocoder_path = onnx_default_path
        logging.info(
            f"Configured vocoder path not found, using default: {onnx_default_path}")
    else:
        raise FileNotFoundError(
            f"No HifiGAN model found. Checked configured path '{CONFIG.vocoder_path}' and defaults.")

    # Load
    if actual_vocoder_path.suffix == '.ckpt':
        vocoder = NsfHifiGAN(model_path=actual_vocoder_path)
        vocoder.to_device(CONFIG.device)
        logging.info(f'Loaded HifiGAN (ckpt): {actual_vocoder_path} on {CONFIG.device}')
        logging.info(f'Using device: {CONFIG.device}')
    elif actual_vocoder_path.suffix == '.onnx':
        import onnxruntime
        CONFIG.model_type = 'onnx'
        # Determine available providers, prioritize DML/CUDA over CPU
        available_providers = onnxruntime.get_available_providers()
        preferred_providers = []
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        elif 'DmlExecutionProvider' in available_providers:
            preferred_providers.append('DmlExecutionProvider')
        preferred_providers.append('CPUExecutionProvider')  # Fallback

        # Build the session using the actual resolved model path
        ort_session = onnxruntime.InferenceSession(
            str(actual_vocoder_path), providers=preferred_providers)
        used_provider = ort_session.get_providers()[0]
        logging.info(
            f'Loaded HifiGAN (onnx): {actual_vocoder_path} using providers {ort_session.get_providers()}')
        logging.info(f'Primary provider: {used_provider}')

        # If using DirectML, keep single worker due to known multi-thread request issue
        if used_provider == 'DmlExecutionProvider':
            if CONFIG.max_workers != 1:
                logging.info('DirectML detected: forcing max_workers=1 to avoid DML multi-thread bug.')
            CONFIG.max_workers = 1
        else:
            # For CPU EP (and CUDA EP), allow configured max_workers; CPU EP will run single-thread per request.
            logging.info('ONNX Runtime configured for per-request 1 thread; multi-worker concurrency is allowed.')
    else:
        raise ValueError(f'Unsupported vocoder model type: {vocoder_path.suffix}')

    # 2. 加载 HN-SEP - 自动检测模型类型
    hnsep_model_path = Path(CONFIG.hnsep_model_path)
    hnsep_model, _ = load_sep_model(str(hnsep_model_path), CONFIG.device)

    # 3. 初始化 Mel Spectrogram 工具
    mel_analyzer = PitchAdjustableMelSpectrogram(
        sample_rate=CONFIG.sample_rate,
        n_fft=CONFIG.n_fft,
        win_length=CONFIG.win_size,
        hop_length=CONFIG.origin_hop_size,
        f_min=CONFIG.mel_fmin,
        f_max=CONFIG.mel_fmax,
        n_mels=CONFIG.n_mels
    )
    logging.info(f'Initialized Mel Analysis with hop_size={CONFIG.origin_hop_size}.')

    logging.info("Models initialized successfully.")