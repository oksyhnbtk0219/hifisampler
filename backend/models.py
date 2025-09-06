import logging
import os
from pathlib import Path
import torch
import yaml

from config import CONFIG
from hnsep.nets import CascadedNet
from util.nsf_hifigan import NsfHifiGAN
from util.wav2mel import PitchAdjustableMelSpectrogram
from util.audio import DotDict

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

class OnnxHnsepModel:
    """ONNX inference wrapper for HN-SEP model"""
    
    def __init__(self, onnx_path, config_args):
        import onnxruntime
        
        # Determine available providers, prioritize performance
        available_providers = onnxruntime.get_available_providers()
        preferred_providers = []
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        elif 'DmlExecutionProvider' in available_providers:
            preferred_providers.append('DmlExecutionProvider')
        preferred_providers.append('CPUExecutionProvider')  # Fallback
        
        self.session = onnxruntime.InferenceSession(str(onnx_path), providers=preferred_providers)
        self.config = config_args
        self.n_fft = config_args['n_fft']
        self.hop_length = config_args['hop_length']
        self.max_bin = self.n_fft // 2
        self.output_bin = self.n_fft // 2 + 1
        self.offset = 64
        
        used_provider = self.session.get_providers()[0]
        logging.info(f'Loaded HN-SEP (ONNX): {onnx_path} using provider {used_provider}')
    
    def forward(self, x):
        """
        Forward pass with complex spectrogram input
        x: complex tensor [batch, channels, freq, time]
        """
        # Convert complex tensor to real/imaginary parts for ONNX
        if torch.is_complex(x):
            # Convert complex to [batch, 2, freq, time] format
            real_part = x.real
            imag_part = x.imag
            x_input = torch.cat([real_part, imag_part], dim=1)
        else:
            x_input = x
        
        # Convert to numpy for ONNX inference
        x_numpy = x_input.detach().cpu().numpy()
        
        # Run ONNX inference
        output = self.session.run(['output'], {'input': x_numpy})[0]
        
        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output)
        
        # Convert back to complex format if needed
        if output_tensor.shape[1] == 2:
            # Split real and imaginary parts
            real_part = output_tensor[:, :1]
            imag_part = output_tensor[:, 1:]
            output_tensor = torch.complex(real_part, imag_part)
        
        return output_tensor
    
    def predict_mask(self, x):
        """Predict mask with offset handling"""
        mask = self.forward(x)
        
        if self.offset > 0:
            mask = mask[:, :, :, self.offset:-self.offset]
            assert mask.size()[3] > 0
        
        return mask
    
    def predict(self, x):
        """Predict separated audio"""
        mask = self.forward(x)
        pred = x * mask
        
        if self.offset > 0:
            pred = pred[:, :, :, self.offset:-self.offset]
            assert pred.size()[3] > 0
        
        return pred
    
    def predict_fromaudio(self, x):
        """Predict from audio input"""
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        T1 = T + self.hop_length
        seg_length = 32 * self.hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1       
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length       
        x = torch.nn.functional.pad(x, (Tl_pad, T_pad - Tl_pad))
        
        # Create Hann window
        window = torch.hann_window(self.n_fft).to(x.device)
        
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=window,
            pad_mode='constant'
        )
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        
        mask = self.forward(spec)
        spec_pred = spec * mask
        spec_pred = spec_pred.reshape(B * C, spec.shape[-2], spec.shape[-1])
        
        x_pred = torch.istft(spec_pred, self.n_fft, self.hop_length, window=window)
        x_pred = x_pred[:, Tl_pad: Tl_pad + T]
        x_pred = x_pred.reshape(B, C, T)
        return x_pred

def load_sep_model(model_path, device=torch.device('cpu')):
    """Load HN-SEP model from checkpoint or ONNX."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_file = os.path.join(model_dir, 'config.yaml')
    
    with open(config_file, "r") as config:
        args_dict = yaml.safe_load(config)
    args = DotDict(args_dict)
    
    model_path_obj = Path(model_path)
    
    # Check if ONNX model should be used
    if model_path_obj.suffix == '.onnx':
        # Use ONNX model
        model = OnnxHnsepModel(model_path, args_dict)
        logging.info(f"Loaded HN-SEP model (ONNX): {model_path}")
        return model, args
    else:
        # Use PyTorch model
        model = CascadedNet(
            args_dict['n_fft'],
            args_dict['hop_length'],
            args_dict['n_out'],
            args_dict['n_out_lstm'],
            True,
            is_mono=args_dict['is_mono'],
            fixed_length=True if args_dict.get('fixed_length', None) is None else args_dict['fixed_length']
        )
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        logging.info(f"Loaded HN-SEP model (PyTorch): {model_path}")
        return model, args
