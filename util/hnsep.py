import logging
import os
from pathlib import Path
import yaml
import torch
from hnsep.nets import CascadedNet
from util.audio import DotDict

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
        # Store the original device for later
        original_device = x.device
        
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
        
        # Convert back to torch tensor and move to original device
        output_tensor = torch.from_numpy(output).to(original_device)
        
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
