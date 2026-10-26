"""Native Auto-AVSR graph plus explicitly named research interventions."""

import sys
import numpy as np
import torch
from torch import nn
from .common import resolve, digest
from .readout import BoundaryReadout
from .text import Tokenizer, required_frames
from .ctc_loss import cpu_ctc
from .spatial_pool import replace_spatial_pool

METHODS = ('B0', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'A5')


class Recognizer(nn.Module):
    def __init__(self, config, method='B0', device='cuda', scope='all'):
        super().__init__()
        if method not in METHODS or scope not in {'all', 'heads'}:
            raise ValueError('Unsupported method or training scope')
        sys.path.insert(0, str(resolve(config, 'source')))
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
        self.tokenizer = Tokenizer(resolve(config, 'tokenizer'), resolve(config, 'units'), config['language'])
        self.native = E2E(len(self.tokenizer.tokens), 'video', ctc_weight=0.1)
        checkpoint = resolve(config, 'checkpoint')
        if not config.get('checkpoint_sha256') or digest(checkpoint) != config['checkpoint_sha256']:
            raise ValueError('Checkpoint must have a verified SHA-256 in the dataset config')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        field = config.get('state_field', 'raw')
        state = payload if field == 'raw' else payload[field]
        prefix = config.get('state_prefix', '')
        if prefix:
            state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        self.native.load_state_dict(state, strict=True)
        replace_spatial_pool(self.native.frontend)
        del state, payload
        self.method = method
        self.scope = scope
        self.intermediate = None
        self.auxiliary = nn.Linear(768, self.native.odim) if method == 'T3' else None
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(self.native.ctc.ctc_lo.state_dict())
            self.native.encoder.encoders[5].register_forward_hook(self._capture)
        self.readout = BoundaryReadout() if method in {'T5', 'A5'} else None
        if scope == 'heads':
            for module in [self.native.frontend, self.native.proj_encoder, self.native.encoder]:
                module.requires_grad_(False)
        self.to(device)
        self.eval()

    def _capture(self, module, inputs, output):
        value = output[0]
        self.intermediate = value[0] if isinstance(value, tuple) else value

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if self.scope == 'heads':
            self.native.frontend.eval()
            self.native.encoder.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, frames):
        if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (96, 96):
            raise ValueError('Expected uint8 T x 96 x 96 frames')
        tensor = torch.as_tensor(np.ascontiguousarray(frames[:, 4:92, 4:92]), device=self.device).float()
        tensor = ((tensor / 255 - 0.421) / 0.165).unsqueeze(0).unsqueeze(2)
        features = self.native.frontend(tensor)
        encoded = self.native.encoder(self.native.proj_encoder(features), None)[0]
        if encoded.shape != (1, len(frames), 768):
            raise ValueError('Frame-to-feature alignment changed')
        return encoded

    def representation(self, encoded, start, end):
        if not 0 <= start < end <= encoded.size(1):
            raise ValueError('Invalid target span')
        if self.readout is None:
            return encoded[:, start:end]
        return self.readout(encoded, start, end, boundary=self.method == 'T5')

    def asr_loss(self, encoded, text):
        from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
        from espnet.nets.pytorch_backend.transformer.mask import target_mask
        ids = self.tokenizer.encode(text)
        if required_frames(ids) > encoded.size(1):
            raise ValueError('Infeasible CTC target; must be excluded before training')
        target = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([encoded.size(1)], device=self.device)
        with torch.autocast(self.device.type, enabled=False):
            logits = self.native.ctc.ctc_lo(self.native.ctc.dropout(encoded.float())).transpose(0, 1)
            ctc = cpu_ctc(logits.log_softmax(2), target.flatten(), lengths, [len(ids)]) / logits.size(1)
        ys_in, ys_out = add_sos_eos(target, self.native.sos, self.native.eos, -1)
        mask = torch.ones(1, 1, encoded.size(1), dtype=torch.bool, device=self.device)
        prediction = self.native.decoder(ys_in, target_mask(ys_in, -1), encoded, mask)[0]
        attention = self.native.criterion(prediction, ys_out)
        loss = .1 * ctc + .9 * attention
        return loss, {'ctc': float(ctc.detach()), 'attention': float(attention.detach())}

    def auxiliary_loss(self, intermediate, text):
        ids = self.tokenizer.encode(text)
        targets = torch.tensor(ids, device=self.device, dtype=torch.long)
        logp = self.auxiliary(intermediate).float().log_softmax(-1).transpose(0, 1)
        return cpu_ctc(logp, targets, [len(logp)], [len(ids)], zero_infinity=False) / len(ids)

    def greedy(self, encoded):
        ids = self.native.ctc.log_softmax(encoded)[0].argmax(-1).unique_consecutive().tolist()
        return self.tokenizer.decode(ids)

    def counts(self):
        return {'total_parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)}
