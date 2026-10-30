"""Experimental decoder rebinding: verifies frozen backbone; does not imply calibration."""
from pathlib import Path
import torch
from training.train_conversation import sha256


def main():
    old = Path('runs/phase12/generator.pt'); new = Path('runs/phase12_consolidation/generator.pt')
    a = torch.load(old, map_location='cpu', weights_only=True); b = torch.load(new, map_location='cpu', weights_only=True)
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a if not k.startswith(('intent_encoder.', 'intent_classifier.'))):
        raise RuntimeError('generator differs outside intent components')
    dest = new.parent
    for file, source in [('response.pt', Path('runs/phase12/response.pt')), ('vector_reader.pt', Path('runs/phase12/vector_reader.pt')),
                         ('diverse_response.pt', dest/'diverse_training/diverse_response.pt')]:
        state = torch.load(source, map_location='cpu', weights_only=True)
        if state['generator_sha256'] != sha256(old): raise RuntimeError('unexpected original binding')
        state['original_generator_sha256'] = state['generator_sha256']
        state['generator_sha256'] = sha256(new)
        state['experimental_rebinding'] = 'intent parameters changed; mandatory downstream revalidation'
        torch.save(state, dest/file)
        print(file, sha256(dest/file))

if __name__ == '__main__': main()
