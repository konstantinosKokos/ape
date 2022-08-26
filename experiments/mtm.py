from typing import Callable

from treePE.data.tree import Tree, Node, breadth_first, depth_first
from treePE.data.tokenization import TreeTokenizer
from treePE.data.batching import make_cfn_mtm

from treePE.neural.positional_encoders import PositionalEncoder
from treePE.neural.models import MaskedTreeModeling

from torch import device
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pickle


def run_one(data_path: str,
            tokenizer_path: str,
            num_heads: int,
            num_layers: int,
            dim: int,
            traversal: Callable[[Tree[Node]], list[Node]], positional_encoder: PositionalEncoder) -> None:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    tokenizer = TreeTokenizer.from_file(tokenizer_path)
    cfn = make_cfn_mtm(tokenizer.PAD_token_id, device('cuda'), traversal)

    train, dev, test = data
    train_dl = DataLoader(train, batch_size=64, collate_fn=cfn, shuffle=True)   # type: ignore
    dev_dl = DataLoader(dev, batch_size=512, collate_fn=cfn, shuffle=False)     # type: ignore
    test_dl = DataLoader(test, batch_size=512, collate_fn=cfn, shuffle=False)   # type: ignore

    model = MaskedTreeModeling(vocab_size=len(tokenizer), num_heads=num_heads, num_layers=num_layers,
                               dim=dim, positional_encoder=positional_encoder)
    model = model.to(device('cuda'))

    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(99):
        print(f'Epoch {epoch}')
        print('=' * 64)
        model.train()
        train_loss, train_correct, train_total = \
            model.go_epoch(data=train_dl, masking_value=tokenizer.MASK_token_id, optimizer=optim)
        print(f'Train loss: {train_loss}, accuracy: {train_correct}/{train_total} ({train_correct / train_total})')
        model.eval()
        dev_loss, dev_correct, dev_total = \
            model.go_epoch(data=dev_dl, masking_value=tokenizer.MASK_token_id, optimizer=None)
        print(f'Dev loss: {dev_loss}, accuracy: {dev_correct}/{dev_total} ({dev_correct / dev_total})')
        test_loss, test_correct, test_total = \
            model.go_epoch(data=test_dl, masking_value=tokenizer.MASK_token_id, optimizer=None)
        print(f'Test loss: {test_loss}, accuracy: {test_correct}/{test_total} ({test_correct / test_total})')
        print('\n')