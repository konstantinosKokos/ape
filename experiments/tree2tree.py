from typing import Callable, Literal

from treePE.data.tree import Tree, Node, breadth_first, depth_first
from treePE.data.tokenization import TreeTokenizer
from treePE.data.batching import make_cfn_tree2tree

from treePE.neural.positional_encoders import PositionalEncoder
from treePE.neural.models import Tree2Tree
from treePE.neural.schedule import make_schedule

from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import pickle


def run_one(data_path: str,
            enc_tokenizer_path: str,
            dec_tokenizer_path: str,
            enc_positional_encoder: PositionalEncoder,
            dec_positional_encoder: PositionalEncoder,
            num_heads: int,
            num_layers: int,
            dim: int,
            enc_traversal: Callable[[Tree[Node]], list[Node]],
            dec_traversal: Callable[[Tree[Node]], list[Node]],
            regression: Literal['depth', 'breadth', 'level'],
            num_epochs: int = 100) -> None:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    enc_tokenizer = TreeTokenizer.from_file(enc_tokenizer_path)
    dec_tokenizer = TreeTokenizer.from_file(dec_tokenizer_path)
    cfn = make_cfn_tree2tree(enc_mask_on=enc_tokenizer.PAD_token_id,
                             dec_mask_on=dec_tokenizer.PAD_token_id,
                             device=device('cuda'),
                             enc_traversal=enc_traversal,
                             dec_traversal=dec_traversal,
                             regression=regression,
                             mask_idx=dec_tokenizer.MASK_token_id)
    train, dev, test = data
    train_dl = DataLoader(train, batch_size=64, collate_fn=cfn, shuffle=True)   # type: ignore
    dev_dl = DataLoader(dev, batch_size=512, collate_fn=cfn, shuffle=False)     # type: ignore
    test_dl = DataLoader(test, batch_size=512, collate_fn=cfn, shuffle=False)   # type: ignore

    model = Tree2Tree(
        dim=dim, num_heads=num_heads,
        enc_vocab_size=len(enc_tokenizer), enc_num_layers=num_layers, enc_positional_encoder=enc_positional_encoder,
        dec_vocab_size=len(dec_tokenizer), dec_num_layers=num_layers, dec_positional_encoder=dec_positional_encoder)
    model = model.to(device('cuda'))

    optimizer = AdamW([{'params': model.encoder.encoder_layers.parameters(), 'lr': 1e-3},
                       {'params': model.decoder.decoder_layers.parameters(), 'lr': 1e-3},
                       {'params': model.encoder.embedding.parameters(), 'lr': 1e-3},
                       {'params': model.decoder.embedding.parameters(), 'lr': 1e-3},
                       {'params': model.encoder.positional_encoder.parameters(), 'lr': 1e-4},
                       {'params': model.decoder.positional_encoder.parameters(), 'lr': 1e-4}],
                      weight_decay=1e-2)
    schedule = make_schedule(warmup_steps=int(0.1 * len(train_dl) * num_epochs),
                             warmdown_steps=int(0.9 * len(train_dl) * num_epochs),
                             total_steps=num_epochs * len(train_dl),
                             max_lr=1,
                             min_lr=1e-2)
    scheduler = LambdaLR(optimizer, [schedule for _ in range(len(optimizer.param_groups))], last_epoch=-1)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        print('=' * 64)
        model.train()
        train_loss, train_correct, train_total = \
            model.go_epoch(data=train_dl, mask_idx=dec_tokenizer.MASK_token_id,
                           pad_idx=dec_tokenizer.PAD_token_id, opt_schedule=(optimizer, scheduler))
        print(f'Train loss: {train_loss}, accuracy: {train_correct}/{train_total} ({train_correct / train_total})')
        model.eval()
        dev_loss, dev_correct, dev_total = \
            model.go_epoch(data=dev_dl, mask_idx=dec_tokenizer.MASK_token_id,
                           pad_idx=dec_tokenizer.PAD_token_id, opt_schedule=None)
        print(f'Dev loss: {dev_loss}, accuracy: {dev_correct}/{dev_total} ({dev_correct / dev_total})')
        test_loss, test_correct, test_total = \
            model.go_epoch(data=test_dl, mask_idx=dec_tokenizer.MASK_token_id,
                           pad_idx=dec_tokenizer.PAD_token_id, opt_schedule=None)
        print(f'Test loss: {test_loss}, accuracy: {test_correct}/{test_total} ({test_correct / test_total})')
        print('\n')
