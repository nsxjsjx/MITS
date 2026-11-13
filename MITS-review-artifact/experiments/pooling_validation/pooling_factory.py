"""Replace only the student node aggregator with a non-parametric pooling rule."""
import torch
from torch import nn


KINDS = ('mean', 'max', 'median', 'sum')


def masked_pool(x, mask, kind):
    """Pool valid tokens only; median averages the two middle values."""
    lengths = mask.sum(1)
    if (lengths == 0).any():
        raise ValueError('All-PAD node')
    if kind == 'max':
        return x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(1).values
    total = x.masked_fill(~mask.unsqueeze(-1), 0).sum(1)
    if kind == 'sum':
        return total
    if kind == 'mean':
        return total / lengths.unsqueeze(-1)
    if kind == 'median':
        ordered = x.masked_fill(~mask.unsqueeze(-1), float('inf')).sort(dim=1).values
        batch = torch.arange(x.size(0), device=x.device)
        return (ordered[batch, (lengths - 1) // 2] + ordered[batch, lengths // 2]) / 2
    raise ValueError(kind)


def factory(Parent, kind):
    if kind not in KINDS:
        raise ValueError(kind)

    class PoolingStudent(Parent):
        def __init__(self, config, role='student', teacher_type='multimodal'):
            super().__init__(config, role, teacher_type)
            if role != 'student':
                return
            if not hasattr(self, 'node'):
                raise ValueError('Pooling control requires the graph branch')
            # Preserve the embedding initialized by the final model and remove
            # only the selected trainable intra-node encoder.
            self.node.kind = kind
            self.node.net = nn.Identity()
            node = self.node
            def pooled_forward(tokens):
                outputs = []
                for lo in range(0, len(tokens), 64):
                    piece = tokens[lo:lo + 64]
                    outputs.append(masked_pool(
                        node.drop(torch.relu(node.embedding(piece))), piece.ne(0), kind
                    ))
                return torch.cat(outputs, dim=0)
            node.forward = pooled_forward
            self.pooling_control_audit = {
                'kind': kind,
                'student_only': True,
                'padding_excluded': True,
                'embedding_initialization_preserved': True,
                'trainable_node_encoder_removed': True,
            }

    return PoolingStudent


def self_check():
    x = torch.tensor([[[1., 4.], [3., 2.], [99., 99.]],
                      [[2., 8.], [4., 6.], [6., 4.]]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = {
        'mean': torch.tensor([[2., 3.], [4., 6.]]),
        'max': torch.tensor([[3., 4.], [6., 8.]]),
        'median': torch.tensor([[2., 3.], [4., 6.]]),
        'sum': torch.tensor([[4., 6.], [12., 18.]]),
    }
    for kind, wanted in expected.items():
        got = masked_pool(x, mask, kind)
        if not torch.equal(got, wanted):
            raise AssertionError(f'{kind} pooling mismatch: {got} != {wanted}')

