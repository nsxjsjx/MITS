import torch
from torch import nn


class GatedFusion(nn.Module):
    """Per-sample branch gates followed by the unchanged concat projection.

    Every gate starts at 1/B after softmax and is multiplied by B, so the first
    forward pass is exactly the ordinary concatenation baseline. The gate adds
    only one LayerNorm and one scalar scorer per branch.
    """

    def __init__(self, base_fusion, branches, dim):
        super().__init__()
        self.base = base_fusion
        self.branches = tuple(branches)
        self.dim = int(dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in branches])
        self.scorers = nn.ModuleList([nn.Linear(dim, 1) for _ in branches])
        for scorer in self.scorers:
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)
        self.last_weights = None

    def forward(self, concatenated):
        expected = len(self.branches) * self.dim
        if concatenated.ndim != 2 or concatenated.size(-1) != expected:
            raise ValueError(f'Expected [batch,{expected}] concatenated branches')
        pieces = list(concatenated.split(self.dim, dim=-1))
        scores = torch.cat(
            [score(norm(piece)) for piece, norm, score in zip(pieces, self.norms, self.scorers)],
            dim=-1,
        )
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        scale = float(len(pieces))
        gated = torch.cat([piece * weights[:, i:i + 1] * scale for i, piece in enumerate(pieces)], dim=-1)
        return self.base(gated)


def factory(Parent):
    """Wrap only the student fusion; teacher construction remains unchanged."""

    class GatedStudent(Parent):
        def __init__(self, config, role='student', teacher_type='multimodal'):
            super().__init__(config, role, teacher_type)
            if role != 'student':
                return
            d = config['dim']
            original = self.fusion
            self.fusion = GatedFusion(original, self.branches, d)
            added = sum(p.numel() for p in self.fusion.norms.parameters()) + sum(
                p.numel() for p in self.fusion.scorers.parameters()
            )
            self.gated_fusion_audit = {
                'branches': list(self.branches),
                'uniform_identity_initialization': True,
                'gate_definition': 'softmax scalar scores from per-branch LayerNorm features',
                'gate_scale': len(self.branches),
                'added_parameters': added,
                'base_concat_projection_unchanged': True,
            }

        def forward(self, graph, tokens, source=None, return_branches=False):
            result = super().forward(graph, tokens, source, return_branches)
            if return_branches and self.role == 'student':
                result['gate_weights'] = self.fusion.last_weights
            return result

    return GatedStudent


def self_check(Parent, config, teacher_type='multimodal'):
    """Check identity initialization, gradients, parameter budget, and teacher isolation."""
    from types import SimpleNamespace

    seed = 17
    Gated = factory(Parent)
    torch.manual_seed(seed)
    baseline = Parent(config, 'student', teacher_type).eval()
    torch.manual_seed(seed)
    gated = Gated(config, 'student', teacher_type).eval()

    baseline_state = baseline.state_dict()
    gated_state = gated.state_dict()
    for name, value in baseline_state.items():
        mapped = 'fusion.base.' + name[len('fusion.'):] if name.startswith('fusion.') else name
        if mapped not in gated_state or not torch.equal(value, gated_state[mapped]):
            raise AssertionError('Shared initialization differs: ' + name)

    x = torch.tensor([[1, 2, 3, 0], [1, 3, 4, 5]])
    graph = SimpleNamespace(
        x=x,
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch=torch.tensor([0, 0]),
    )
    tokens = torch.tensor([[1, 2, 3, 4, 0, 0]])
    with torch.no_grad():
        a = baseline(graph, tokens)
        b = gated(graph, tokens, return_branches=True)
    if not torch.allclose(a['logits'], b['logits'], atol=1e-7, rtol=1e-6):
        raise AssertionError('Uniform gate must reproduce baseline logits at initialization')
    if not torch.allclose(a['distill_features'], b['distill_features'], atol=1e-7, rtol=1e-6):
        raise AssertionError('Uniform gate must reproduce baseline features at initialization')
    expected = torch.full_like(b['gate_weights'], 1.0 / len(config['branches']))
    if not torch.allclose(b['gate_weights'], expected, atol=1e-7, rtol=0):
        raise AssertionError('Gates are not initially uniform')

    gated.train()
    out = gated(graph, tokens)
    out['logits'].sum().backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in gated.fusion.scorers.parameters()):
        raise AssertionError('Gate scorers receive no gradient')

    base_n = sum(p.numel() for p in baseline.parameters())
    gated_n = sum(p.numel() for p in gated.parameters())
    if gated_n <= base_n or (gated_n - base_n) / base_n >= 0.002:
        raise AssertionError('Gate parameter increase is not lightweight')

    torch.manual_seed(seed)
    teacher_a = Parent(config, 'teacher', teacher_type)
    torch.manual_seed(seed)
    teacher_b = Gated(config, 'teacher', teacher_type)
    if any(not torch.equal(v, teacher_b.state_dict()[k]) for k, v in teacher_a.state_dict().items()):
        raise AssertionError('Teacher construction changed')
    print('GATE_SELF_CHECK_PASS: identity start; trainable gates; <0.2% parameters; teacher unchanged', flush=True)
