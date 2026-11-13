import torch


def check_lowmem_gradient(loss_fn):
    from lowmem import objective
    torch.manual_seed(37)
    n = 5
    student_features = torch.randn(n, 6, requires_grad=True)
    logits = torch.randn(n, requires_grad=True)
    teacher_features = torch.randn(n, 6, requires_grad=True)
    labels = torch.tensor([0., 1., 0., 1., 1.])
    paired = torch.tensor([True, False, True, True, False])
    args = dict(alpha=1., beta=.5, temperature=2., loss_type='feature_kl')
    full, _ = loss_fn({'logits': logits, 'distill_features': student_features}, labels,
                      {'distill_features': teacher_features}, paired, **args)
    expected = torch.autograd.grad(full, (logits, student_features))
    logits2 = logits.detach().clone().requires_grad_()
    features2 = student_features.detach().clone().requires_grad_()
    for lo in range(0, n, 2):
        hi = min(n, lo + 2)
        value, _ = objective(
            {'logits': logits2[lo:hi], 'distill_features': features2[lo:hi]},
            labels[lo:hi], {'distill_features': teacher_features[lo:hi]}, paired[lo:hi],
            n, int(paired.sum()), args, loss_fn,
        )
        value.backward()
    assert torch.allclose(expected[0], logits2.grad, atol=1e-7, rtol=1e-5)
    assert torch.allclose(expected[1], features2.grad, atol=1e-7, rtol=1e-5)
    assert teacher_features.grad is None
    print('ACCUMULATION_GRADIENT_CHECK_PASS', flush=True)
