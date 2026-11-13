"""Construct final task-specific students and matched branch ablations."""
import copy
import torch


def natural_parent(task, full_config, seed, MITS):
    from lowmem import factory as lowmem_factory
    from gate_factory import factory as gated_factory
    Base = lowmem_factory(MITS, full_config, seed)
    return gated_factory(Base) if task == 'reentrancy' else Base


def _source_key(candidate_key, retained, full_branches):
    # Reentrancy uses GatedFusion. Map candidate gate modules back to the
    # corresponding branch index in the complete task-specific model.
    for prefix in ('fusion.norms.', 'fusion.scorers.'):
        if candidate_key.startswith(prefix):
            rest = candidate_key[len(prefix):]
            index_text, suffix = rest.split('.', 1)
            new_index = int(index_text)
            old_index = full_branches.index(retained[new_index])
            return f'{prefix}{old_index}.{suffix}'
    return candidate_key


def matched_ablation_parent(task, full_config, retained, seed, MITS):
    """Return a model class initialized as an exact submodel of the final student."""
    FullParent = natural_parent(task, full_config, seed, MITS)
    reduced_config = copy.deepcopy(full_config)
    reduced_config['branches'] = list(retained)
    CandidateParent = natural_parent(task, reduced_config, seed, MITS)
    full_branches = list(full_config['branches'])
    dim = int(full_config['dim'])
    selected_columns = []
    for branch in retained:
        old = full_branches.index(branch)
        selected_columns.extend(range(old * dim, (old + 1) * dim))

    class MatchedAblation(CandidateParent):
        def __init__(self, config, role='student', teacher_type='multimodal'):
            super().__init__(config, role, teacher_type)
            if role != 'student':
                return
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                reference = FullParent(full_config, 'student', teacher_type)
            source = reference.state_dict()
            target = self.state_dict()
            for name, value in target.items():
                source_name = _source_key(name, retained, full_branches)
                if source_name not in source:
                    raise ValueError('Missing matched source parameter: ' + source_name)
                expected = source[source_name]
                if name in ('fusion.0.weight', 'fusion.base.0.weight'):
                    expected = expected[:, selected_columns]
                if expected.shape != value.shape:
                    raise ValueError(f'Unexpected shape for {name}: {expected.shape} versus {value.shape}')
                value.copy_(expected)
            self.load_state_dict(target, strict=True)
            self.final_ablation_audit = {
                'full_branches': full_branches,
                'retained_branches': list(retained),
                'retained_initial_parameters_identical': True,
                'fusion_columns_retained_by_branch_identity': True,
            }

    return MatchedAblation


def state_hash(model):
    import hashlib
    return hashlib.sha256(b''.join(
        value.detach().cpu().numpy().tobytes()
        for value in model.state_dict().values()
    )).hexdigest()

