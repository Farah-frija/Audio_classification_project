import torch


class ESAM(torch.optim.Optimizer):
    """
    Efficient Sharpness-Aware Minimization (ESAM)
    Du et al., 2022 — https://arxiv.org/abs/2203.02714

    Two improvements over vanilla SAM:

    1. Stochastic weight perturbation (rho_param):
       Only a random fraction `rho_param` of parameters are perturbed
       each step instead of all of them. Cuts cost of first_step in half
       while still steering toward flat minima.
       rho_param=0.5 → perturb 50% of weights (recommended)
       rho_param=1.0 → same as vanilla SAM

    2. Hard sample selection (beta):
       The second gradient is computed only on the `beta` fraction of
       the batch with the HIGHEST per-sample loss — the hardest samples.
       Easy normal samples are dropped from the update step entirely.
       On ICBHI this directly helps sensitivity: the optimizer spends
       its second-step budget on the abnormal samples the model is
       currently missing, not on easy normals it already handles well.
       beta=0.5 → top 50% hardest samples (recommended)
       beta=1.0 → all samples (same as vanilla SAM)
    """

    def __init__(self, params, base_optimizer, rho=0.05,
                 rho_param=0.5, beta=0.5, **kwargs):
        assert rho >= 0.0,            f"rho must be >= 0, got {rho}"
        assert 0.0 < rho_param <= 1.0, f"rho_param must be in (0,1], got {rho_param}"
        assert 0.0 < beta <= 1.0,     f"beta must be in (0,1], got {beta}"

        defaults = dict(rho=rho, rho_param=rho_param, beta=beta, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.rho_param      = rho_param
        self.beta           = beta

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save original weights
                self.state[p]["old_p"] = p.data.clone()
                # Stochastic mask: only rho_param fraction of params get perturbed
                self.state[p]["mask"]  = (torch.rand_like(p) < self.rho_param)
                e_w = p.grad * scale.to(p)
                p.add_(e_w * self.state[p]["mask"].float())
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Restore weights before the real update
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm

    @staticmethod
    def select_hard_samples(loss_per_sample, beta):
        """
        Returns a boolean mask of the top `beta` fraction of samples
        ranked by per-sample loss (highest loss = hardest samples).

        loss_per_sample : (B,) tensor from criterion(reduction='none')
        beta            : float in (0, 1]
        """
        if beta >= 1.0:
            return torch.ones(len(loss_per_sample), dtype=torch.bool,
                              device=loss_per_sample.device)
        k    = max(1, int(len(loss_per_sample) * beta))
        topk = torch.topk(loss_per_sample.detach(), k, largest=True).indices
        mask = torch.zeros(len(loss_per_sample), dtype=torch.bool,
                           device=loss_per_sample.device)
        mask[topk] = True
        return mask