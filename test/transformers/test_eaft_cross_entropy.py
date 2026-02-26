"""
Tests for Entropy-Adaptive Fine-Tuning (EAFT) cross entropy loss integration
in both standalone and fused linear cross entropy.

EAFT scales each token's loss by an adaptive weight based on top-k entropy
of the logits distribution:
    weight = (entropy_approx / 3.0) ^ alpha
where entropy_approx is computed from the top-k logits.
"""

import pytest
import torch
import torch.nn.functional as F
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose, set_seed


device = infer_device()


# ──────────────────────────────────────────────────────────────────────
# Reference (pure PyTorch) implementation of EAFT cross entropy
# ──────────────────────────────────────────────────────────────────────


class TorchEAFTCrossEntropy(torch.nn.Module):
    """Reference PyTorch implementation of EAFT cross entropy."""

    def __init__(
        self,
        ignore_index: int = -100,
        alpha: float = 1.0,
        topk: int = 20,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.topk = topk
        self.reduction = reduction

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_f32 = source.float()
        per_token_loss = F.cross_entropy(source_f32, target, ignore_index=self.ignore_index, reduction="none")

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=source.device, dtype=source.dtype)

        with torch.no_grad():
            V = source_f32.shape[-1]
            k = min(self.topk, V)
            topk_val, _ = torch.topk(source_f32, k=k, dim=-1)
            logsumexp_topk = torch.logsumexp(topk_val, dim=-1, keepdim=True)
            log_probs_topk = topk_val - logsumexp_topk
            probs_topk = torch.exp(log_probs_topk)
            entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
            entropy_term = entropy_approx / 3.0
            adaptive_weight = torch.pow(entropy_term, self.alpha)
            adaptive_weight = torch.where(valid_mask, adaptive_weight, torch.zeros_like(adaptive_weight))

        weighted_losses = per_token_loss * adaptive_weight

        if self.reduction == "mean":
            loss = weighted_losses.sum() / valid_mask.sum()
        elif self.reduction == "sum":
            loss = weighted_losses.sum()
        else:
            loss = weighted_losses

        return loss


class TorchEAFTFusedLinearCE(torch.nn.Module):
    """Reference LM head + EAFT CE for comparison with fused version."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        alpha: float = 1.0,
        topk: int = 20,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.eaft_ce = TorchEAFTCrossEntropy(
            ignore_index=ignore_index,
            alpha=alpha,
            topk=topk,
            reduction=reduction,
        )

    def forward(self, x, y):
        logits = self.lin(x).float()
        return self.eaft_ce(logits, y)


class LigerEAFTFusedLinearCE(torch.nn.Module):
    """Liger fused linear + EAFT CE."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        alpha: float = 1.0,
        topk: int = 20,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            use_eaft=True,
            eaft_alpha=alpha,
            eaft_topk=topk,
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y, self.lin.bias)


# ──────────────────────────────────────────────────────────────────────
# Test 1: Standalone EAFT Cross Entropy (forward only, loss correctness)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 64, 128),
        (4, 128, 256),
        (1, 32, 512),
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("topk", [10, 20])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_eaft_cross_entropy_correctness(B, T, V, alpha, topk, dtype):
    """Compare Liger EAFT CE against the pure-PyTorch reference."""
    set_seed(42)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    # Reference
    ref_ce = TorchEAFTCrossEntropy(alpha=alpha, topk=topk, reduction="mean")
    ref_loss = ref_ce(_input1, target)

    # Liger
    liger_loss, _, _, _ = LigerCrossEntropyFunction.apply(
        _input2,
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        True,  # use_eaft
        alpha,  # eaft_alpha
        topk,  # eaft_topk
    )

    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 2: Standalone EAFT CE with gradient correctness
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 64, 128),
        (4, 128, 256),
    ],
)
@pytest.mark.parametrize("alpha", [1.0])
def test_eaft_cross_entropy_gradient(B, T, V, alpha):
    """Verify gradients of Liger EAFT CE vs reference."""
    set_seed(42)

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.float32)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    ref_ce = TorchEAFTCrossEntropy(alpha=alpha, reduction="mean")
    ref_loss = ref_ce(_input1, target)
    ref_loss.backward()

    liger_loss, _, _, _ = LigerCrossEntropyFunction.apply(
        _input2,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        False,
        False,
        True,
        alpha,
        20,
    )
    liger_loss.backward()

    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)
    assert_verbose_allclose(_input2.grad, _input1.grad, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 3: EAFT with ignore_index
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 64, 128),
    ],
)
def test_eaft_cross_entropy_with_ignore_index(B, T, V):
    """EAFT CE correctly handles ignore_index tokens."""
    set_seed(42)

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.float32)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    # Assign ~30% tokens as ignore_index
    num_ignore = B * T // 3
    indices = torch.randperm(B * T)[:num_ignore]
    target[indices] = -100

    ref_ce = TorchEAFTCrossEntropy(alpha=1.0, reduction="mean")
    ref_loss = ref_ce(_input1, target)
    ref_loss.backward()

    liger_loss, _, _, _ = LigerCrossEntropyFunction.apply(
        _input2,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        False,
        False,
        True,
        1.0,
        20,
    )
    liger_loss.backward()

    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)
    assert_verbose_allclose(_input2.grad, _input1.grad, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 4: LigerCrossEntropyLoss module with EAFT enabled
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_eaft_cross_entropy_module(alpha):
    """LigerCrossEntropyLoss(use_eaft=True) matches reference."""
    set_seed(42)
    B, T, V = 2, 64, 128

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.float32)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    ref_ce = TorchEAFTCrossEntropy(alpha=alpha, reduction="mean")
    ref_loss = ref_ce(_input1, target)
    ref_loss.backward()

    liger_ce = LigerCrossEntropyLoss(use_eaft=True, eaft_alpha=alpha)
    liger_loss = liger_ce(_input2, target)
    liger_loss.backward()

    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)
    assert_verbose_allclose(_input2.grad, _input1.grad, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 5: Fused Linear EAFT CE (forward + backward)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (4, 64, 128, 256),
        (2, 128, 256, 512),
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 1.0])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-3),
    ],
)
def test_eaft_fused_linear_cross_entropy(B, T, H, V, alpha, dtype, atol, rtol):
    """Compare Liger fused linear EAFT CE against reference (unfused) implementation."""
    set_seed(42)

    torch_model = TorchEAFTFusedLinearCE(H=H, V=V, dtype=dtype, alpha=alpha).to(device)
    liger_model = LigerEAFTFusedLinearCE(H=H, V=V, dtype=dtype, alpha=alpha).to(device)

    # Share weights
    liger_model.lin.weight.data = torch_model.lin.weight.data.clone()
    if torch_model.lin.bias is not None:
        liger_model.lin.bias.data = torch_model.lin.bias.data.clone()

    _input = torch.randn(B * T, H, device=device, dtype=dtype)
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    ref_loss = torch_model(input1, target)
    liger_loss = liger_model(input2, target)

    assert_verbose_allclose(liger_loss, ref_loss, atol=atol, rtol=rtol)

    ref_loss.backward()
    liger_loss.backward()

    assert_verbose_allclose(input2.grad, input1.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_model.lin.weight.grad, torch_model.lin.weight.grad, atol=atol, rtol=rtol)


# ──────────────────────────────────────────────────────────────────────
# Test 6: Fused Linear EAFT CE with ignore_index
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (4, 64, 128, 256),
    ],
)
def test_eaft_fused_linear_cross_entropy_with_ignore(B, T, H, V):
    """Fused linear EAFT CE correctly handles ignore_index."""
    set_seed(42)
    dtype = torch.float32
    alpha = 1.0

    torch_model = TorchEAFTFusedLinearCE(H=H, V=V, dtype=dtype, alpha=alpha).to(device)
    liger_model = LigerEAFTFusedLinearCE(H=H, V=V, dtype=dtype, alpha=alpha).to(device)
    liger_model.lin.weight.data = torch_model.lin.weight.data.clone()

    _input = torch.randn(B * T, H, device=device, dtype=dtype)
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    # Assign some tokens as ignore_index
    num_ignore = B * T // 3
    indices = torch.randperm(B * T)[:num_ignore]
    target[indices] = -100

    ref_loss = torch_model(input1, target)
    liger_loss = liger_model(input2, target)

    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)

    ref_loss.backward()
    liger_loss.backward()

    assert_verbose_allclose(input2.grad, input1.grad, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 7: EAFT is a no-op when alpha=0 (weight becomes 1.0 for all tokens)
# ──────────────────────────────────────────────────────────────────────


def test_eaft_alpha_zero_is_noop():
    """When alpha=0, EAFT weight is 1.0 for all tokens, effectively standard CE (up to normalization)."""
    set_seed(42)
    B, T, V = 2, 64, 128

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.float32)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    # Reference: EAFT with alpha=0 means weight = (entropy/3)^0 = 1.0
    # So it should reduce to normal CE (mean of per-token losses)
    ref_ce = TorchEAFTCrossEntropy(alpha=0.0, reduction="mean")
    ref_loss = ref_ce(_input1, target)

    liger_ce = LigerCrossEntropyLoss(use_eaft=True, eaft_alpha=0.0)
    liger_loss = liger_ce(_input2, target)

    # Both should match standard CE within tolerance
    assert_verbose_allclose(liger_loss, ref_loss, atol=1e-4, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Test 8: EAFT + token_scaling mutual exclusion
# ──────────────────────────────────────────────────────────────────────


def test_eaft_and_token_scaling_mutual_exclusion():
    """use_eaft and use_token_scaling cannot both be True."""
    with pytest.raises(AssertionError):
        LigerFusedLinearCrossEntropyLoss(use_eaft=True, use_token_scaling=True)


# ──────────────────────────────────────────────────────────────────────
# Test 9: EAFT with bfloat16
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha", [1.0])
def test_eaft_cross_entropy_bfloat16(alpha):
    """EAFT CE works with bfloat16 inputs."""
    set_seed(42)
    B, T, V = 2, 64, 128
    dtype = torch.bfloat16

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype)
    _input1 = _tensor.detach().clone().float().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    ref_ce = TorchEAFTCrossEntropy(alpha=alpha, reduction="mean")
    ref_loss = ref_ce(_input1, target)

    liger_ce = LigerCrossEntropyLoss(use_eaft=True, eaft_alpha=alpha)
    liger_loss = liger_ce(_input2, target)

    # Wider tolerance for bfloat16
    assert_verbose_allclose(liger_loss, ref_loss, atol=5e-2, rtol=5e-2)


# ──────────────────────────────────────────────────────────────────────
# Test 10: EAFT changes the loss (sanity check)
# ──────────────────────────────────────────────────────────────────────


def test_eaft_versus_standard_ce_differs():
    """EAFT loss should generally differ from standard CE loss."""
    set_seed(42)
    B, T, V = 2, 64, 128

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.float32)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device)

    # Standard CE
    standard_ce = LigerCrossEntropyLoss()
    standard_loss = standard_ce(_input1, target)

    # EAFT CE
    eaft_ce = LigerCrossEntropyLoss(use_eaft=True, eaft_alpha=1.0)
    eaft_loss = eaft_ce(_input2, target)

    # They should be different (with very high probability for random data)
    assert not torch.allclose(standard_loss, eaft_loss, atol=1e-6), (
        "EAFT loss should differ from standard CE loss for random data"
    )
