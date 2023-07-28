import torch
from torch.nn import Linear, Conv2d, Sequential
from libs.models import HebbianModels


class HebbianTrainer:
    def __init__(self, model, strategy=None, lr=1e-4, num_classes=10) -> None:
        self.model = model
        self.strategy = strategy
        self.lr = lr
        self.num_classes = num_classes

    def __call__(self, inputs, labels):
        with torch.no_grad():
            x = inputs
            for m in self.model.modules():
                if isinstance(m, (Sequential, HebbianModels)):
                    continue
                elif isinstance(m, Linear):
                    u = x @ m.weight.T
                    # add teached signal to neuron output
                    teacher = torch.ones_like(u)
                    assert labels is not None, "No labels provided for teacher signal"
                    repeats = m.out_features // self.num_classes
                    # if not one-hot - one-hot
                    if len(labels.shape) == 1:
                        labels_onehot = torch.nn.functional.one_hot(
                            labels, num_classes=self.num_classes
                        ).float()
                    labels_onehot = labels_onehot.repeat_interleave(repeats, dim=-1)
                    teacher[:, : labels_onehot.size(-1)] *= labels_onehot
                    u *= teacher
                    uw = u.unsqueeze(1).repeat(
                        1, m.weight.T.size(0), 1
                    ) * m.weight.T.unsqueeze(0)
                    y = self.strategy(u)
                    diffs = self.lr * y.unsqueeze(1) * (x.unsqueeze(-1) - uw)
                    m.weight += diffs.mean(dim=0).T
                elif isinstance(m, Conv2d):
                    x_unf = torch.nn.functional.unfold(
                        input=x,
                        kernel_size=m.kernel_size,
                        dilation=m.dilation,
                        padding=m.padding,
                        stride=m.stride,
                    )
                    wf = m.weight.flatten(1)
                    u = torch.einsum("bij,bik->bjk", x_unf, wf.mT.unsqueeze(0)).unsqueeze(-1)
                    # add teached signal to neuron output
                    teacher = torch.ones_like(u)
                    assert labels is not None, "No labels provided for teacher signal"
                    repeats = m.out_channels // self.num_classes
                    # if not one-hot - one-hot
                    if len(labels.shape) == 1:
                        labels_onehot = torch.nn.functional.one_hot(
                            labels, num_classes=self.num_classes
                        ).float()
                    labels_onehot = (
                        labels_onehot.unsqueeze(-1)
                        .unsqueeze(1)
                        .repeat_interleave(repeats, dim=2)
                        .broadcast_to(teacher.shape)
                    )
                    teacher *= labels_onehot

                    y = self.strategy(u)
                    diffs_all = self.lr * y * (x_unf.mT.unsqueeze(2) - u * wf)
                    diffs = diffs_all.mean(1)
                    m.weight += diffs.mean(dim=0).reshape(m.weight.shape)
                x = m(x)
