import torch
import torch.nn as nn
from tqdm import tqdm
from libs.hebbian_trainer import HebbianTrainer
import libs.models as models
import libs.hebbian_strategies as strategies
from libs.dataset import generate
from libs.evaluate import test_loops
from statistics import mean
from libs.init import init

# init
RUN_ID, CONFIG, WANDB, device = init()

# dataset
train_loader, test_loader, train_test_loader = generate(
    CONFIG.dataset.batch_size, CONFIG.dataset.num_workers, device
)

# model
mdl = getattr(models, CONFIG.model.name)
model = mdl()
model.to(device)
print(model)

# hebbian trainer init
args = {}
for an, av in CONFIG.hebbian.strategy.arguments.items():
    args[an] = av
trainer = HebbianTrainer(
    model=model,
    strategy=getattr(strategies, CONFIG.hebbian.strategy.name)(**args),
    num_classes=CONFIG.num_classes,
    lr=CONFIG.hebbian.learning_rate,
)

# testing
print('Testing accuracy with initial weights')
test_loops(train_loader=train_test_loader, test_loader=test_loader, model=model)


if CONFIG.hebbian.use is True:
    # hebbian weights init
    print("Initialising weights using Hebbian learning")
    for images, labels in tqdm(train_loader):
        trainer(images, labels)

    # testing
    print('Testing accuracy with hebbian weights')
    test_loops(train_loader=train_test_loader, test_loader=test_loader, model=model)


# running regular gradient descent
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG.backprop.learning_rate)

for i in range(CONFIG.backprop.epochs):
    print(f"├── Backpropagation [{i+1}/{CONFIG.backprop.epochs}]")
    losses = []
    for images, labels in (pbar := tqdm(train_loader)):
        optimizer.zero_grad()
        y = model(images)
        loss = loss_fn(y, labels)
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        pbar.set_description_str(f"├── {mean(losses):.4f}")

# testing
test_loops(train_loader=train_test_loader, test_loader=test_loader, model=model)
