'''
pseudo python code to explain aggregating metrics with defaultdict
'''
from collections import defaultdict


metrics = defaultdict(float)

for i, (x, y) in enumerate(dataloader):
    y_ = model(x)
    loss = ...
    metrics['loss'] += float(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 99:
        for k, v in metrics.items():
            writer.add_scalar(k, v / 100)

        metrics.clear()
