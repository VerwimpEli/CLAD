<img src="./examples/static/logo.svg" width="80" align="left" alt="CLAD logo">

# &nbsp; CLAD

# A Continual Learning benchmark for Autonomous Driving

Welcome to the official repository for the CLAD benchmark. The goal of CLAD is to introduce a more realistic testing
bed for continual learning. We used [SODA10M](https://soda-2d.github.io/index.html), an industry scale dataset for
autonomous driving to create two benchmarks. CLAD-C is an online classification benchmark with natural, temporal 
correlated and continuous distribution shifts. CLAD-D is a domain incremental continual object detection benchmark.
Below are further details, examples and installation instructions for both benchmarks.

## Installation

CLAD is provided as a python module and depends only on pytorch and torchvision. Optionally you can also use 
[Avalanche](https://avalanche.continualai.org/) and [Detectron2](https://github.com/facebookresearch/detectron2) to 
easily benchmark your own solutions. 

Clone this GitHub repo:
```bash
git clone git@github.com:VerwimpEli/CLAD.git
```
Add the installation directory to your python path. On Linux:
```bash
export PYTHONPATH=$PYTHONPATH:[clad_installation_folder]
```
_(Optional)_ Install the Avalanche master branch. Their pip-module doens't have all functionalities we use.
```bash
pip install git+https://github.com/ContinualAI/avalanche.git
```

_(Optional)_ Install Detectron2, follow the instructions
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for your Pytorch and Cuda installations.

## CLAD-C

### Benchmark introduction

CLAD-C is a classification benchmark for continual learning  from a stream of chronologically ordered images. 
A chronological stream induces continous, yet realistic distribution 
shifts, both in the label distribution and the domain distributions. The image below gives an overview of the 
distribution changes throughout the stream. The x-axis displays the time, along which the images are given.
An example of a distribution shift happens between $T_1$ and $T_2$, which is during the night. If you look at the 
classes that are present during this period, you'll see that there's almost no pedestrians and cyclist left. A similar
thing happens during the other night, or when the car is on the highway. Also, the tricycle is most frequent in 
Guangzhou, not showing up much in the other cities. Beyond this, there are much more frequent but smaller 
distribution shifts not clearly visible in this plot.

<p align="center">
    <img src="./examples/static/3a_collage.png" width="70%" alt="An illustration of the distribution shifts in CLAD-C">
</p>

As an example, these are three subsequent batches if the batch size is set to 10. Note the domination of the cars and
the multiple appearances of the same images from slightly different angles.  

<p align="center">
    <img src="./examples/static/batches_examples.png" width="70%" alt="example batches of CLAD-C">
</p>


### Evaluation
The goal of the challenge is to maximize $AMCA$, or Average Mean Class Accuracy. This is the mean accuracy over all 
classes, averaged at different points during the datastream. We chose this metric because of the high class imbalance 
in the datastream and such that each class is equally important. We calculate this mean accuracy at different points 
during the stream, since the continual learner should be resistent to distributions shifts which isn't tested if you 
only test at the end of the stream. Somewhat arbitrary, we chose the switches between day and night as testing points
(the $T_i$ in the plot above). This is because we noted that at these points naively trainig is most likely to have 
failed. Summarized, the metric we use in this challenge is:

$$
\begin{equation}
AMCA = \frac{1}{T} \sum_{t} \frac{1}{C} \sum_c a_{c, t}
\end{equation}
$$

where $T$ are number of testing points and $C$ is the number of classes.


### Original Challenge Rules

The original challenge at ICCV had some restrictions, which we believe are still worth considering now. Of course, if there's a good reason to deviate from them, there's no reason for not doing so now. Below are the original rules, order by our perceived importance at this point.

1. Maximal replay memory size is 1000 samplesl
2. Maximum batch size is 10
3. No computationally heavy operations are allowed between training and testing (i.e. ideally the model should almost always be directly usable for predictions).
4. Maximum number of parameters are 105% those of a typical Resnet50



## Minimal example
The method ```get_cladc_train``` returns a sequence of training sets (which is actually just one large stream of data), 
and should be trained once, in the returned 
order. After each set, the model should be tested. `get_cladc_val` or `get_cladc_test` returns a single validation 
or test set. For more elaborate examples, both with and without Avalanche, see [here](./examples). 

```python
import clad

import torch
import torchvision.models
from torch.nn import Linear
from torch.utils.data import DataLoader

model = torchvision.models.resnet18(weights=False)
model.fc = Linear(model.fc.in_features, 7, bias=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_sets = clad.get_cladc_train('../../data')

val_set = clad.get_cladc_val('../../data')
val_loader = DataLoader(val_set, batch_size=10)
tester = clad.AMCAtester(val_loader, model)

for t, ts in enumerate(train_sets):
    print(f'Training task {t}')
    loader = DataLoader(ts, batch_size=10, shuffle=False)
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print('testing....')
    tester.evaluate()
    tester.summarize(print_results=True)
```

## Results

To be expected soon, some baseline models and the results of the ICCV '21 challenge on this benchmark. 

## CLAD-Detection

