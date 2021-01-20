# About single_variable_LR
This project is just use to fitting the linear equation: y = a * x + b (single variable linear regression), and get the value a, b.
Why am I writing this porject?
Cause when you get some data(They're linear,just like y = a * x + b. For sure, maybe there's some interference accompanying them--In my project, it is defined as torch.randn()), you can build a model to figure out the relationship between them.

## Train and visual
```bash
python3 train.py
```
This command will save the weight.pt and draw some parameters' curves (fitting.png and loss.png).
I will finish multi-variables linear regression in the near future and upload it to my github repository (https://github.com/aiologybay).