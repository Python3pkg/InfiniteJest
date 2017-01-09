# InfiniteJest
An package for integrating data envelopment analysis in to agent based models

## Example DEA Workflow

```python
>> import pandas as pd
>> from InfiniteJest.dea import DEA

>> df = pd.read_csv('my_data.csv')
>> dea = DEA(inputs = df[['x1','x2']], outputs = df[['y1,y2']], model = 'ccr')
>> dea.solve()
>> dea.eff_scores
[0.85714286, 1.0, 1.0, 1.0, 0.666667, 1.0, 1.0]
```

