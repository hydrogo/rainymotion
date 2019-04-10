# rainymotion

<img src="https://raw.githubusercontent.com/hydrogo/rainymotion/master/docs/notebooks/images/rainymotion_logo.png" alt="rainymotion logo" width="100px"/>


`rainymotion`:
Python library for radar-based precipitation nowcasting based on optical flow techniques

## Idea
The main idea of the `rainymotion` library is to provide an open baseline solution for radar-based precipitation nowcasting.

## Development
`rainymotion` is based only on free and open source software -- we tried to make a clue between the best scientific libraries to make them work together for providing reliable precipitation nowcasts.

<img src="https://raw.githubusercontent.com/hydrogo/rainymotion/master/docs/notebooks/images/rainymotionisbasedonfoss.png" alt="rainymotion logo" width="400px">

## Documentation

`rainymotion` [documentation](http://rainymotion.readthedocs.io) is hosted by [Read the Docs](https://readthedocs.org/).

## Example

To obtain precipitation nowcasts using the `rainymotion` models you need to follow the only three steps:

1. initialize the model (you have 4 variants)
2. load data to the model's placeholder
3. run the model

```python
# import rainymotion model
from rainymotion.models import Dense

# initialize model instance
model = Dense()

# load the data using your custom DataLoader function
model.input_data = DataLoader("/path/to/data")

# run the model
nowcasts = model.run()
```

## Reference

Please cite `rainymotion` as _Ayzel, G., Heistermann, M., and Winterrath, T.: Optical flow models as an open benchmark for radar-based precipitation nowcasting (rainymotion v0.1), Geosci. Model Dev., 12, 1387-1402, https://doi.org/10.5194/gmd-12-1387-2019, 2019._
