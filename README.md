# rainymotion

<img src="https://raw.githubusercontent.com/hydrogo/rainymotion/master/docs/source/notebooks/images/rainymotion_logo.png?token=AHVXCrIEMLmGdiBWKkgdJvKaMI3zEixFks5bPKkPwA%3D%3D" alt="rainymotion logo" width="100px"/>


`rainymotion`:
Python library for radar-based precipitation nowcasting based on optical flow techniques

## Idea 
The main idea of the `rainymotion` library is to provide an open baseline solution for radar-based precipitation nowcasting. 

## Development
`rainymotion` is based only on free and open source software -- we tried to make a clue between the best scientific libraries to make them work together for providing reliable precipitation nowcasts.

<img src="https://raw.githubusercontent.com/hydrogo/rainymotion/master/docs/source/notebooks/images/rainymotionisbasedonfoss.png?token=AHVXCi_RMqwkS_B0pbmzBHO3ZtdPN5Iiks5bPLEIwA%3D%3D" alt="rainymotion logo" width="400px">

## Documentation

[correct link to docs](example.com)

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