# Acne Classifier

## Usage

First, install the required dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## Training

To train and save the model:

```bash
python train.py --save

# To run the tests
python train.py --test
```

**NOTE 1** Training is not necessary if you want to use the pre-trained model
included in this repository (acne\_model.h5).

**NOTE 2** This instructions are for a Linux system running the bash shell, you
will need to adjust accordingly.

## Running the interactive predictor

```bash
# To run the predictor on an image file
python predict.py --path someimage.png

# To open a camera an run the interactive predictor
python predict.py --live
```

## Contributing

Here are some ToDo's in case you are looking for something to contribute:

- [ ] Flip the preview in the live predictor
- [ ] Run the predictor on the face ROI instead of the entire image

## License

MIT
