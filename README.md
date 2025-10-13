# LangAcq1
A simple language learning environment and agent.  
For detailed explanation, see the blog article (to be published in December 2025).

### Environment

Code: `LangAcq1Env.py`

One or two objects (card suites) move in the 'scene.'  
Objects have colors (yellow, red, blue, or green).  

The environment outputs:
- the features (shape and color) and locations of objects in the scene
  - a feature is a one-hot vector
  - the location is indicated in a 2D map
- text describing the scene (in [Interlingua](https://en.wikipedia.org/wiki/Interlingua))


#### Video

https://github.com/user-attachments/assets/9b52f468-5424-47da-b2d7-98abac1cfc59

### Agent

Code: `LangAcq1a.py`

It learns to generate text describing the scene from the environment.  
It uses saccades (attention shifts) to obtain features from objects in the scene.

Pre-training:
- CBOW embedding with POSClassifier from [NeuralParser](https://github.com/rondelion/NeuralParser).
- Learning associations of shape/color words with corresponding features with one object setting of the environment. 

Training:
- Saccade (attention shift)
- Language model


## How to Install
### Environment
* Install [Gymnasium](https://gymnasium.farama.org) and [PyGame](https://www.pygame.org/news) to run the environment (LangAcq1.py and LangAcq1.json).
* Download suite images from [here](https://github.com/rondelion/CursorMover/tree/main/env/28×28).

### Agent
* Install PyTorch.
* Clone [NeuralParser](https://github.com/rondelion/NeuralParser) for pre-training.
* Clone [AEPredictor](https://github.com/rondelion/AEPredictor) as the main learning tool.
  * uses simple_autoencoder.py from [cerenaut-pt-core](https://github.com/Cerenaut/cerenaut-pt-core).