# LangAcq1
A simple language learning environment and agent.

### Environment
One or two objects (card suites) move in the 'scene.'  
Objects have colors (yellow, red, blue, or green).  

The environment outputs:
- the features (shape and color) and locations of objects in the scene
  - a feature is a one-hot vector
  - the location is indicated in a 2D map
- text describing the scene


#### Video

https://github.com/user-attachments/assets/9b58a4d6-f725-445b-993d-06e5e26ad703

## How to Install
* Install [Gymnasium](https://gymnasium.farama.org) and [PyGame](https://www.pygame.org/news) to run the environment (LangAcq1.py and LangAcq1.json).
* Download suite images from [here](https://github.com/rondelion/CursorMover/tree/main/env/28×28).