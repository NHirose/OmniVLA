# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://omnivla-nav.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

### Installation
Please set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

### Inference
1. Download our checkpoints and place them in our directory. "omnivla-original" is the trained checkpoints of the OmniVLA. And "omnivla-finetuned-cast" is finetuned checkpoints with the [CAST](https://huggingface.co/datasets/catglossop/CAST-dataset) dataset.
    ```
    git clone https://huggingface.co/NHirose/omnivla-original
    git clone https://huggingface.co/NHirose/omnivla-finetuned-cast
    ```
2. Run OmniVLA using a sample current image, goal images, GPS pose, and language prompt. You can view the generated trajectory in the output figure 1_ex.jpg.
    ```
    cd inference
    python run_omnivla.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 560. 
    
4. Run OmniVLA to control the real robot. Modify "run_omnivla.py" to update the robot’s state (camera image, GPS signal) and adjust the goal information accordingly. Then, feed the generated velocity commands to your robot.

5. To try the finetuned checkpoints with the CAST dataset, update the path and step number in "InferenceConfig" within "run_omnivla.py".

### Training
We will release our training code for OmniVLA soon!!

### Acknowledgement
We implement our ideas and design choices on top of the pretrained checkpoints. Our work builds upon the [OpenVLA-OFT](https://openvla-oft.github.io/) codebase, with additional code added to create OmniVLA. As such, our implementation leverages many components of the OpenVLA-OFT codebase. We sincerely appreciate the effort and contributions of the OpenVLA-OFT team!

## Citing
```
@misc{hirose2025omnivla,
      title={OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation}, 
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2025},
      eprint={2509.19480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19480}, 
}
