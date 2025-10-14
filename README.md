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
    python inference/run_omnivla.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 560. 
    
4. Run OmniVLA to control the real robot. Modify "run_omnivla.py" to update the robot’s state (camera image, GPS signal) and adjust the goal information accordingly. Then, feed the generated velocity commands to your robot.

5. To try the finetuned checkpoints with the CAST dataset, update the path and step number in "InferenceConfig" within "run_omnivla.py".

### Training
1. Downloading MBRA project code base:
    ```
    cd ..
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. Downloading MBRA model (MBRA: original MBRA model, MBRA_BDD: MBRA model for BDD dataset. You can switch the MBRA model at line 743--744 in vla-scripts/train_omnivla.py.):
    ```
    cd OmniVLA_internal
    git clone https://huggingface.co/NHirose/MBRA/
    git clone https://huggingface.co/NHirose/MBRA_BDD
    ```
3. You can set the training or debugging mode at line 10 in vla-scripts/train_omnivla.py. Note that even in debugging mode, the code requires at least 20 GB of GPU memory (we use an NVIDIA RTX 4090).

4. You can configure visualization at line 11 in vla-scripts/train_omnivla.py. During training, it should be set to False.
    
5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path openvla/openvla-7b --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ```
6. Finetuning our OmniVLA (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla.py  --vla_path ./omnivla-original --dataset_name omnivla --num_images_in_input 2 --batch_size X --wandb_entity "X" --wandb_project "omnivla"
    ````
In our training setup, we use 8 Nvidia H100 GPUs (80 GB each) across 8 nodes. The batch sizes are configured as [LeLaN, GNM, Frodobots, BDD] = [4, 1, 1, 1], with gradient accumulation set to 4 steps.
    
### Dataset

Our codebase provides a simple dataloader with sample images in prismatic/vla/datasets/dummy_dataset.py. For OmniVLA, you need to provide the following inputs: the current egocentric image, the egocentric goal image, the language prompt, the modality ID, a sequence of action commands, an action selection mask, and the goal pose in the robot’s local coordinate frame. If your dataset is missing one or two modalities among the egocentric goal image, goal pose, or language prompt, you can provide dummy values for the missing inputs. However, make sure not to select a modality ID that includes the unavailable modality. Please refer to the sample code for details on the data conversion preprocess.

In addition to the OmniVLA inputs, you can optionally include the history of egocentric images and a downsampled egocentric goal image for action reannotation using the MBRA model. If you choose not to use the reannotated action commands, you can provide dummy values and set the action selection mask to 0.0, or alternatively, modify the code as needed.
    
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
