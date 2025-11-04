# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://omnivla-nav.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

### Installation
Please set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

### Inference
1. Download our checkpoints and place them in our directory. "omnivla-original" is the trained checkpoints of the OmniVLA for paper submission. "omnivla-original-balance" contains the trained checkpoints of OmniVLA that account for the data balance in the LeLaN dataset. And "omnivla-finetuned-cast" is finetuned checkpoints with the [CAST](https://huggingface.co/datasets/catglossop/CAST-dataset) dataset.
    ```
    git clone https://huggingface.co/NHirose/omnivla-original
    git clone https://huggingface.co/NHirose/omnivla-original-balance    
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
We provide the training code along with a sample dataloader to help you quickly understand the required data loading structure. Since preparing the full training dataset is resource-intensive, we include this simplified code base for convenience.

1. Downloading MBRA project code base:
    ```
    cd ..
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. Downloading MBRA model:
    ```
    cd OmniVLA_internal
    git clone https://huggingface.co/NHirose/MBRA/
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
7. Memo finetuning our OmniVLA on our large navigation dataset:
    ```
    conda activate omnivla_2
    cd /media/noriaki/Noriaki_Data/OmniVLA
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "noriaki-hirose"   --wandb_project "omnivla"
    ```

### Training with GNM, LeLaN, Frodobots, BDD and CAST datasets
We provide training code that supports multiple public datasets. Before following the full training process, please first ensure that you can run the example training with the sample dataloader.

1. Downloading all datasets from the original website. ([GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://github.com/NHirose/learning-language-navigation), [Frodobots](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA), [CAST](https://openvla-oft.github.io/)) Please verify that the downloaded datasets work properly in their original codebase, except BDD dataset.

2. Downloading the modified BDD dataset with MBRA annotations from [here](https://huggingface.co/datasets/NHirose/BDD_OmniVLA) and extract it. The image sequences in the modified dataset remain subject to the [original BDD license](http://bdd-data.berkeley.edu/download.html), while the additional MBRA annotations are released under the MIT license.

3. Downloading the lerobot code base for the Frodobots dataset dataloader:
    ```
    git clone https://github.com/huggingface/lerobot.git 
    ```
4. Edit the data path in config_nav/mbra_and_dataset_config.yaml:

5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "X"   --wandb_project "omnivla"
    ```
       
In our training setup, we use 8 Nvidia H100 GPUs (80 GB each) across 8 nodes. The batch sizes are configured as [LeLaN, GNM, Frodobots, BDD] = [4, 1, 1, 1], with gradient accumulation set to 4 steps. When finetuning with CAST dataset, we set the batch size as [LeLaN, CAST, GNM, Frodobots, BDD] = [2, 2, 1, 1, 1]. To do so, you need to directly edit train_omnivla_dataset.py.
    
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
