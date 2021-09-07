# ETS2 Trainer

Pytorch implementation of [Densedepth](https://github.com/ialhashim/DenseDepth) to be trained with the ETS2 Dataset. 
Part of my degree in videogames design and development final project. All my thanks to [Ibraheem Alhashim](https://github.com/ialhashim)
for his fantastic work and for releasing it open to help others like me...

Most of the code is inspired by the fantastic work of [Monodepth2](https://github.com/nianticlabs/monodepth2), whose
owners and developers I would like to thank too.

> :warning: **Highly experimental code**: This code is not meant to be used in production systems.

## Training

Download the dataset from [ETS2 Dataset](ets2-dataset.dmariaa.es). You can transform the .bmp files to .jpg to 
downsize them; the actual dataset after files transformed weights about 115GB.

Clone this repository and use venv or virtualenv to build a python environment and avoid polluting others in the 
same machine. Install the requirements from requirements.txt and run train.py.

The training I used for my project was:

```
> train.py --model-name ets2.clip.depthnorm --cuda-device 0 --resume-training --num-workers 4 --epochs 9 --batch-size 7 --log-frequency 100
```

On a Nvidia TITAN X with 11GB the ram, it took about 75 hours to train.

## Evaluation 

The evaluation code, very similar to *Monodepth2* evaluation code, runs on kitty depth dataset. You can get and 
prepare the dataset as instructed [here](https://github.com/nianticlabs/monodepth2#-kitti-evaluation) and run evaluation.py to get
some decent results.