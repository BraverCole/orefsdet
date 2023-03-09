# FewX

**OreFSDet** is based on [**FewX**](https://github.com/fanq15/FewX) ( an open source toolbox on top of Detectron2 for data-limited instance-level recognition tasks, e.g.) 

## OreFsdet and baseline on ore dataset
<table >
    <tr align="center">
        <th rowspan="2">Method</th>
        <th colspan="2">5-shot</th>
        <th colspan="2">15-shot</th>
        <th colspan="2">25-shot</th>
    </tr>
    <tr align="center">
        <td>AP</td>
        <td>AP75</td>
        <td>AP</td>
        <td>AP75</td>
        <td>AP</td>
        <td>AP75</td>
    </tr>
    <tr align="center">
        <td>Attentionrpn(baseline)</td>
        <td>25.1</td>
        <td>27.0</td>
        <td>29.2</td>
        <td>34.5</td>
        <td>30.8</td>
        <td>37.0</td>
    </tr>
    <tr align="center">
        <td>orefsdet</td>
        <td>36.2</td>
        <td>33.0</td>
        <td>39.3</td>
        <td>45.6</td>
        <th>44.7</th>
        <th>48.4</th>
    </tr>
</table>

## Results on MS COCO


The model can be obtained frome here <a href="https://drive.google.com/file/d/1VdGVmcufa2JBmZUfwAcDj1OL5tKTFhQ1/view?usp=sharing"> base model</a>&nbsp;\.



## Step 1: Installation
You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.7. I use the Pre-Built Detectron2 with CUDA 10.1 and pytorch 1.7 and you can run this code to install it.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

## Step 2: Prepare dataset
- Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

- `cd datasets`, change the `DATA_ROOT` in the `generate_support_data.sh` to your data path and run `sh generate_support_data.sh`.

``` 
cd FewX/datasets
sh generate_support_data.sh
```

## Step 3: Training and Evaluation

Run `sh all.sh` in the root dir. (This script uses `4 GPUs`. You can change the GPU number. If you use 2 GPUs with unchanged batch size (8), please [halve the learning rate](https://github.com/fanq15/FewX/issues/6#issuecomment-674367388).)C

```
cd FewX
sh all.sh
```


## TODO
 - [ ] Add other dataset results to FSOD.
 - [ ] Add [CPMask](https://arxiv.org/abs/2007.12387) code with partially supervised instance segmentation, fully supervised instance segmentation and few-shot instance segmentation.

## Citing FewX
If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

  ```
  @inproceedings{fan2021fsvod,
    title={Few-Shot Video Object Detection},
    author={Fan, Qi and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={arxiv},
    year={2021}
  }
  @inproceedings{fan2020cpmask,
    title={Commonality-Parsing Network across Shape and Appearance for Partially Supervised Instance Segmentation},
    author={Fan, Qi and Ke, Lei and Pei, Wenjie and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={ECCV},
    year={2020}
  }
  @inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
  }
  ```


