# Why Don’t Prompt-Based Fairness Metrics Correlate? (ACL main 2024)

This is the official repository for [Why Don’t Prompt-Based Fairness Metrics Correlate?](https://arxiv.org/abs/2307.16704), accepted at ACL main 2024. Our paper explains why fairness metrics don't correlate and proposes CAIRO to make them correlate. Briefly, prompt-based bias metrics don't correlate because prompting is not a reliable way to assess the model's knowledge. In addition, metrics differ in how they define and quantify bias. For example, according to one metric, race bias refers to the deviation in the model's toxicity when prompted with sentences about black and white people, while another metric could measure the difference in the model's sentiment when prompted with sentences about Asian and Middle Eastern people.

<div style="text-align: center">
<img src="CAIRO.png" width="400">
<p style="text-align: center;">  </p>
</div>

## Usage
Please follow the instructions in our [tutorial](https://colab.research.google.com/drive/1wUJhuPR1PKu-BcxP2Lx_9dfzffCQ3-kE?usp=sharing).

## Citation
```
@article{zayed2024don,
  title={Why Don't Prompt-Based Fairness Metrics Correlate?},
  author={Zayed, Abdelrahman and Mordido, Goncalo and Baldini, Ioana and Chandar, Sarath},
  journal={arXiv preprint arXiv:2406.05918},
  year={2024}
}
```
