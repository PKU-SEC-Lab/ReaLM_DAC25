# ReaLM: Reliable and Efficient Large Language Model Inference with Statistical Algorithm-Based Fault Tolerance

ReaLM (Accepted by DAC'25) is a novel algorithm/circuit co-design framework that enables reliable and energy-efficient LLM inference by leveraging inherent model resilience and statistical ABFT techniques. This repository contains code for LLM resilience characterization. Our paper can be found [here](https://arxiv.org/abs/2503.24053).

## Abstract

The demand for efficient large language model (LLM) inference has propelled the development of dedicated accelerators. As accelerators are vulnerable to hardware faults due to aging, variation, etc, existing accelerator designs often reserve a large voltage margin or leverage algorithm-based fault tolerance (ABFT) techniques to ensure LLM inference correctness. However, previous methods often overlook the inherent fault tolerance of LLMs, leading to high computation and energy overhead. To enable reliable yet efficient LLM inference, in this paper, we propose a novel algorithm/circuit co-design framework, dubbed ReaLM. For the first time, we systematically characterize the fault tolerance of LLMs by performing a large scale error injection study of representative LLMs and natural language understanding tasks. Then, we propose a statistical ABFT algorithm that fully leverages the error robustness to minimize error recovery as much as possible. We also customize the error detection circuits to enable a low-cost online collection of error statistics. Extensive experiments show that with only 1.42% circuit area and 1.79% power overhead, our ReaLM can reduce perplexity degradation from 18.54 to 0.29. Compared to existing methods, ReaLM consistently reduces recovery costs across different operating voltages and improves energy efficiency by up to 35.83% without compromising LLM performance.
![resilience_characterization](/figs/resilience_characterization.png)

## Installation

### Environment Setup

```python
conda create -n realm python=3.8 -y
conda activate realm

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
--extra-index-url https://download.pytorch.org/whl/cu113

pip install transformers==4.36.0 accelerate datasets zstandard

python setup.py install
```

### Alternative

We also provide the environment file

```
conda env create -f environment.yml

python setup.py install
```


## Quick Start

Our project provides straightforward error resilience characterization through configurable error injection. The core implementation resides in the `./src` directory, containing:

- Preconfigured python files for **multiple LLMs** (opt-1.3b, Mistral-7b, Llama-2-7b, Llama-3-8b, etc.) and **datasets** (LAMBADA, WikiText2, HellaSwag, X-Sum, GSM8K, etc.)
- Flexible error injection parameters:
  - **Bit Error Rate (BER)**: Configure via `err_prob_list`
  - **Target layers**: Specify within `quantize_model_error`
  - **Network components**: Choose network components such as `QKT` and `Down` by configuring `NoisyW8A8BMM` and `NoisyW8A8Linear`

**Example**: To evaluate Llama-3-8b's resilience on HellaSwag:

```
cd src
CUDA_VISIBLE_DEVICES=0 python main_Llama3_HellaSwag.py
```

To characterize the correlation between error magnitude and frequency in impacting LLM performance, you can execute scripts following our `main_<MODEL>_<DATASET>_MSD.py` naming convention.

**Example**: To evaluate Llama-2-7b's resilience on LAMBADA and WikiText2:
```
cd src
CUDA_VISIBLE_DEVICES=1 python main_Llama2_LAMBADA_WikiText2_MSD.py
```

## Citation 
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@misc{xie2025realmreliableefficientlarge,
      title={ReaLM: Reliable and Efficient Large Language Model Inference with Statistical Algorithm-Based Fault Tolerance}, 
      author={Tong Xie and Jiawang Zhao and Zishen Wan and Zuodong Zhang and Yuan Wang and Runsheng Wang and Ru Huang and Meng Li},
      year={2025},
      eprint={2503.24053},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2503.24053}, 
}
```

## Acknowledgements

This project is built upon [SmoothQuant](https://github.com/mit-han-lab/smoothquant) [ICML'23]. We thank the authors for their excellent work.

