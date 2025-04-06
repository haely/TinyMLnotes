## Quick notes from Yi W's Linkedin- 

### Pt 1
Comparing real JAX- and PyTorch-based LLM training systems helps me understand experiment configuration including parallelism. 

JAX is compiler-oriented, so systems like AXLearn use a descriptive style to define the sharding specifications for each part of the model. This information is then passed to the GSPMD compiler for execution.

PyTorch, on the other hand, follows an imperative programming paradigm. As a result, systems like TorchTitan provide functions such as parallelize_llama that programmatically transform a model into its parallelized form.

In practice, however, the descriptive and imperative styles are often mixed — regardless of whether the system is based on JAX or PyTorch. For instance, TorchTitan introduces a TrainSpec, which adopts a descriptive approach: it specifies Python functions that handle model parallelization, serving a role analogous to the GSPMD compiler in the JAX ecosystem.

https://github.com/pytorch/torchtitan/blob/2157f53211c58ec4d89ee369c01135e5e5ce6bce/torchtitan/models/llama/parallelize_llama.py#L67C9-L106
https://github.com/apple/axlearn/blob/f0fc172a9f0dcb53860d42132bf8a497b0cc0dd2/axlearn/experiments/text/gpt/fuji.py#L181-L226
https://github.com/pytorch/torchtitan/issues/1055

![image](https://github.com/user-attachments/assets/e967c80b-054f-4519-9e6e-c0535c66a1b4)

### DDP
To show how data parallelism (DP), fully sharded data parallelism (FSDP), and tensor parallelism (TP) work—along with their combined forms, DP+TP and FSDP+TP. From here on, we can explore new combinations such as DP+FSDP+TP and introduce additional parallel techniques like sequence parallelism, pipeline parallelism, and expert parallelism. The OmniGraffle file is at 
https://wangkuiyi.github.io/dp+fsdp+tp/a.png

### Hugging face's format
![image](https://github.com/user-attachments/assets/bd89b815-14da-4408-b56a-cf9435fbd530)

### Pt 2
https://wangkuiyi.github.io/config.html
![image](https://github.com/user-attachments/assets/0464cd1c-7cb1-4e3c-ad61-67c43241dbb1)

https://irhum.github.io/blog/pjit/
https://yugeten.github.io/posts/2025/01/ppogrpo/
