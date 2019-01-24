# Fixed-Point-Training
Code needed to reproduce results from my ICLR 2019 paper on fixed-point quantization of the backprop algorithm.

Here you will find theano code for a CIFAR-10 ResNet 20 network:
The folder train-baseline includes baseline training with required data dump for doing precision analysis. The part of analysis related to gradient precisions is also found in that folder in the form of several python scripts.
The folder FFanalysis includes code neede for feedforward precision analysis.
The folder train-internal includes code for fixed-point training (all tensors quantized) after precisions are determined.

Please get in touch if you have any question or comment.

Charbel Sakr and Naresh Shanbhag. "Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm." International Conference on Learning Representations. 2019.

@inproceedings{sakr2018pertensor,

title={Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm},

author={Charbel Sakr and Naresh Shanbhag},

booktitle={International Conference on Learning Representations},

year={2019},

}

Charbel
