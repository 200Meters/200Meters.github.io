---
layout: post
title: Configuring a Windows Machine Learning VM with TensorFlow and GPU Support
date: 2021-01-06
---
## Setting up An Azure Machine Learning VM with TensorFlow 2.3 and GPU Support
While a CPU-based machine works well for many machine learning models and even for small text and image processing loads, if you want to do any substantial processing of deep learning models, particularly text or image job, a GPU-based machine becomes a necessity. For GPU processing, you have a few options. You can build your own GPU machine, but with GPU's running quickly into multiple thousands of dollars, there is a steep startup cost. Alternatively, you could use Google colab for free. However Colab has limitations on the length of the job (12hrs) and configuring Colab for anything more than a single notebook becomes overly complex. Services like Azure ML offer full MLOps support, but also entail latency for rapid and recursive experiment runs. Finally, you can build a VM or use a pre-configured data science VM to get an experience that has the flexibility of a local machine without the upfront cost of building your own GPU machine.

I recently was working on a text processing model which on my CPU machine would take about 45 days to train. I decided to cut this time by building a GPU machine in Azure specifically for this job. There are some nuances when building a GPU machine in Azure with TensorFlow support so I thought I would add a blog to the others out there to provide some tips and guidance (and to serve as my own notes if I have to do it again!).

Incidentally, the processing time for the first model run was reduced to 33 hours, so depending on your CPU machine, the time invested in configuring a GPU machine may well be worth it.

### Overview
Setting up a GPU machine in Azure to support TensorFlow has the following overall steps. The remainder of this article will go through each in detail.
1. Confirm required hardware and software
2. Create a data science VM in Azure
3. Confirm Visual Studio installation
4. Install Anaconda
5. Install JupyterLab
6. Install NVIDIA CUDA Toolkit and cuDNN
7. Confirm CUDA and cuDNN operation
8. Test for GPU support in JupyterLab

### Confirm required hardware and software
For the purpose of this article, we'll be setting up a machine capable of running TF 2.3.0 on GPU hardware. As a first step, determine the required hardware and software to do this. 

1. **GPU hardware:** In order to use the GPU with TF, you'll need to select a VM with the right GPU hardware to run Nvidia's DNN tools (CUDA and cuDNN). It should be noted that Nvidia's tools will not run on the least expensive Azure VM's that have a GPU. As such, you'll need to find a VM with the right GPU hardware. As of the time of this article, you can visit this ![Nvidia page](https://developer.nvidia.com/cuda-gpus) to see which GPU's will work. Keep this page handy when selecting a VM. 
2. **VM type:** Azure offers many different VM's depending on the anticipated workload. A full list can be found on the ![VM page](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes) and pricing can be found on the ![vm pricing page](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/). For GPU VM's suitable for deep learning, you'll want to look at the N family. Based on the requirements of Nvidia, the smallest VM (with a single GPU) Standard NC6. In my case this was about $916/mo or about $1.31/hr. Based on my subscription, I needed to request a quota increase in order to provision the machine. Build some extra time in for the quota increase request and approval if you will also need an increase. In my case it took about 8 hours to get the increase approved.
3. **CUDA/cuDNN Version**: Tensorflow has specific requirements for the version of CUDA/cuDNN in order to run on a GPU. When TF releases a version it's typically not to the most recent version of CUDA/cuDNN so make sure you take the time to figure out the right versions. In my case, i was installing TF 2.3.0 which required CUDA Toolkit 11.0. CUDA 11.0 required cuDNN 8.0.4. to find these requirements, use the following links:
- ![TensorFlow CUDA requirements](https://www.tensorflow.org/install/gpu#software_requirements)
- ![CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- ![cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) (lists which cuDNN supports which CUDA, and provides a download link)

### Create a Data Science Machine in Azure
Azure makes it very easy to provision VM's, and they even have data science machines pre-configured with much of the software you would want to use. The trick with getting TensorFlow 2.3.0 support on an Azure VM is making sure that you select the right machine in order to use the GPU processors available. Note also that not all machines are available in all regions and the prices vary between regions. 

