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
For the purpose of this article, we'll be setting up a machine capable of running TF 2.4.0 on GPU hardware. As a first step, determine the required hardware and software to do this. 

1. **GPU hardware:** In order to use the GPU with TF, you'll need to select a VM with the right GPU hardware to run Nvidia's DNN tools (CUDA and cuDNN). It should be noted that Nvidia's tools will not run on the least expensive Azure VM's that have a GPU. As such, you'll need to find a VM with the right GPU hardware. As of the time of this article, you can visit this [Nvidia page](https://developer.nvidia.com/cuda-gpus) to see which GPU's will work. Keep this page handy when selecting a VM. 
2. **VM type:** Azure offers many different VM's depending on the anticipated workload. A full list can be found on the [VM page](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes) and pricing can be found on the [vm pricing page](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/). For GPU VM's suitable for deep learning, you'll want to look at the N family. Based on the requirements of Nvidia, the smallest VM (with a single GPU) Standard NC6. In my case this was about $916/mo or about $1.31/hr. Based on my subscription, I needed to request a quota increase in order to provision the machine. Build some extra time in for the quota increase request and approval if you will also need an increase. In my case it took about 8 hours to get the increase approved.
3. **CUDA/cuDNN Version**: Tensorflow has specific requirements for the version of CUDA/cuDNN in order to run on a GPU. When TF releases a version it's typically not to the most recent version of CUDA/cuDNN so make sure you take the time to figure out the right versions. In my case, I was installing TF 2.4.0 which required CUDA Toolkit 11.0. CUDA 11.0 required cuDNN 8.0.4. to find these requirements, use the following links:
- [TensorFlow CUDA requirements](https://www.tensorflow.org/install/gpu#software_requirements)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) (lists which cuDNN supports which CUDA, and provides a download link)
4. **Python/Anaconda**: TensorFlow 2.4.0 requires Python versions 3.6-3.8. Install a Python version and environment manager to support your TF environment. 

### Create a Data Science Machine in Azure
Azure makes it very easy to provision VM's, and they even have data science machines pre-configured with much of the software you would want to use. The trick with getting TensorFlow 2.3.0 support on an Azure VM is making sure that you select the right machine in order to use the GPU processors available. Note also that not all machines are available in all regions and the prices vary between regions. In order to provision the vm, perform the following steps using the  Azure Portal. Alternatively, you can provision the machine via script, but for a single machine, the portal is pretty easy.

- Open the Azure portal and select 'Create resource'
- Type in Data Science and select 'Data Science Virtual Machine - Windows 2019'
- Select 'Create' to begin the provisioning process
- Select the machine type you want to use from the NC family. Note that not all NC machines are available in each location. Fill in the details for resource group, VM name, adminsitrator user name, and password.
- Click next and either assign a disk or select the default disk
- Click next to set up networking options 
- Click next to assign management options such as monitoring
- Click next for advanced optionsd. Select 'Select an extension to install' and choose the NVIDIA GPU driver extension 
- Click next and add desired tags.
- Click next to review and create

Once the machine has been provisioned, you can go ahead and log into it via RDP. Note again that if you do not have sufficient quota for an NC machine, you will need to ask for a quota increase before you can select the NC machine.

### Confirm Visual Studio Installation
By starting with a data science vm, you'll have visual studio community edition installed. The CUDA toolkit and cuDNN rely on Visual Studio for some of the run time components so VS is needed. You will also use VS to verify the GPU operation. This isa good time to check if there are any VS updates and install them if there are. Finally, the DSVM's do come with CUDA and cuDNN alreadt installed and configured. As of this writing, it came with version 10.1 of the CUDA toolkit. If you are are going to upgrade to 11.0 for TF 2.3.0, then you will want to uninstall the CUDA toolkit and cuDNN at this time so there are no conflicts later. Use the standard Windows Settings>Apps to uninstall them.

### Install Anaconda
The DSVM's do come with Python and miniconda. I like to have Anaconda available for environment management. If you would like to have Anaconda installed, go to [Anaconda Individual Site](https://www.anaconda.com/products/individual) to download and install the Windows version.

### Install CUDA and cuDNN
In order to take advantage of the GPU's you'll need to install the CUDA Toolkit and CUDNN. In order to download the software, you'll need to go to [NVIDIA.com](https://www.nvidia.com) and create a free developer account. Once you have the account and have logged in, go to the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download the CUDA toolkit, and go to the [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) to get the cuDNN.

- **CUDA Installation:** After downloading the CUDA Toolkit, go ahead and run the Windows install. For detailed instructions, visit the Windows installation page linked from the CUDA archive (in this case [Windows documentation for 11.0](https://docs.nvidia.com/cuda/archive/11.0/cuda-installation-guide-microsoft-windows/index.html)). The instructions are verbose, so I won't repeat them here. I will however point out that you'll need to compile and run a couple of applications in Visual Studio to confirm the installation.

- **cuDNN Installation:** The CUDA toolkit relies on the cuDNN in order to operate. Once you have installed the CUDA toolkit, you'll need to install the cuDNN. The process of installing the cuDNN is a matter of extracting files from the cuDNN zip file download and copying them to the appropriate location in the CUDA installation. Again, NVIDIA provides the details in the [cuDNN install guide for Windows ](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows) so I won't repeat them here, but it is worth noting the you will want to make sure you copy the files to the right location and that the correct environment variables are set on your Windows machine. Again, see the documentation, but take note of these two steps in particular. 



