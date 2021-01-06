---
layout: post
title: Configuring a Windows Machine Learning VM with TensorFlow and GPU Support
date: 2021-01-06
---
## Setting up An Azure Machine Learning VM with TensorFlow and GPU Support
While a CPU-based machine works well for many machine learning models and even for small text and image processing loads, if you want to do any substantial processing of deep learning models, particularly text or image job, a GPU-based machine becomes a necessity. For GPU processing, you have a few options. You can build your own GPU machine, but with GPU's running quickly into multiple thousands of dollars, there is a steep startup cost. Alternatively, you could use Google colab for free. However Colab has limitations on the length of the job (12hrs) and configuring Colab for anything more than a single notebook becomes overly complex. Services like Azure ML offer full MLOps support, but also entail latency for rapid and recursive experiment runs. Finally, you can build a VM or use a pre-configured data science VM to get an experience that has the flexibility of a local machine without the upfront cost of building your own GPU machine.

I recently was working on a text processing model which on my CPU machine would take about 45 days to train. I decided to cut this time by building a GPU machine in Azure specifically for this job. There are some nuances when building a GPU machine in Azure with TensorFlow support so I thought I would add a blog to the others out there to provide some tips and guidance (and to serve as my own notes if I have to do it again!).

Incidentally, the processing time for the first model run was reduced to 33 hours, so depending on your CPU machine, the time invested in configuring a GPU machine may well be worth it.

### Overview
Setting up a GPU machine in Azure to support TensorFlow has the following overall steps. The remainder of this article will go through each in detail.
