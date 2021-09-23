## Getting Started with SeqMRI 

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Download the FastMRI single coil knee data from [FastMRI](https://fastmri.med.nyu.edu/)

### Download data and organise as follows

```
# For knee dataset         
└── datasets
    ├── knee       
        ├── knee_singlecoil_train
        ├── knee_singlecoil_val
```

### Train & Evaluate in Command Line

#### Loupe Training

Please refer to [train_loupe.sh](../examples/loupe/train_loupe.sh) for details of each arguments. 

```bash
# 4x line constrained sampling
bash examples/loupe/train_loupe.sh   1 4  ssim real-knee 4 5e-5 10 0.5 128 0 cuda:0 1 0 0 0 0

# 4x 2d point sampling 
bash examples/loupe/train_loupe.sh   1 22  ssim real-knee  4 1e-3 10 0.5 128 0 cuda:0 0 0 0 0 0

# 8x 2d point sampling
bash examples/loupe/train_loupe.sh   1 16  ssim real-knee  8 1e-3 10 0.5 128 0 cuda:0 0 0 0 0 0

# 16x 2d point sampling
bash examples/loupe/train_loupe.sh   1 12  ssim real-knee  16 1e-3 10 0.5 128 0 cuda:0 0 0 0 0 0
```

```bash
# 4x line constrained sampling
bash examples/loupe/train_loupe.sh   1 4  ssim brain 4 5e-5 10 0.5 128 0 cuda:0 1 0 0 0 0

# 4x 2d point sampling 
bash examples/loupe/train_loupe.sh   1 22  ssim brain  4 1e-3 10 0.5 128 0 cuda:0 0 0 0 0 0

```


#### Loupe Evaluation


```bash 
bash examples/loupe/test_loupe.sh  EXP_DIR  real-knee
```

Remeber to replace EXP_DIR with the path to the directory that contains the saved checkpoint. 


#### Sequential Sampling Training 

```bash
# 4x line constrained sampling 
bash examples/sequential/train_sequential.sh  NUM_STEP 1 4 ssim real-knee  4 5e-5  cuda:0 10 0.5 128 0 1

# 4x 2d point sampling 
bash examples/sequential/train_sequential.sh  NUM_STEP 1 22 ssim real-knee  4 1e-3  cuda:0 10 0.5 128 0 0

# 8x 2d point sampling
bash examples/sequential/train_sequential.sh  NUM_STEP 1 16 ssim real-knee  8 1e-3  cuda:0 10 0.5 128 0 0

# 16x 2d point sampling
bash examples/sequential/train_sequential.sh  NUM_STEP 1 12 ssim real-knee  16 1e-3  cuda:0 10 0.5 128 0 0
```

Remember to change NUM_STEP to the a value in [1,2,4] for sequential sampling. 

#### Sequential Sampling Evaluation 

```bash 
bash examples/sequential/test_sequential.sh EXP_DIR real-knee
```

### Model Zoo 

#### Line-constrained Sampling

| Model   | Accelearation | SSIM | Link | 
|---------|---------------|------|------|
| Loupe   |    4x         |  89.5    |  [URL](https://drive.google.com/drive/folders/1A-JFRd5KJ_HoCd2gYePjln67YzcsTiK5?usp=sharing) |
| Seq1   |    4x         |   90.8    |  [URL](https://drive.google.com/drive/folders/1vcIaIdSnlDPElbQm8kusBOfxR8FfMlzc?usp=sharing) |
| Seq4   |    4x         |   91.2    |  [URL](https://drive.google.com/drive/folders/1Y_fvnne5Gx0zaXFC0ANZYnlun7CeJ2Kv?usp=sharing) |


#### 2D Point Sampling 

| Model   | Accelearation | SSIM | Link | 
|---------|---------------|------|------|
| Loupe   |    4x         |  92.4    |  [URL](https://drive.google.com/drive/folders/1cTpc1V8EuLVyZ4iy3EIW_XhEzgmiecgN?usp=sharing) |
| Seq1   |    4x         |   92.7    |  [URL](https://drive.google.com/drive/folders/1ptKDYk7Dbff9kOJBXUPkpmLqoNoPA4_z?usp=sharing) |
| Seq4   |    4x         |   92.9     |  [URL](https://drive.google.com/drive/folders/1KG8vzruVlJkxlyywZUXDkQdXGyFJzaNB?usp=sharing) |

More models for traditional baseline methods (Random, Spectrum, etc..) are coming soon. 
