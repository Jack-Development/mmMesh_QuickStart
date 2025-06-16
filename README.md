# mmMesh - QuickStart

**Last Updated:** June 17, 2025

This is a fork of the [original mmMesh repository](https://github.com/HavocFiXer/mmMesh) by [HavocFiXer](https://github.com/HavocFiXer).
The original repository has not been updated since September 2021, but is currently one of the only open source methods for extracting SMPL data from mmWave radar data.

However, the original code lacks a clear explanation on how to implement it on your own data, as such, this fork was created to resolve that issue. This repository has been refactored to allow for easier setup and running of the code, specifically during steps 2 and 3.

Any further contributions from the community are welcome, so if you find an issue or an optimisation, then feel free to create an issue and pull request.


## Environment

* **Tested on Python 3.10.4**
* Linux (Ubuntu) environment for running Python scripts. Windows is only needed to start the radar.


## Repository Layout

```
├── 0.preliminary                    # SMPL model extraction
│   └── extract_SMPL_model.py
├── 1.mmWave_data_capture            # UDP capture scripts
│   ├── capture.py
│   ├── streaming.py
│   └── DataCaptureDemo_1843new.lua
├── 2.point_cloud_generation         # Convert .bin to .dat and point clouds
│   ├── data
│       ├── input
│       └── output
│   ├── configuration.py
│   └── pc_generation.py
├── 3.deep_model                     # Deep model training & inference
│   ├── data                        # Created automatically on first run
│       ├── input
│           ├── mmwave_data         # mmWave `.dat` files (train.dat, test.dat)
│           └── mocap_data          # MoCap files (`.pkl`)
│       └── output
│   ├── data_loader
│       └── dataloader.py
│   ├── models
│       ├── mmwave_model.py
│       ├── modules.py
│       ├── networks.py
│       └── utils.py
│   ├── smpl_models
│       ├── smpl
│       ├── smpl_utils_extend.py
│       └── smpl_wrapper.py
│   ├── train
│       ├── evaluator.py
│       ├── trainer.py
│       └── utils.py
│   ├── config.py
│   └── start.py
├── HISTORY.md
├── LICENSE
├── README.md
└── .gitignore
```

## 0. Preliminary: SMPL Model Extraction

1. Download the SMPL model package (v1.0.0 for Python 2.7) from the [SMPL website](https://smpl.is.tue.mpg.de/downloads).
2. Extract the archive so that the folder `SMPL_python_v.1.0.0/smpl/models/` is accessible.
3. Run:

   ```bash
   python 0.preliminary/extract_SMPL_model.py ./SMPL_python_v.1.0.0/smpl/models/
   ```
4. Two files will be generated in `0.preliminary`: `smpl_f.pkl` and `smpl_m.pkl`.
5. Install Chumpy if you encounter import errors:

   ```bash
   pip install chumpy
   ```

## 1. Real-time mmWave Data Capture

### On Windows

1. Open `mmWave Studio 2.0.0.2`.
2. Load `1.mmWave_data_capture/DataCaptureDemo_1843new.lua`.
3. Click **Run!** to begin streaming radar chirps over the network.

### On Ubuntu

1. Transfer the Ethernet connection from the Windows machine to your Ubuntu machine.
2. In the `1.mmWave_data_capture` directory, run:

   ```bash
   python capture.py <duration_in_minutes>
   ```

   Example, to capture 5 minutes:

   ```bash
   python capture.py 5
   ```
3. Captured frames will be saved or can be accessed via the `getFrame` method in `streaming.py`.

## 2. Point-Cloud Generation from mmWave Binary Data

1. Place your output binary file in `2.point_cloud_generation/data/input`.
2. Edit `2.point_cloud_generation/configuration.py` to match your radar parameters.
3. Edit `2.point_cloud_generation/configuration.py` to specify the desired pc_size and train/test split.
4. Run:

   ```bash
   python pc_generation.py
   ```

   Frame numbers are automatically detected based on the file size.
5. This will produce two `.dat` files: `train.dat` and `test.dat` in `2.point_cloud_generation/data/output`.

## 3. Deep Model: Training

1. **Arrange Data**

   * **mmWave**: Copy `train.dat` and `test.dat` into `3.deep_model/data/input/mmwave_data`.
   * **SMPL**: Copy `smpl_f.pkl` and `smpl_m.pkl` (from Step 0) into `3.deep_model/smpl_models/smpl`.
   * **MoCap**: Generate mocap `.pkl` data using [SOMA](https://github.com/nghorbani/soma) and place them in `3.deep_model/data/input/mocap_data` (See [Solve Already Labeled MoCaps With MoSh++](https://github.com/nghorbani/soma/blob/main/src/tutorials/solve_labeled_mocap.ipynb)).
   * If the `data/input` directories don't exist, running the training script once will create them.

2. **Training**
  Set your training parameters in `3.deep_model/config.py`

   ```bash
   python start.py
   ```

   * Training logs, checkpoints, and tensorboard summaries are saved under `3.deep_model/data/output` by default.

## 4. Deep Model: Inference
  NOT CURRENTLY IMPLEMENTED

## Visualisation

## Visualisation

During training, TensorBoard summaries (metrics, loss curves, etc.) are automatically saved to:
```
3.deep_model/data/output/logs
```

To launch TensorBoard, run:

```bash
tensorboard --logdir 3.deep_model/data/output/logs
```

## Citation

If you use this work, please cite the original researchers:

```bibtex
@inproceedings{xue2021mmmesh,
  title={mmMesh: towards 3D real-time dynamic human mesh construction using millimeter-wave},
  author={Xue, Hongfei and Ju, Yan and Miao, Chenglin and Wang, Yijiang and Wang, Shiyang and Zhang, Aidong and Su, Lu},
  booktitle={Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services},
  pages={269--282},
  year={2021}
}
```

---

## Acknowledgements

* Forked code from [HavocFiXer/mmMesh](https://github.com/HavocFiXer/mmMesh).
* SMPL Pytorch code partially adapted from [CalciferZh/SMPL](https://github.com/CalciferZh/SMPL).
* Anchor Point Module grouping code adapted from [yanx27/Pointnet\_Pointnet2\_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
