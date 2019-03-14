Instruction

environment

    - ubuntu16.04
    - cuda 8.0
    - python3.6
    - opencv3

python requirements

    - numpy
    - keras >=2.0
    - scikit-learn
    - jupyter
    - scipy
    - matplotlib
    - Pillow
    - h5py
    - scikit-image
    - tensorflow-gpu<1.5 (it's depends version of cuda)
    ......

you can just run command as follow

> pip install -r ./requirementes.txt
> jupyter notebook

directory list
```
├── cbir
│   ├── crow.py
│   ├── ddt.py
│   ├── features.py
│   ├── __init__.py
│   ├── model.py  #vgg-16 model
│   ├── pwa.py  
│   ├── query.py
│   ├── rmac.py
│   └── utils.py
├── groundtruth.txt
├── image_data.ipynb  #extract features and save to results
├── new.ipynb  
├── old.ipynb  #query 
├── README.md
├── requirements.txt
└── visual.ipynb  #feature map visualize 
```
