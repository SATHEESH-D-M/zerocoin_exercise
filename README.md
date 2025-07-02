# Zerocoin Studios - Internship selection exercise 

- Name : Satheesh D M

- Roll : MA24M023

### Project structure

```bash
zerocoin_exercise                               #(root folder)
├── LICENSE
├── README.md
├── commits.txt                                 # commit history
├── data
│   ├── output.png                          
│   ├── output_comparison.png
│   └── sample.png                              # input image 
├── exercise.pdf                                # exercise task pdf
├── environment.yml                             # conda env yml file
├── jupyter_nbooks                              # try outs, ipynb files
│   ├── gaussian_blur_and_tone_mapping.cl
│   └── initial.ipynb
├── logs                                        # created on executing pipeline
├── src                                         # source code
│    ├── gaussian_blur_and_tone_mapping.cl      # kernels       
│    ├── pipeline.py                            # entrypoint script
│    └── utils.py                               
└── requirements.txt                            # pip requirements
```

### Instuctions

#### Step - 1 : Clone this repository
```bash
git clone https://github.com/SATHEESH-D-M/zerocoin_exercise.git
```

#### Step - 2 : Set up the project environment.
- conda environment
```bash
conda deactivate
conda env create -f environment.yml
conda activate zcoin
```

- venv (linux / macOS)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- venv (windows CMD)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step - 3 : Execute the pipeline
From the project root folder, run
```bash
python src/pipeline.py --input <input_img_path_str> --output <output_img_path_str> --mLuminance <float>
```
- replace <input_img_path_str> by the image path. 

- replace <output_img_path_str> by the output image path. 
- replace <float> with any value between 0.0 and 1.0

- example: (runs the default image)
```bash
python src/pipeline.py --input "data/sample.png" --output "data/output.png" --mLuminance 0.8
```
- this produces 2 processed images in the output location.

- run logs are saved in the ```./logs``` folder for each execution.

### Comments

- all images used as inputs and returned as outputs are in ```./data``` folder.

- ```./commits.txt``` has a detailed commit history made throughout the project.
- ```./src``` is the source code folder that is used in the processing pipeline.
- ```./src/gaussian_blur_and_tone_mapping.cl``` is the kernel code.
- ```./src/utils.py``` has all the reusable modular code.
- ```./src/pipeline.py``` makes use of ```./src/utils.py``` and is pipeline specific (not reusable).
- ```./jupyter_nbooks``` contains the trial and error .ipynb files.


