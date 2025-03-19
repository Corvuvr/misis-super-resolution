# STUDY
env=CorvuvrStudy
if { conda env list | grep $env; } >/dev/null 2>&1; then
    echo "$env already exists."
else
    conda create -n $env Python=3.10
fi
conda activate $env
pip install tensorflow[and-cuda] pandas pillow matplotlib basicsr facexlib gfpgan --no-cache-dir --no-input
conda deactivate

# RT4KSR
env=CorvuvrRT4KSR
if { conda env list | grep $env; } >/dev/null 2>&1; then
    echo "$env already exists."
else
    conda create -n $env Python=3.10
fi
conda activate $env
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
cd RT4KSR
    pip install -r requirements.txt  --no-input
cd -
conda deactivate

# ESRGAN
env=CorvuvrESRGAN
if { conda env list | grep $env; } >/dev/null 2>&1; then
    echo "$env already exists."
else
    conda create -n $env Python=3.7
fi
conda activate $env
pip install tensorflow[and-cuda] basicsr facexlib gfpgan  --no-input
cd Real-ESRGAN 
    pip install -r requirements.txt --no-cache-dir --no-input
    python setup.py develop 
cd -
conda deactivate