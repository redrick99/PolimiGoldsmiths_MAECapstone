# PolimiGoldsmiths_MAECapstone
Repository for the project L19 of the course Music and Acoustic Engineering Capstone from Politecnico di Milano in collaboration with Goldsmiths University of London

# Tutorial - How to run the Python Script
Here is a complete walkthrough on how to run the python script in order to test the visualization program. We'll see if we can provide an executable in the near future, so that this process will be easier to manage, but in the meantime here are the steps.

## Step 0 - Preparation
Here is everything that you have to do in order to be able to use the application.

### a) Install Python
**To run the python script you need Python, version >= 3.8, <= 3.10** (it might work with other versions, those are the ones we tested). If you have Python already installed, you can check the version by running one of the following commands:
```
python -V
python --version
python3 -V
python3 --version
```

If your python version is below version 3.8, we recommend updating or installing to version 3.8. If your python version is above 3.10, you can try to run the script and if it doesn't work you will have to downgrade to a version of python / install another one between 3.8 and 3.10 included.

Here are some websites you can visit to install python on your OS:
- [Windows](https://www.tomshardware.com/how-to/install-python-on-windows-10-and-11) 
- [MacOS](https://www.dataquest.io/blog/installing-python-on-mac/)
- [Linux](https://docs.python-guide.org/starting/install3/linux/)

Of course, make sure to install the python version we specified earlier (between 3.8 and 3.10 included)

### b) Download the code
- Method 1 (Download Repo) Here on GitHub, download all the repo as a .zip (there is a button for it in the repo page) and extract it to a folder on your PC. For the sake of this tutorial, we extracted the repo to the following path:
```
C:\Users\Pippo\Documents\Polimi\PolimiGoldsmiths_MAECapstone
```

- Method 2 (git clone) If you want to clone the repo so that you can pull all updates that we may release in the future, you can do so by doing the following steps:
  
  1. Install git on your system (if you haven't already)
  2. Open your terminal app or Windows Powershell on Windows
  3. Navigate to a path were you want to put the cloned repo. In our example, we run:
  
  ```
  cd C:\Users\Pippo\Documents\Polimi
  ```
  
  4. Type on the terminal/shell:
  
  ```
  git clone [url-of-the-github-repo]
  ```
  
  5. It will ask for passwords/permissions
  6. The repo should be cloned! You can now navigate to the repo folder
  7. If the repo failed to installed or git asks for special tokens or keyes, just ask me (Riccardo) what to do as it would be difficult to explain it here (sorry!)
  
## Step 1 - Run the python script
You should now be ready to run the python script. Here is how to do it.

1. Using the terminal/shell, navigate to the /python folder of the cloned or extracted repo on your pc
2. Create a new python virtual environment
3. Activate your new python virtual environment
4. Install required modules
5. Run the script

Here is the terminal/shell commands you can use to achieve this (we will use the example folder path from before):
```
cd C:\Users\Pippo\Documents\Polimi\PolimiGoldsmiths_MAECapstone\python
python -m venv venv
source ./venv/bin/activate
python3 -m pip install -r requirements.txt
python3 ./src/main.py
```

The script should now be up and running and you should see a prompt on the terminal screen. Just follow it and answer its questions!
