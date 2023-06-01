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

## Step 2 - Using the python script
Here is everything you have to know to be able to run the python script without issues (hopefully)

### a) Initialization
When you start the script, a bunch of different questions will be asked on the console: you have to answer those questions according to how you want to use the application. Here are the questions:

```
Question 1: Do you want to use live or recorded audio? (l/r):
```

Type `r` to use pre-recorded sample songs specifically adjusted to be played on this application (every instrument needs to be on a separate track so you can't just upload your own .wav file unfortunatly)

```
Question 2: Please select one of the songs (type only the index):
```

You can choose one of the available songs by typing the corresponding index. Song 1 and 2 have a bunch of synths with many tracks, along with drums, while Song 3 is a one-track song with a piano. More songs will be added eventually.

**If you want to add any song to the list of available songs please tell me, I will provide you with a new set of files if i can (just remember that I have to recreate the entire song in the DAW to be able to do this)**

```
Question 3: Do you want to hear the song as its being processed? (y/n):
```

Type `y` if you want to hear the song on the background while it's being processed. It is recommended you do so, so you can see the effects of the song on the visualization

```
Question 4: Choose the track for which you want to assign an instrument (from 0 to 0, leave blank to save):
```

Here you can just push the enter key and the application will work. This question is intended only for testing, I will provide automatic instrument selection based on the chosen song (since the audio is pre-recorded in this case)

```
Using external osc controller? (y/n):
```

Here you can type whatever you want since the osc controller feature is yet to be implemented.

**Done! The application should be now up and running and you should see some green text for initialization, and then some other blue text to see the format of the extracted features**


## Step 3 - Receiving OSC Messages
If you are not familiar with OSC Messages, [here is an explemenation of the protocol](https://ccrma.stanford.edu/groups/osc/index.html)

Receiving OSC Messages should be easy as long as you make sure that the address and port parameters match those of your application of choice. You will need to implement an OSC Receiver on the visualization end.

[Here is a tutorial on how to do this on openframeworks](https://www.youtube.com/watch?v=UXjMk5ti6wk&ab_channel=Packt)

You will need to set the address and port of the receiver to:
```
Address: "127.0.0.1"
Port: 12345
```

The OSC messages you will receive while the python script is running are of two types and have different characteristics. Each feature of the messages has its own argument inside the OSC Message.

### Low Level Feature Message
- **Description:** For every track of the sound card (or track of the song if using recorded audio), a Low Level Feature message is sent for every audio frame processed for features. The track corresponding to the message can be individuated by the number at the end of the message's OSC Address
- **OSC Address:** "/Lfmsg_ch*n*", where *n* is the number of the track from which the message is coming from
- **Args**
  - Spectral Centroid: *float*
  - Spectral Bandwidth: *float*
  - Spectral Flatness: *float*
  - Spectral Rolloff: *float*
  - 4 * Pitches: 4 arguments containing the 4 most prominent pitches found as frequencies: *float*
- **How often it's sent:** Between 5 to 20 times per second depending on the chunk size of the processed audio

### High Level Feature Message
- **Description:** High Level features are computed over the sum of all the tracks, so the corresponding OSC Messages are *not* differentiated based on the track number like Low Level feature messages are. 
- **OSC Address:** "/Hfmsg_ch0"
- **Args**
  - Amount of Arousal: *float from -1 (not arousintg at all) to 1 (very arousing)*
  - Amount of Valence: *float from -1 (very sad) to 1 (very happy)*
- **How often it's sent:** 1 Message is sent every 0.5 to *a few* seconds
- **KEEP IN MIND** the valence and arousal values are limited to -1 and 1, but will always be smaller than those values (they are usually between -0.5 and 0.5)

