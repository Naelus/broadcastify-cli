# broadcastify-cli
This is a tool to download brodcastify archives. It requires a broadcastify.com
premium subscription.

This is a fork of https://github.com/NotJoeMartinez/broadcastify-cli where I'm adding additional features and making tweaks to fit my use case

### Installation 
You will need to install ffmpeg prior to installing broadcast-cli (in windows you can use winget install ffmpeg
If you are going to use gpu transcoding currently I only have nvidia support implemented you will need to install the cuda 12 dlls, see [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu) for instructions  
```bash
pip install broadcastify-cli
```

```bash
git clone https://github.com/NotJoeMartinez/broadcastify-cli
cd broadcastify-cli
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```


### Configuration
make a `.env` file and enter your username and password in the variables
```
USERNAME="yourUserName"
PASSWORD="yourPassword"
```

### Usage

You need to get the feed id from broadcastify feed page you intend to download.
It's the last digits of the url.
```
url: https://www.broadcastify.com/archives/feed/5318
feed id: 5318 
```

Once you have the feed id you can pass it to the `download` command using the 
`--feed-id` flag and you should specify a date with the `--date` flag in the 
format `MM/DD/YY`. **It will download all available archives if `--date` is not
specified. Or you can use the flag --past-days <n> to download the past N number of days** 

This command will download all recordings from [Dallas City Police](https://www.broadcastify.com/archives/feed/5318) on April 17th 2024:
```bash
broadcastify-cli download --feed-id 5318 --date 04/17/2024
```

This will create a new directory tree formated like `archives/feed_id/YYYYMMDD`

```
archives
└── 5318
    └── 04172024
        ├── 202404170021-373175-5318.mp3
        ├── 202404170050-396916-5318.mp3
        ├── 202404170120-465838-5318.mp3
        ├── 202404170150-841921-5318.mp3
        ├── 202404170220-588343-5318.mp3
        ├── 202404170250-11638-5318.mp3
...
```

### TODO:
Need to make changes to transcription to give timestamped outputs

Would be nice to also do diarization, need to look into that

Need to update readme with further info on other flags and tweaks I've added so far


