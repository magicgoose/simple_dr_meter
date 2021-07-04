# simple_dr_meter
This is a fully compliant Dynamic Range meter. It computes DR at defined by http://dr.loudness-war.info/ and generates logs with all the proper formatting.

## Prerequisites:

* `ffmpeg` and `ffprobe` are on your PATH
* CPython 3.6+ (I don't think I am using anything specific to CPython, but I only test it on CPython)
* Python packages from PyPI, listed in `requirements.txt` — can be installed with a command like `pip3 install -r requirements.txt`

## How to use:

	$ main.py <input>

where `<input>` is one of:
* a directory with audio files
* a CUE sheet (.cue file)
* a single audio file

An audio file may also contain an embedded cuesheet, it will be used if found.

It will generate `dr.txt` with the proper DR log inside the target folder (or next to the input file), and show something in the stdout while working. That's it.

## Some extra words:

The work was initially based on https://github.com/simon-r/dr14_t.meter and it would be quite a tough start should I not take a look at it, but almost everything was rewritten since then.

The main differences from dr14_t.meter are:

* it can read ANY audio file with any number of channels, because all the complexity of decoding and reading tags is delegated to ffmpeg/ffprobe
* supports CUE sheets as input!
* audio file size is practically unlimited, because audio data isn't fully loaded into RAM
* faster processing, even faster than the official foobar2000 plugin.

on the other hand, it doesn't have any extra fancy features like graphs, etc. mainly because I don't need it, feel free to do pull requests anyway.

So, basically it's a simple and practically complete implementation of DR measurement, made to make it easier to spread the knowledge and take a stand against the Loudness War™.

## TODO:

* Release it on PyPI! I have no experience with that but I should probably learn how to do this.