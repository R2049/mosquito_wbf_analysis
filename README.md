# Mosquito Wing-Beat-Frequency Analysis


Analysis code for: A Mosquito Test and Measurement Platform for High-Fidelity Acoustic Acquisition of Wing Beat Frequencies


## Analyses



The package performs three analyses:



1\. Free-flying female and male mosquito recordings:

&#x20;  - candidate-WBF detection availability;

&#x20;  - recording-level median candidate WBF;

&#x20;  - recording-level median local peak prominence;

&#x20;  - accepted candidate-WBF frequency distributions.



2\. Tethered female mosquito recordings:

&#x20;  - candidate-WBF detection availability;

&#x20;  - recording-level median candidate WBF;

&#x20;  - recording-level median local peak prominence.



3\. Mosquito-absent ACL-2 ambient recordings:

&#x20;  - ambient power spectral density outside and inside the isolation enclosure;

&#x20;  - integrated ambient-power reduction over predefined frequency bands;

&#x20;  - one-third-octave ambient-power reduction.



The tethered recording configurations were not paired and were collected in different environments. Their results are therefore descriptive and must not be interpreted as a controlled estimate of the enclosure effect. The matched ACL-2 mosquito-absent recordings provide the direct assessment of observed ambient-noise reduction inside the enclosure.



## Important measurement note



All spectral levels are relative digital measurements derived from WAV files.

They are not calibrated sound-pressure levels and must not be reported as dB SPL.



The software converts multichannel recordings to mono and removes DC offset.

It does not normalize, amplify, filter, denoise, or otherwise alter recording amplitude before analysis.



## Required folder structure



Organize the WAV files as follows:



&#x20;   data/

&#x20;   ├── free\_flying/

&#x20;   │   ├── female/

&#x20;   │   │   └── \*.wav

&#x20;   │   └── male/

&#x20;   │       └── \*.wav

&#x20;   ├── tethered/

&#x20;   │   ├── outside\_acl2/

&#x20;   │   │   └── \*.wav

&#x20;   │   └── inside\_acl2\_with\_enclosure/

&#x20;   │       └── \*.wav

&#x20;   └── ambient/

&#x20;       ├── ambient\_acl2.wav

&#x20;       └── ambient\_inside\_enclosure.wav



Only mosquito recordings should be placed in the free-flying and tethered folders. Ambient recordings must be placed in the ambient folder.


The `outside\_acl2` directory refers to tethered recordings obtained outside the ACL-2 laboratory without the isolation enclosure. The `inside\_acl2\_with\_enclosure` directory refers to tethered recordings obtained inside the sound-isolation enclosure under ACL-2 conditions.



## Installation



Python 3.12.



Create and activate a virtual environment:



Windows:



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate



macOS or Linux:



&#x20;   python3 -m venv .venv

&#x20;   source .venv/bin/activate



Install the dependencies:



&#x20;   pip install -r requirements.txt



## Running the complete analysis



From the package root, run:



&#x20;   python run\_analysis.py --data-dir data --output-dir results --config config.json



The default paths are `data`, `results`, and `config.json`, so the shorter command is also valid:



&#x20;   python run\_analysis.py



## Analysis settings



The settings are stored in `config.json`.



The supplied configuration uses:



- 1.0-s analysis windows;

- 0.5-s hop duration;

- Hann-windowed Welch power spectral density;

- 8,192-sample spectral segments for mosquito recordings;

- 16,384-sample spectral segments for ambient recordings;

- female candidate-WBF search range of 300–600 Hz;

- male candidate-WBF search range of 500–900 Hz;

- minimum primary peak prominence of 10 dB;

- minimum harmonic prominence of 4 dB;

- harmonic-frequency tolerance of 15 Hz;

- local background region of ±100 Hz;

- exclusion of ±10 Hz around the candidate peak from the local background.



A candidate WBF is accepted when its primary peak satisfies the prominence threshold and at least one supporting peak near two or three times the candidate frequency satisfies the harmonic-prominence threshold.



The fundamental is not assumed to be the highest-amplitude spectral component, because an upper harmonic may be stronger.



## Interpretation of candidate detection



A rejected analysis window is a window in which no candidate satisfied the predefined spectral criteria. It does not necessarily establish that mosquito-generated sound was absent.



Candidate detection is an acoustic classification. Without synchronized video or another behavioral reference, it does not independently confirm flight behavior.



## Overlapping windows



Recordings are divided into overlapping 1-s windows with a 0.5-s hop.

Adjacent windows are therefore not independent biological observations.



Window-level results are summarized at the recording level before comparison.

The recording-level summaries—not individual windows—are used in boxplots and condition summaries.



## Output structure



The program creates:



&#x20;   results/

&#x20;   ├── analysis\_metadata.json

&#x20;   ├── free\_flight/

&#x20;   ├── tethered/

&#x20;   └── ambient/



### Free-flight outputs



- `free\_flight\_per\_window.csv`

- `free\_flight\_per\_recording.csv`

- `free\_flight\_summary.csv`

- `free\_flight\_detection\_by\_sex.png`

- `free\_flight\_median\_wbf\_by\_sex.png`

- `free\_flight\_peak\_prominence\_by\_sex.png`

- female and male candidate-WBF histograms



### Tethered outputs



- `tethered\_per\_window.csv`

- `tethered\_per\_recording.csv`

- `tethered\_summary.csv`

- `tethered\_detection.png`

- `tethered\_peak\_prominence.png`



The tethered results are descriptive because the outside-ACL-2 and inside-enclosure recordings were not paired and were obtained in different environments.



### Ambient outputs



- `ambient\_spectrum\_comparison.csv`

- `ambient\_integrated\_band\_summary.csv`

- `ambient\_one\_third\_octave\_summary.csv`

- `ambient\_acl2\_vs\_inside\_spectrum.png`

- `ambient\_noise\_reduction\_one\_third\_octave.png`



Observed integrated ambient-power reduction is calculated as:



&#x20;   10 log10(P\_ACL2 / P\_enclosure)



Positive values indicate lower recorded ambient power inside the isolation enclosure.



## Citation





