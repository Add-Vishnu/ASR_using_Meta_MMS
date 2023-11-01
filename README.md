# ASR_using_Meta_MMS

First, we install transformers and some other libraries
```
pip install torch accelerate torchaudio datasets
pip install --upgrade transformers
````

**Note**: In order to use MMS you need to have at least `transformers >= 4.30` installed. If the `4.30` version
is not yet available [on PyPI](https://pypi.org/project/transformers/) make sure to install `transformers` from 
source:
```
pip install git+https://github.com/huggingface/transformers.git
```


we load the model and processor

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

We load the model and processor used to detect the language

```
model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

```

## Functions
- **transcribe : This function takes the audio input fed to mms-1b-all model. It detects the language and transcribes.
- **detect_language : This function takes the audio input fed to mms-lid-126 model. It detects and returns the language.
- **transcribe_lang : This function takes 2 inputs (audio,language). Using the same model in memory and we simply switch out the language adapters by calling the load_adapter() function for the model and set_target_lang() for the tokenizer. We pass the target language as an input Eg: "fra" for French.
- ```
  processor.tokenizer.set_target_lang(lang)
  model.load_adapter(lang)
  ```
