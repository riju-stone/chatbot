```

├── Classifier
│   ├── data
│   │   ├── dev.txt
│   │   ├── inference.txt
│   │   ├── process_data.py
│   │   ├── processed_data.pkl
│   │   └── train.txt
│   ├── data_loader
│   │   └── bucket_and_batch.py
│   ├── model
│   │   ├── dialogue_acts.py
│   │   └── __init__.py
│   ├── Model_Backup
│   │   └── model.pt
│   └── train_and_test
│       └── train.py
├── evaluate.py
├── Generator
|   |-- blender
|   |   |-- configs
|   |   |   |-- added_tokens.json
|   |   |   |-- config.json
|   |   |   |-- merges.txt
|   |   |   |-- special_tokens_map.json
|   |   |   |-- tokenizer_config.json
|   |   |   └── vocab.json
|   |   └── parameters
|   |       |-- pytorch_model.bin
|   |       └── tf_model.h5
│   ├── DialoGPT
│   │   ├── Configs
│   │   │   ├── config.json
│   │   │   ├── merges.txt
|   |   |   |-- tokenizer_config.json
│   │   │   └── vocab.json
│   │   └── Parameters
│   │       ├── pytorch_model.bin
│   │       └── tf_model.h5
│   ├── Experimental Codes
│   │   ├── test_advanced_experimental_2.py
│   │   └── test_advanced_experimental.py
│   ├── generator.py
│   └── __init__.py
├── __init__.py
├── interact_generator_only.py
├── interact.py
├── interact_retrieval_only.py
├── interact_verbose.py
├── Readme.txt
├── Ranker
│   └── rank.py
├── Large_Scale_Retriever
│   ├── data
│   │   ├── advicea.csv
│   │   ├── adviceq.csv
│   │   ├── askphilosophya.csv
│   │   ├── askphilosophyq.csv
│   │   ├── askreddita.csv
│   │   ├── askredditq.csv
│   │   ├── asksciencea.csv
│   │   ├── askscienceq.csv
│   │   ├── casuala.csv
│   │   ├── casualq.csv
│   │   ├── eli5a.csv
│   │   ├── eli5q.csv
│   │   ├── mla.csv
│   │   ├── mlq.csv
|   |   |-- fetch_reddit_comments.sql
│   │   └── fetch_reddit_posts.sql
│   ├── Database
│   │   └── reddit.db
│   ├── Faiss_index
│   │   ├── large.index
│   │   └── thread_idx.pkl
│   ├── faiss_it.py
│   ├── fill_data.py
│   └── retrieve.py
├── Scripted_Retriever
│   ├── chatterbot_corpus
│   │   ├── ai.yml
│   │   ├── botprofile.yml
│   │   ├── computers.yml
│   │   ├── conversations.yml
│   │   ├── emotion.yml
│   │   ├── food.yml
│   │   ├── gossip.yml
│   │   ├── greetings.yml
│   │   ├── health.yml
│   │   ├── history.yml
│   │   ├── humor.yml
│   │   ├── literature.yml
│   │   ├── money.yml
│   │   ├── movies.yml
│   │   ├── politics.yml
│   │   ├── psychology.yml
│   │   ├── science.yml
│   │   ├── sports.yml
│   │   └── trivia.yml
│   ├── processed_scripts
│   │   ├── Bot_Profile.pkl
│   │   ├── Chatterbot.pkl
│   │   ├── embedded_bot_queries.pkl
│   │   ├── embedded_chatterbot_queries.pkl
│   │   ├── intent_query_script.pkl
│   │   └── intent_response_script.pkl
│   ├── random_reddit_rata
│   │   ├── jokesq.csv
│   │   ├── nostupidq.csv
│   │   ├── showerthoughtsq.csv
│   │   ├── tilq.csv
│   │   └── writingpromptsa.csv
│   ├── setup.py
│   └── subscripts
│       ├── fill_bot_profile.py
│       ├── fill_chatterbot.py
│       ├── intent_query_script.py
│       ├── intent_response_script.py
│       └── process_pkl.py
├── Encoder
│   ├── embeddings
│   │   ├── BERT
│   │   └── USE_QA
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
│   ├── encoder_client.py
│   ├── query_encoder.py
│   └── response_encoder.py
├── TTS
│   ├── best_model_config.json
│   ├── config.json
│   ├── config_kusal.json
│   ├── dataset_analysis
│   │   ├── AnalyzeDataset.ipynb
│   │   ├── analyze.py
│   │   └── README.md
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── Kusal.py
│   │   ├── LJSpeechCached.py
│   │   ├── LJSpeech.py
│   │   └── TWEB.py
│   ├── debug_config.py
│   ├── extract_feats.py
│   ├── hard-sentences.txt
│   ├── images
│   │   ├── example_model_output.png
│   │   └── model.png
│   ├── layers
│   │   ├── attention.py
│   │   ├── custom_layers.py
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── tacotron.py
│   ├── LICENSE.txt
│   ├── models
│   │   ├── __init__.py
│   │   └── tacotron.py
│   ├── notebooks
│   │   ├── Benchmark.ipynb
│   │   ├── ReadArticle.ipynb
│   │   ├── synthesis.py
│   │   └── TacotronPlayGround.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── server
│   │   ├── conf.json
│   │   ├── README.md
│   │   ├── server.py
│   │   ├── synthesizer.py
│   │   └── templates
│   │       └── index.html
│   ├── setup.py
│   ├── synthesis.py
│   ├── tests
│   │   ├── generic_utils_text.py
│   │   ├── __init__.py
│   │   ├── layers_tests.py
│   │   ├── loader_tests.py
│   │   ├── tacotron_tests.py
│   │   └── test_config.json
│   ├── text2speech.py
│   ├── ThisBranch.txt
│   ├── train.py
│   ├── TTS.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── tts_model
│   │   ├── best_model.pth.tar
│   │   └── config.json
│   ├── utils
│   │   ├── audio_lws.py
│   │   ├── audio.py
│   │   ├── data.py
│   │   ├── generic_utils.py
│   │   ├── __init__.py
│   │   ├── text
│   │   │   ├── cleaners.py
│   │   │   ├── cmudict.py
│   │   │   ├── __init__.py
│   │   │   ├── numbers.py
│   │   │   └── symbols.py
│   │   └── visual.py
│   └── version.py
└── Utils
    ├── functions_old.py
    ├── functions.py
    └── __init__.py

```