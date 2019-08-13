python eyey/fetch.py --folders INBOX --csvdir ./data # fetch
python eyey/model.py --outdir ./trained --csvdir ./data # train
python eyey/tagger.py --outdir trained --folders INBOX # run
