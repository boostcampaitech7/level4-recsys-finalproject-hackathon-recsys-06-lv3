# MovieLens-20m

# SASRec+
# python app.py dataset.max_length=200 SASRec.hidden_units=256
# SASRec+ 3000
# python app.py dataset.max_length=200 +dataset.num_negatives=3000 SASRec.hidden_units=256
# SASRec vanilla
# python app.py dataset.max_length=200 +seqrec_module.loss=bce +dataset.num_negatives=1 dataset.negative_sample="full" SASRec.hidden_units=256