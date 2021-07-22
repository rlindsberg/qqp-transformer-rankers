import logging
import pandas as pd
from transformers import BertTokenizerFast
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_crr
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.eval import results_analyses_tools
from transformer_rankers.models import pointwise_bert
from transformer_rankers.datasets import downloader
import wandb

wandb.login()

task = 'qqp'
data_folder = "./data/"
logging.info("Starting downloader for task {}".format(task))

dataDownloader = downloader.DataDownloader(task, data_folder)
dataDownloader.download_and_preprocess()

train = pd.read_csv("./data/{}/train.tsv".format(task), sep="\t")

valid = pd.read_csv(data_folder+task+"/valid.tsv", sep="\t")
# valid = valid[:100] #sampling so that eval is faster


# Random negative samplers
ns_train = negative_sampling.RandomNegativeSampler(list(train["question1"].values), 1)
ns_val = negative_sampling.RandomNegativeSampler(list(valid["question1"].values) + \
    list(train["question1"].values), 1)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

#Create the loaders for the datasets, with the respective negative samplers
dataloader = dataset.QueryDocumentDataLoader(train_df=train, val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train,
                                negative_sampler_val=ns_val, task_type='classification',
                                train_batch_size=32, val_batch_size=8, max_seq_len=100,
                                sample_data=-1, cache_path="{}/{}".format(data_folder, task))

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()


model = pointwise_bert.BertForPointwiseLearning.from_pretrained('bert-base-cased')

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                val_loader=val_loader, test_loader=test_loader,
                                num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                validate_every_epochs=1, num_validation_batches=-1,
                                num_epochs=10, lr=0.0005, sacred_ex=None,
                                validate_every_steps=-1, num_training_instances=-1)

#Train the model
logging.info("Fitting monoBERT for {}".format(task))
trainer.fit()

#Predict for test (in our example the validation set)
logging.info("Predicting")
preds, labels, softmax = trainer.test()
print('softmax\n ')
print(softmax)
res = results_analyses_tools.\
    evaluate_and_aggregate(preds, labels, ['R_10@1', 'R_10@5', 'map'])

for metric, v in res.items():
    logging.info("Test {} : {:4f}".format(metric, v))


pointwise_bert.BertForPointwiseLearning.save_pretrained(model, 'transformer_ranker_10epoch')
