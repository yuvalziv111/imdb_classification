import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import pyarrow as pa
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.metrics import precision_recall_fscore_support as score
from transformers import EarlyStoppingCallback
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline
from transformers.trainer_utils import number_of_arguments
from typing import Optional, Dict, Any, Callable
import sentencepiece
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem import PorterStemmer
from datasets import Dataset
import json
import pandas as pd
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
import setfit
from setfit import SetFitModel, SetFitTrainer
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import datasets
from optuna import Trial

d = {}
#Creating a custom trainer for evaluating multiple metrics
class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)

    def evaluate(self, dataset: Optional[Dataset] = None, metric_key_prefix: str = "test") -> Dict[str, float]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if dataset is not None:
            self._validate_column_mapping(dataset)
            if self.column_mapping is not None:
                eval_dataset = self._apply_column_mapping(dataset, self.column_mapping)
            else:
                eval_dataset = dataset
        else:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided to `Trainer.evaluate` nor the `Trainer` initialzation.")

        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        print("***** Running evaluation *****")
        y_pred = self.model.predict(x_test, use_labels=False)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu()

        # Normalize string outputs
        if y_test and isinstance(y_test[0], str):
            encoder = LabelEncoder()
            encoder.fit(list(y_test) + list(y_pred))
            y_test = encoder.transform(y_test)
            y_pred = encoder.transform(y_pred)

        final_results = []
        for metric in self.metric:
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(metric, config_name=metric_config)
            if metric != 'accuracy':
                metric_kwargs = self.metric_kwargs
                results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)
            else:
                metric_kwargs = {}
                results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)
            final_results.append(results)
        if not isinstance(results, dict):
            results = {"metric": results}
        self.model.model_card_data.post_training_eval_results(
            {f"{metric_key_prefix}_{key}": value for key, value in results.items()}
        )
        return final_results

#Creating the dataset class
class Create_Data(Dataset):
    def __init__(self, data_proc):
        self.data = data_proc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']  # Adjust column names if needed
        return line, label

#parameters for hyperparameters optimization
def params_dict(trial: Trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-5, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate",0.0001, 0.01, log=True),
        "num_epochs": trial.suggest_int("num_epochs", low = 5, high = 20),
        "batch_size": trial.suggest_categorical("batch_size", [6, 8]),
        "seed": trial.suggest_int("seed", 3, 42),
        "max_iter": trial.suggest_int("max_iter", 100, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        'head_learning_rate': trial.suggest_float("head_learning_rate",1e-5, 1e-2, log=True)
    }

#model init function for the trainer
def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", **params).to(device)



#preprocessing function- remove stopwords (except for negative word), stemming, removing punctuation marks
def preprocess_text(text):
    #text preprocessing
    # Tokenize the text
    if not isinstance(text, str):
        tokens = ""
    else:
        tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    remove_list = ['not','no','off','nor',"wasn't","haven't","now","isn't","hasn't","don't","doesn't","didn't","couldn't","aren't",'out','more','than']
    for item in remove_list:
        stop_words.remove(item)
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation marks
    table = str.maketrans('', '', string.punctuation)
    tokens =[token.translate(table) for token in tokens if token.translate(table)]

    ### Stemming
    stemmed = []
    stemmer = PorterStemmer()
    for token in tokens:
        if token.isalpha():

            word = stemmer.stem(token)
            stemmed.append(word)

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(stemmed)

    return preprocessed_text

#assigning the GPU device
device = torch.device("cuda")
torch.cuda.empty_cache()

#read the dataset
dataset = datasets.load_dataset("stanfordnlp/imdb")
df = pd.DataFrame([dataset['train']['text'], dataset['train']['label']]).T.rename(columns = {0:'text', 1:'label'})
nltk.download('stopwords')
nltk.download('punkt')

#rename the headers of the dataset
df= df.rename(columns = {'remark' : 'text', 'topic':'label'})
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)
predict_set = df[df['label']==str(np.nan)]
df = df[df['label']!=str(np.nan)]

#read the test set
df_test = pd.read_csv(r"C:\Users\yuval\OneDrive\שולחן העבודה\text classification\test for setfit.csv")

topics = df["label"].unique()
train_set = pd.DataFrame()
test_set = pd.DataFrame()

df['text'] = df['text'].apply(lambda x: preprocess_text(x))

#split into train and test
for topic in topics:
    # select only the rows with the current topic
    topic_rows = df[df["label"] == topic]
    if len(topic_rows) >= 100:
        topic_rows = topic_rows.sample(n=50)
    if df.loc[df['label'] == topic].shape[0] < 10:
        df = df.loc[df['test']['label'] != topic]
        print (fr"excluded topic number {int(topic)}")
        continue
    # split the topic rows into train and test sets
    train, test = train_test_split(topic_rows, test_size=0.8)

    # add the train and test sets to the overall train and test sets
    train_set = pd.concat([train_set, train])
    test_set = pd.concat([test_set, test])
test_set = df_test
print("Train set size:", len(train_set))
print("Test set size:", len(test_set))
remarks = df['text'].apply(lambda x: [str(i) for i in x.split(',')]).tolist()
print ("downloading tokenizer")

#read the pretrained model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='SetFit/MiniLM-L12-H384-uncased__sst2__all-train')
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
model = SetFitModel.from_pretrained(pretrained_model_name_or_path='SetFit/MiniLM-L12-H384-uncased__sst2__all-train', labels=2)
torch.cuda.empty_cache()
model.to(device)

# Prepare training data
train_data = Dataset.from_pandas(train_set)
val_data = Dataset.from_pandas(test_set)
test = test_set.copy()

early_stopping = EarlyStoppingCallback(early_stopping_patience = 3)
#assign the trainer class
trainer = MyTrainer(
    train_dataset=train_data,
    eval_dataset=val_data,
    model_init=model_init,
    metric_kwargs={"average": "macro"},
    # callbacks=[early_stopping],
    column_mapping={"text": "text", "label": "label"},
    metric =['f1', 'accuracy','precision', 'recall']
)
#a function for the target function to maximize- a combination of the f1 score and accuracy
def custom_compute_objective(metrics):
    # Option 1: Remove extra argument
    f1 = metrics[0]['f1']
    accuracy = metrics[1]['accuracy']

    return 0.5 * f1 + 0.5 * accuracy

trainer.compute_objective = custom_compute_objective

#Hyperparameters optimization search, takes long time
search = False
if search == True:
    best_run =trainer.hyperparameter_search( direction="maximize",backend="optuna", hp_space=params_dict,n_trials=30,
                                         compute_objective=custom_compute_objective)
else:
    best_run_path = fr"C:\Users\yuval\OneDrive\שולחן העבודה\text classification\best_params.json"
    with open(best_run_path, 'r') as json_file:
        best_run = json_file.readlines()[0]
        best_run = json.loads(best_run)
trainer.apply_hyperparameters(best_run, final_model=True)
print ("Best hyperparameters are: ", best_run)
trainer.train()
metrics = trainer.evaluate(val_data, topics)
print("The Evaluation metrics are: ", metrics)

path = r'C:\Users\yuval\OneDrive\שולחן העבודה\text classification\Setfit_text_classification.pth'

# save the model
def custom_save_pretrained(model, save_directory):
    import os
    os.makedirs(save_directory, exist_ok=True)
    model.labels = ['1', '0']
    model.save_pretrained(save_directory)

# Save the model to a directory
custom_save_pretrained(model, r"C:\Users\yuval\OneDrive\שולחן העבודה\text classification\SetFit/MiniLM-L12-H384-uncased__sst2__all-train")