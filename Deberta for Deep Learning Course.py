import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW, DebertaV2ForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import MultiLabelBinarizer
import transformers
import sentencepiece
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem import PorterStemmer
import string
import datasets
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

d = {}
# a functions that calculates the classification metrics
def calc_metrics(predicted_labels, true_labels, num_topics, labels):
    global mapping
    topic_correct_counts = [0] * num_topics
    topic_sample_counts = [0] * num_topics

    lst = []
    for i in range(len(true_labels)):
        lst.append([predicted_labels[i], true_labels[i]])

    f1_micro_avg = f1_score(y_true = true_labels, y_pred = predicted_labels, average = "macro")
    precision = precision_score(y_true = true_labels, y_pred = predicted_labels, average = "macro")
    recall = recall_score(y_true = true_labels, y_pred = predicted_labels, average = "macro")
    # accuracy = accuracy_score(true_labels, predicted_labels, average = "micro")
    return precision, recall, f1_micro_avg

    print (labels)
    topic_accuracies = [correct_count / sample_count if sample_count > 0 else 0.0 for correct_count, sample_count in
                        zip(topic_correct_counts, topic_sample_counts)]

    return topic_accuracies

#a dataset class
class Create_Data(Dataset):
    def __init__(self, data_proc):
        self.data = data_proc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        return line, label

# Defining the Class for tokenization and processing the text for input to the data_loader
class NLPModel(nn.Module):
    def __init__(self, tokenizer, max_seq_len=None, model=None):
        super(NLPModel, self).__init__()
        self.config = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='yangheng/deberta-v3-base-absa-v1.1')
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.model = model
        return
#a function for changing the original model parameters
    def unfreeze_layers(self, num_layer):
        if isinstance(num_layer, int) is True:
            for ind, x in enumerate(self.modules()):
                # Comparing to feature extractor, now we are gonna unfreeze part of the layers and train our model using them
                if ind > num_layer:
                    for param in x.parameters():
                        param.requires_grad = True
        else:
            for ind, x in enumerate(self.modules()):
                for param in x.parameters():
                    param.requires_grad = True
    def forward(self, input_ids, attention_mask,labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        prob = F.softmax(logits,dim=1)
        return prob, loss


# Training function
def activate_train(data_loader, optimizer, scheduler, device):
    global ROBERTA_Class
    ROBERTA_Class.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0

    for texts, labels in data_loader:
        inputs = ROBERTA_Class.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=ROBERTA_Class.max_seq_len)
        true_labels += labels.numpy().flatten().tolist()
        optimizer.zero_grad()
        batch = {}

        batch = {'input_ids': inputs['input_ids'].to(device),
                              'attention_mask': inputs['attention_mask'].to(device),
                              'labels': torch.tensor(labels.type(torch.LongTensor)).to(device)}
        outputs = ROBERTA_Class(**batch)
        loss = outputs[1]

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ROBERTA_Class.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += outputs[0].argmax(axis=-1).flatten().tolist()

    precision, recall, f1_score = calc_metrics(predictions_labels,true_labels,len(df['label'].unique()), df['label'].unique())
    avg_epoch_loss = total_loss / len(data_loader)
    return predictions_labels, true_labels, avg_epoch_loss,precision, recall, f1_score

# Function for validation
def validate(data_loader):
    global model
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    for epoch in epoch_list:
        for texts,labels in data_loader:
            inputs = ROBERTA_Class.tokenizer(text=texts,
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=ROBERTA_Class.max_seq_len)
            true_labels += labels.numpy().flatten().tolist()
            batch = {'input_ids': inputs['input_ids'].to(device),
                     'attention_mask': inputs['attention_mask'].to(device),
                     'labels': torch.tensor(labels.type(torch.LongTensor)).to(device)}
            with torch.no_grad():
                outputs = ROBERTA_Class(**batch)
                loss = outputs[1]
                one_hot_labels = torch.zeros(int(labels.size()[0]), len(df['label'].unique()))
                for label, i in zip(labels, range(len(df['label'].unique()))):
                    label = int(label)
                    one_hot_labels[i][label] = 1

                total_loss += loss.item()
                predictions_labels += outputs[0].argmax(axis=-1).flatten().tolist()
        avg_epoch_loss = total_loss / len(data_loader)
        print (fr"epoch: {epoch}, loss: {avg_epoch_loss}")
        precision, recall, f1_score = calc_metrics(predictions_labels, true_labels, len(df['label'].unique()), df['label'].unique())
        print ("val precisions: ", precision)
        print("val recall: ", recall)
        print ("val f1_scores: ", f1_score)

        return predictions_labels, true_labels, avg_epoch_loss,precision, recall
#hyperparameters for training
max_len = 1000 # Max lenght of the text for input
batch_size = 8
epochs = 10
epoch_list = range(1, epochs+1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Device: ", device)
torch.cuda.empty_cache()

# Load the data
dataset = datasets.load_dataset('stanfordnlp/imdb')
df = pd.DataFrame([dataset['train']['text'], dataset['train']['label']]).T.rename(columns = {0:'text', 1:'label'}).iloc[101:,:]
df_test = pd.read_csv(fr"C:\Users\yuval\OneDrive\שולחן העבודה\text classification\test for setfit.csv")
nltk.download('stopwords')
nltk.download('punkt')

#preprocessing function- remove stopwords (except for negative word), stemming, removing punctuation marks etc.
def preprocess_text(text):
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

df= df.rename(columns = {'remark' : 'text', 'hierarchy topic':'label'})
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)
predict_set = df[df['label']==str(np.nan)]
df = df[df['label']!=str(np.nan)]

topics = df["label"].unique()
train_set = pd.DataFrame()
test_set = pd.DataFrame()

df['text'] = df['text'].apply(lambda x: preprocess_text(x))

#balance the train set
for topic in topics:
    # select only the rows with the current topic
    topic_rows = df[df["label"] == topic]
    if len(topic_rows) > 100:
        topic_rows = topic_rows.sample(n=8)
    # split the topic rows into train and test sets
    train, test = train_test_split(topic_rows, test_size=0.2)

    # add the train and test sets to the overall train and test sets
    train_set = pd.concat([train_set, train])
    test_set = pd.concat([test_set, test])

test_set = df_test
print("Train set size:", len(train_set))
print("Test set size:", len(test_set))
remarks = df['text'].apply(lambda x: [str(i) for i in x.split(',')]).tolist()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='yangheng/deberta-v3-base-absa-v1.1')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='yangheng/deberta-v3-base-absa-v1.1',num_labels=len(df['label'].unique()),ignore_mismatched_sizes=True)
model.to(device)


ROBERTA_Class = NLPModel(tokenizer=tokenizer, max_seq_len=max_len,model=model)
ROBERTA_Class.unfreeze_layers(5)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
# Prepare training data
train_data = Create_Data(train_set)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
inputs_for_roberta = []

val_data = Create_Data(test_set)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#training function
def train_model(train_dataloader,val_dataloader):
  optimizer = AdamW(ROBERTA_Class.parameters(), lr = 1e-4, eps = 1e-8, weight_decay=0.01)
  # ROBERTA_Class, optimizer = ipex.optimize(ROBERTA_Class, optimizer, dtype=torch.float32)
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, num_training_steps = total_steps)
  loss = []
  accuracy = []
  val_loss_list = []
  val_accuracy_list = []

  for epoch in tqdm(range(epochs)):
      train_labels, true_labels, train_loss,train_precision, train_recall, train_f1_score = activate_train(train_dataloader, optimizer, scheduler, device)
      train_acc = accuracy_score(true_labels, train_labels)
      print(fr'epoch: {epoch+1} train accuracy %.2f, train loss: {train_loss}' % (train_acc))
      print ("train precision:", train_precision)
      print("train recall:", train_recall)
      print ("f1 score:", train_f1_score)
      loss.append(train_loss)
      accuracy.append(train_acc)

      val_labels, val_true_labels, val_loss, val_precision, val_recall = validate(val_dataloader)

      val_acc= accuracy_score(val_true_labels, val_labels,)
      print(f'epoch: {epoch+1} validation accuracy {val_acc}')
      print ("val precision: ", val_precision)
      print ("val recall: ",val_recall)

      val_loss_list.append(val_loss)
      val_accuracy_list.append(val_acc)
  predicted_labels= []
  true_labels = []
  for i in range(len(val_labels)):
    predicted = d[val_labels[i]]
    true = d[val_true_labels[i]]
    predicted_labels.append(predicted)
    true_labels.append(true)
  data_remarks = val_dataloader.dataset.data['text'].values.tolist()


  predicted_results = pd.DataFrame({'predicted': predicted_labels, 'true': true_labels, 'remark': data_remarks})
  return loss, accuracy, val_loss_list, val_accuracy_list, val_precision, val_recall , predicted_results
# Training the Model
roberta_loss,roberta_accuracy,roberta_val_loss_list,roberta_val_accuracy_list,val_precision, val_recall, results = train_model(train_dataloader,val_dataloader)

results.to_csv(fr'C:\Users\yuval\OneDrive\שולחן העבודה\deep learning text classification project\results_predictions.csv')

optimizer =AdamW(ROBERTA_Class.parameters(), lr = 1e-4, eps = 1e-8, weight_decay=0.01)
output_model = fr'C:\Users\yuval\OneDrive\שולחן העבודה\deep learning text classification project\Deberta_classification.pth'

# save the model
def save(model,optimizer):                                                # save model
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

save(model, optimizer)