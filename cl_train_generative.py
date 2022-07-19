# Importing libraries
import os, time, torch, datetime
from re import I
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from utils import EarlyStopping
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console
from shutil import copyfile

# pd.set_option('display.max_colwidth', -1)


# define a rich console logger
console = Console(record=True)

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

training_logger = Table(Column("Epoch", justify="center"), Column("train_loss", justify="center"),
                        Column("val_loss", justify="center"), Column("Epoch Time", justify="center"),
                        title="Training Status", pad_edge=False, box=box.ASCII)
lang_map = {'en': 'english', 'es': 'spanish', 'ru': 'russian'}
extra_data_map = {'en': ['es', 'ru'], 'es': ['en', 'ru'], 'ru': ['en', 'es']}




class YourDataSetClass(Dataset):
    """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.summ_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'sentences_texts': source_text
        }


def train(tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function
    """
    train_losses = []
    model.train()
    for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Processing batches..'):
        y = data['target_ids'].to(device, dtype=torch.long)
        lm_labels = y.clone()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


def validate(tokenizer, model, device, loader):
    """
    Function to be called for validating the trainner with the parameters passed from main function
    """
    validate_losses = []
    model.eval()
    for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Validating batches..'):
        y = data['target_ids'].to(device, dtype=torch.long)
        lm_labels = y.clone()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
        loss = outputs[0]
        validate_losses.append(loss.item())
    return validate_losses


def build_data(dataframes, source_text, target_text):
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens(['<sep>', '<lang>'])#, 'generate_english', 'generate_spanish', 'generate_russian'])
    # special_tokens_dict = {'additional_special_tokens': ['<sep>']}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model_params['new_tokens_size'] = len(tokenizer)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state=0).reset_index(drop=True)
    val_dataset = dataframes[1].reset_index(drop=True)
    test_dataset = dataframes[2].reset_index(drop=True)
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"VALIDATION Dataset: {val_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    test_set = YourDataSetClass(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)

    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    test_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader, tokenizer


def generate(tokenizer, model, device, loader, model_params):
    """
  Function to evaluate model for spanbert-predictions

  """
    model.eval()
    predictions = []
    actuals = []
    data_list = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(,

            # generated_ids = model.generate(input_ids = ids, attention_mask = mask,
            #                                max_length=256, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=1)
            # max_length=(int(sys.argv[1])), num_beams=(int(sys.argv[2])), length_penalty=(float(sys.argv[3])), no_repeat_ngram_size=3, early_stopping=True)
            # max_length=256, num_beams=4, length_penalty=1.5, no_repeat_ngram_size=3, early_stopping=True)

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            data_list.extend(data["sentences_texts"])
    return predictions, actuals, data_list


def T5Trainer(training_loader, validation_loader, tokenizer, model_params):
    """
    T5 trainer
    """

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    # model.resize_token_embeddings(model_params['new_tokens_size'])

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_params["LEARNING_RATE"])
    # optimizer = Adafactor(params = model.parameters(), relative_step=True, lr = model_params["LEARNING_RATE"])

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=False,
                                   path=f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        start_time = time.time()
        train_losses = train(tokenizer, model, device, training_loader, optimizer)
        valid_losses = validate(tokenizer, model, device, validation_loader)
        epoch_time = round(time.time() - start_time)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # preparing the processing time for the epoch and est. the total.
        epoch_time_ = str(datetime.timedelta(seconds=epoch_time))
        total_time_estimated_ = str(
            datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"] - epoch - 1))))
        training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}',
                                f'{epoch_time_} (Total est. {total_time_estimated_})')
        console.print(training_logger)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    console.log(f"[Replace best model with the last model]...\n")
    os.rename(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin', f'{model_params["OUTPUT_PATH"]}/model_files/last_epoch_pytorch_model.bin')
    copyfile(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin', f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
    os.remove(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')



def T5Generator(validation_loader, model_params, output_file):
    console.log(f"[Loading Model]...\n")
    # Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = model.to(device)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals, data_list = generate(tokenizer, model, device, validation_loader, model_params)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals, 'Original Sentence': data_list})
        final_df.to_csv(os.path.join(model_params["OUTPUT_PATH"], output_file))

    console.save_text(os.path.join(model_params["OUTPUT_PATH"], 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(model_params["OUTPUT_PATH"], "model_files")}\n""")
    console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(model_params["OUTPUT_PATH"], output_file)}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(model_params["OUTPUT_PATH"], 'logs.txt')}\n""")


if __name__ == '__main__':
    # domain: Rest16, Lap14, Mams, Mams_short
    # lang: en, es, ru
    for train_settings in [('Rest16', 'es', 'Lap14', 'en'), ('Rest16', 'ru', 'Lap14', 'en')]:
            
            train_domain = train_settings[0]
            train_language = train_settings[1]
            test_domain = train_settings[2]
            test_language = train_settings[3]
            training_file = '/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_train_{}_{}.csv'.format(train_domain, train_language)
            validation_file = '/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_val_{}_{}.csv'.format(train_domain, train_language)
            test_file = '/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_test_{}_{}.csv'.format(test_domain, test_language)
            print("Experiment: Training on {}.{}, Testing on {}.{}".format(train_domain, train_language, test_domain, test_language))

            # Loading files
            training = pd.read_csv(training_file)
            validation = pd.read_csv(validation_file)
            test = pd.read_csv(test_file)
            
            # For cross-lingual
            training['sentences_texts'] = training['sentences_texts'].map(lambda txt: f'{lang_map[train_settings[1]]} : {txt}')
            validation['sentences_texts'] = validation['sentences_texts'].map(lambda txt: f'{lang_map[train_settings[1]]} : {txt}')
            test['sentences_texts'] = test['sentences_texts'].map(lambda txt: f'{lang_map[train_settings[3]]} : {txt}')

            # Loading extra data
            training_ext1 = pd.read_csv(training_file.replace(f'_{train_settings[1]}.csv', f'_{train_settings[1]}_to_{extra_data_map[train_settings[1]][0]}_processed.csv'))
            training_ext2 = pd.read_csv(training_file.replace(f'_{train_settings[1]}.csv', f'_{train_settings[1]}_to_{extra_data_map[train_settings[1]][1]}_processed.csv'))
            validation_ext1 = pd.read_csv(validation_file.replace(f'_{train_settings[1]}.csv', f'_{train_settings[1]}_to_{extra_data_map[train_settings[1]][0]}_processed.csv'))
            validation_ext2 = pd.read_csv(validation_file.replace(f'_{train_settings[1]}.csv', f'_{train_settings[1]}_to_{extra_data_map[train_settings[1]][1]}_processed.csv'))
            training_ext1['sentences_texts'] = training_ext1['sentences_texts'].map(lambda txt: f'{lang_map[extra_data_map[train_settings[1]][0]]} : {txt}')
            training_ext2['sentences_texts'] = training_ext2['sentences_texts'].map(lambda txt: f'{lang_map[extra_data_map[train_settings[1]][1]]} : {txt}')
            validation_ext1['sentences_texts'] = validation_ext1['sentences_texts'].map(lambda txt: f'{lang_map[extra_data_map[train_settings[1]][0]]} : {txt}')
            validation_ext2['sentences_texts'] = validation_ext2['sentences_texts'].map(lambda txt: f'{lang_map[extra_data_map[train_settings[1]][1]]} : {txt}')

            # Concatenating
            training = pd.concat([training_ext1, training_ext2, training], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
            validation = pd.concat([validation_ext1, validation_ext2, validation], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)

            model_params = {
                "OUTPUT_PATH": f"./generative-predictions/CL_{'_'.join(train_settings[:2])}",  # output path
                "MODEL": "google/mt5-base",  # model_type: t5-base/t5-large
                "TRAIN_BATCH_SIZE": 8,  # training batch size
                "VALID_BATCH_SIZE": 4,  # validation batch size
                "TRAIN_EPOCHS": 300,  # number of training epochs
                "VAL_EPOCHS": 1,  # number of validation epochs
                "LEARNING_RATE": 9e-4,  # learning rate
                "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
                "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
                "early_stopping_patience": 20,  # number of epochs before stopping training.
            }

            training_loader, validation_loader, test_loader, tokenizer = build_data(dataframes=[training, validation, test],
                                                                                    source_text="sentences_texts", target_text="sentences_opinions")

            # T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)
            T5Generator(test_loader, model_params=model_params, output_file=f'{test_domain}_{test_language}_predictions.csv')
