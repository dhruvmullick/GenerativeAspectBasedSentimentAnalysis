# Importing libraries
import datetime
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
import time
import torch
from shutil import copyfile

import numpy as np
import pandas as pd
import spacy
import transformers
from rich import box
from rich.console import Console
from rich.table import Column, Table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBartTokenizer, LogitsProcessorList

import e2e_tbsa_preprocess
import evaluate_e2e_tbsa
import utils
from utils import EarlyStopping, YourDataSetClass
from logit_processor import CopyWordLogitsProcessor

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

LANG_MAP = {'en': 'english', 'es': 'spanish', 'ru': 'russian'}
SEED_LIST = [0]
LR_LIST = [5e-5]

if sys.argv[1] == "false":
    FULL_DATASET = False
else:
    FULL_DATASET = True

if len(sys.argv) >= 3:
    assert sys.argv[2][0] == '[' and sys.argv[2][-1] == ']'
    SEED_LIST = sys.argv[2][1:-1]
    SEED_LIST = SEED_LIST.split(sep='_')
    SEED_LIST = [int(x) for x in SEED_LIST]

print("SEEDS: {}".format(SEED_LIST))

# if len(sys.argv) >= 4:
#     if sys.argv[3] == "false":
#         USE_LOGIT_PROCESSOR = False
#     else:
#         USE_LOGIT_PROCESSOR = True

USE_LOGIT_PROCESSOR = True

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


def build_data(model_params, dataframes, source_text, target_text):
    # tokenzier for encoding the text
    tokenizer = MBartTokenizer.from_pretrained(model_params["MODEL"], src_lang=utils.get_mbart_lang(train_language),
                                               tgt_lang=utils.get_mbart_lang(train_language))

    # tokenizer = MBartTokenizer.from_pretrained(model_params["MODEL"], src_lang="en_XX", tgt_lang="en_XX")
    tokenizer.add_tokens(['<sep>', '<lang>'])  # , 'generate_english', 'generate_spanish', 'generate_russian'])

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state=model_params["SEED"]).reset_index(drop=True)
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
    test_params = {'batch_size': model_params["TEST_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader, tokenizer


def generate(tokenizer, model, device, loader, model_params, use_logit_processor=False):
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

            if not use_logit_processor:
                print("Not using logits processor")
                logits_processor_list = LogitsProcessorList([])
            else:
                print("Using logits processor")
                logits_processor_list = LogitsProcessorList([CopyWordLogitsProcessor(ids, mask, tokenizer)])

            generated_ids = model.generate(input_ids=ids, attention_mask=mask,
                                       logits_processor=logits_processor_list,
                                       max_length=256, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=1,
                                       decoder_start_token_id=tokenizer.lang_code_to_id[
                                           utils.get_mbart_lang(test_language)])

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

    torch.manual_seed(model_params['SEED'])  # pytorch random seed
    np.random.seed(model_params['SEED'])  # numpy random seed

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # training_logger = Table(Column("Epoch", justify="center"), Column("train_loss", justify="center"),
    #                         Column("val_loss", justify="center"), Column("Epoch Time", justify="center"),
    #                         title="Training Status", pad_edge=False, box=box.ASCII)

    training_logger = Table(Column("Epoch", justify="center"), Column("train_loss", justify="center"),
                            Column("val_f1", justify="center"), Column("Epoch Time", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = MBartForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = transformers.Adafactor(params=model.parameters(), lr=model_params["LEARNING_RATE"],
                                       scale_parameter=False, relative_step=False)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=False, delta=0.1,
                                   path=f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')
    avg_train_losses = []
    # avg_valid_losses = []

    for epoch in range(model_params["TRAIN_EPOCHS"]):

        start_time = time.time()
        train_losses = train(tokenizer, model, device, training_loader, optimizer)
        # valid_losses = validate(tokenizer, model, device, validation_loader)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)

        print("Early stopping: Calculating VALIDATION SCORE: ")
        prediction_file_name_validation = 'evaluation_predictions_val.csv'
        predictions_filepath_validation = '{}/{}'.format(model_params["OUTPUT_PATH"], prediction_file_name_validation)
        transformed_target_path_validation = '{}/{}'.format(model_params["OUTPUT_PATH"], "transformed_target_val.csv")
        transformed_sentiment_path_validation = '{}/{}'.format(model_params["OUTPUT_PATH"],
                                                               "transformed_sentiment_val.csv")

        T5Generator(validation_loader, model_params=model_params, output_file=prediction_file_name_validation,
                    model=model, tokenizer=tokenizer, use_logit_processor=False)

        valid_f1 = preprocess_and_evaluate(predictions_filepath_validation, "", transformed_sentiment_path_validation,
                                           transformed_target_path_validation, True)

        epoch_time = round(time.time() - start_time)
        # preparing the processing time for the epoch and est. the total.
        epoch_time_ = str(datetime.timedelta(seconds=epoch_time))

        # total_time_estimated_ = str(
        #     datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"] - epoch - 1))))
        # training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}',
        #                         f'{epoch_time_} (Total est. {total_time_estimated_})')

        total_time_estimated_ = str(
            datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"] - epoch - 1))))
        training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_f1}',
                                f'{epoch_time_} (Total est. {total_time_estimated_})')
        console.print(training_logger)

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(valid_loss, model)
        early_stopping(valid_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    console.log(f"[Replace best model with the last model]...\n")
    os.rename(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin',
              f'{model_params["OUTPUT_PATH"]}/model_files/last_epoch_pytorch_model.bin')
    copyfile(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin',
             f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
    os.remove(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')


def T5Generator(validation_loader, model_params, output_file, model=None, tokenizer=None, use_logit_processor=False):
    torch.manual_seed(model_params['SEED'])  # pytorch random seed
    np.random.seed(model_params['SEED'])  # numpy random seed

    console.log(f"[Loading Model]...\n")
    # Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")

    if model is None:
        console.log("Using model from path")
        model = MBartForConditionalGeneration.from_pretrained(path)
    else:
        console.log("Using existing model")

    if tokenizer is None:
        console.log("Using tokenizer from path")
        tokenizer = MBartTokenizer.from_pretrained(path)
    else:
        console.log("Using existing tokenizer")

    model = model.to(device)

    # evaluating test dataset

    console.log(f"[Initiating Generation]...\n")

    predictions, actuals, data_list = generate(tokenizer, model, device, validation_loader, model_params, use_logit_processor)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals, 'Original Sentence': data_list})
    final_df.to_csv(os.path.join(model_params["OUTPUT_PATH"], output_file))

    console.save_text(os.path.join(model_params["OUTPUT_PATH"], 'logs.txt'))

    console.log(f"[Generation Completed.]\n")

    console.print(f"""[Model] Model saved @ {os.path.join(model_params["OUTPUT_PATH"], "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(model_params["OUTPUT_PATH"], output_file)}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(model_params["OUTPUT_PATH"], 'logs.txt')}\n""")


def run_program_for_seed(seed, lr):
    MODEL_DIRECTORY = f"./generative-predictions_bart_test_copy_{USE_LOGIT_PROCESSOR}_seed_{seed}/{train_domain}_{train_language}"

    model_params = {
        "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
        "MODEL": "facebook/mbart-large-cc25",
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TEST_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 50,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": lr,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
        "early_stopping_patience": 10,  # number of epochs before stopping training.
        "SEED": seed
    }

    print(model_params)

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    predictions_file = f'{test_domain}_{test_language}_predictions.csv'
    prediction_file_path = os.path.join(model_params["OUTPUT_PATH"], predictions_file)
    transformed_targets_file = f'{test_domain}_{test_language}_transformed-targets.csv'
    transformed_targets_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_targets_file)
    transformed_sentiments_file = f'{test_domain}_{test_language}_transformed-sentiments.csv'
    transformed_sentiments_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_sentiments_file)

    training_loader, validation_loader, test_loader, tokenizer = build_data(model_params,
                                                                            dataframes=[training, validation, test],
                                                                            source_text="sentences_texts",
                                                                            target_text="sentences_opinions")

    T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)

    print("Generating and evaluating with logit processor = " + str(not USE_LOGIT_PROCESSOR))

    predictions_file = f'{test_domain}_{test_language}_predictions_non_copy.csv'
    prediction_file_path = os.path.join(model_params["OUTPUT_PATH"], predictions_file)
    transformed_targets_file = f'{test_domain}_{test_language}_transformed-targets_non_copy.csv'
    transformed_targets_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_targets_file)
    transformed_sentiments_file = f'{test_domain}_{test_language}_transformed-sentiments_non_copy.csv'
    transformed_sentiments_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_sentiments_file)

    T5Generator(test_loader, model_params=model_params, output_file=predictions_file, use_logit_processor=(not USE_LOGIT_PROCESSOR))
    preprocess_and_evaluate(prediction_file_path, seed, transformed_sentiments_file_path,
                            transformed_targets_file_path, False, lr)

    print("Generating and evaluating with logit processor = " + str(USE_LOGIT_PROCESSOR))

    predictions_file = f'{test_domain}_{test_language}_predictions_copy.csv'
    prediction_file_path = os.path.join(model_params["OUTPUT_PATH"], predictions_file)
    transformed_targets_file = f'{test_domain}_{test_language}_transformed-targets_copy.csv'
    transformed_targets_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_targets_file)
    transformed_sentiments_file = f'{test_domain}_{test_language}_transformed-sentiments_copy.csv'
    transformed_sentiments_file_path = os.path.join(model_params["OUTPUT_PATH"], transformed_sentiments_file)

    T5Generator(test_loader, model_params=model_params, output_file=predictions_file, use_logit_processor=USE_LOGIT_PROCESSOR)
    preprocess_and_evaluate(prediction_file_path, seed, transformed_sentiments_file_path,
                            transformed_targets_file_path, False, lr)


def preprocess_and_evaluate(prediction_file_path, seed, transformed_sentiments_file_path,
                            transformed_targets_file_path, validation, lr=0):
    nlp = spacy.load(utils.get_spacy_language(test_language), disable=['parser', 'ner'])
    print(f"Preprocessing file at {prediction_file_path}")
    e2e_tbsa_preprocess.transform_gold_and_truth(test_language, nlp, prediction_file_path,
                                                 transformed_targets_file_path, transformed_sentiments_file_path)
    print(f"Evaluating files at {transformed_targets_file_path} and {transformed_sentiments_file_path}")
    predicted_data, gold_data = evaluate_e2e_tbsa.read_transformed_sentiments(transformed_sentiments_file_path)

    if validation is False:
        print(f"Evaluated test file at: {transformed_sentiments_file_path}")
        print(f"SEED: {seed}, LR: {lr}, TEST SET OUTPUT: {evaluate_e2e_tbsa.evaluate_ts(gold_data, predicted_data)}")
        print("\n--------------------------\n")
    else:
        return evaluate_e2e_tbsa.evaluate_ts(gold_data, predicted_data)[2]


if __name__ == '__main__':

    for train_settings in [('Rest16', 'en'), ('Rest16', 'es'), ('Lap14', 'en'), ('Mams', 'en'), ('Rest16', 'ru')]:

        for test_settings in [('Lap14', 'en')]:

            train_domain = train_settings[0]
            train_language = train_settings[1]
            test_domain = test_settings[0]
            test_language = test_settings[1]

            if FULL_DATASET:
                training_file = './data/processed_full_train_{}_{}.csv'.format(train_domain, train_language)
            else:
                training_file = './data/processed_train_{}_{}.csv'.format(train_domain, train_language)

            validation_file = './data/processed_val_{}_{}.csv'.format(train_domain, train_language)
            test_file = './data/processed_test_{}_{}.csv'.format(test_domain, test_language)
            print("----------------\n\n"
                  "Experiment: Training on {}.{}, Testing on {}.{}".format(train_domain, train_language, test_domain,
                                                                           test_language))

            training = pd.read_csv(training_file)
            validation = pd.read_csv(validation_file)
            test = pd.read_csv(test_file)

            for lr in LR_LIST:
                for seed in SEED_LIST:
                    run_program_for_seed(seed, lr)