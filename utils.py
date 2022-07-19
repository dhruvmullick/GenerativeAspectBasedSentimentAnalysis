import numpy as np
import torch, math
from nltk.corpus import stopwords
from torch.utils.data import Dataset

import preprocess_for_eval

SEPARATOR = '<sep>'
POSITIVE, NEGATIVE, NEUTRAL = 'positive', 'negative', 'neutral'
POLARITIES = [POSITIVE, NEUTRAL, NEGATIVE]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        print("USING VAL F1 score")
        score = val_loss
        # score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # self.trace_func(
            #     f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(
                f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_full_language_name(language):
    full_name = ''
    if language == 'en':
        full_name = 'english'
    elif language == 'es':
        full_name = 'spanish'
    elif language == 'ru':
        full_name = 'russian'
    return full_name


def get_spacy_language(language):
    if language == 'en':
        return 'en_core_web_sm'
    if language == 'es':
        return 'es_core_news_sm'
    if language == 'ru':
        return 'ru_core_news_sm'


def normalise_sentence(sentence, language, spacy_nlp):
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('\"', '')
    sentence = sentence.replace('\'s ', ' ')
    sentence = sentence.replace('(', '')
    sentence = sentence.replace(')', '')
    sentence = sentence.lower()
    if sentence == '':
        return ''
    tokenised_sentence = sentence.split(" ")
    stop_words = set(stopwords.words(get_full_language_name(language)))
    tokenised_sentence = [w for w in tokenised_sentence if w not in stop_words and w != '' and w != ' ']

    lemmas = []
    for w in tokenised_sentence:
        if w == SEPARATOR or w in POLARITIES:
            lemmas.append(w)
            continue

        doc = spacy_nlp(w)
        # doc should have only one token, but if more then concatenate. e.g. with '(dvd)' changes to ( + dvd + ).
        lemma = ''
        for token in doc:
            lemma += token.lemma_
        lemmas.append(lemma)

    return ' '.join(lemmas)


def get_cleaned_polarities(sentence):
    sentence = preprocess_for_eval.clean_labels(sentence)
    sentence = preprocess_for_eval.add_missed_sep(sentence)
    polarities = sentence.split(SEPARATOR)
    polarities = [p.strip() for p in polarities if
                  not p.strip().startswith((NEGATIVE, POSITIVE, NEUTRAL)) and p.strip()]
    return polarities


def get_polarities_for_line(line, language, spacy_nlp):
    if len(line) < 3:
        print("line len smaller than 3: ")
        print(line)
    generated_sentence = normalise_sentence(line[1].strip(), language, spacy_nlp)
    true_sentence = normalise_sentence(line[2].strip(), language, spacy_nlp)
    generated_polarities = get_cleaned_polarities(generated_sentence)
    true_polarities = get_cleaned_polarities(true_sentence)
    return generated_polarities, true_polarities


def get_aspect_targets(polarity_sentence):
    return [' '.join(polarity.split()[:-1]) for polarity in polarity_sentence]


def get_mbart_lang(lang_code):
    if lang_code == 'en':
        return 'en_XX'
    elif lang_code == 'es':
        return 'es_XX'
    elif lang_code == 'ru':
        return 'ru_RU'


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
