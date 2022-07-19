import csv
import utils
import sys
import os
import spacy

# PREDICTIONS_FILE = 'dummymodel/predictions_dummy.csv'
# PREDICTIONS_FILE = 'generative-predictions/{}/{}_{}_predictions.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'generative-predictions/{}/transformed-targets_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'generative-predictions/{}/transformed-sentiments_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])

PREDICTIONS_FILE = 'spanbert-predictions-transformed/train_spanbert_{}.csv/test_spanbert_{}.csv/predictions.json'
# PREDICTIONS_FILE = 'models/combined/predictions_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# PREDICTIONS_FILE = 'dummymodel/{}_predictions_dummy.csv'

TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'spanbert-predictions-transformed/tbsa-preprocessed/train_spanbert_{}.csv/test_spanbert_{}.csv/transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'spanbert-predictions-transformed/tbsa-preprocessed/train_spanbert_{}.csv/test_spanbert_{}.csv/transformed-sentiments.csv'
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/combined/transformed-sentiments_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'dummymodel/transformed/{}_transformed-targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'dummymodel/transformed/{}_transformed-sentiments.csv'

SEPARATOR = '<sep>'

sentiment_to_identifier = {
    'positive': 'POS',
    'negative': 'NEG',
    'neutral': 'NEU'
}


def get_sentiment(polarity_list):
    return [polarity.split()[-1] for polarity in polarity_list]


def get_sentence_matched_with_targets(aspects_targets, real_sentence):
    real_target_idx = []
    for tg in aspects_targets:
        if not tg.strip():
            continue
        idx_start = real_sentence.find(tg)
        if idx_start == -1:
            continue
        idx_end = idx_start + len(tg) - 1
        real_target_idx.append((idx_start, idx_end))
    return real_target_idx


def get_sentence_matched_with_targets_and_sentiments(aspects_targets, aspects_sentiments, real_sentence):
    real_idx = []
    for i, tg in enumerate(aspects_targets):
        if not tg.strip():
            continue
        if aspects_sentiments[i] not in sentiment_to_identifier.keys():
            # print("no sentiment found in {}".format(real_sentence))
            continue
        idx_start = real_sentence.find(tg)
        if idx_start == -1:
            continue
        idx_end = idx_start + len(tg) - 1
        real_idx.append((idx_start, idx_end, sentiment_to_identifier[aspects_sentiments[i]]))

    return real_idx


def transform_line_for_target_extraction(line, language, spacy_nlp):
    # if len(line) < 3:
    #     return [], []
    generated_polarities, true_polarities = utils.get_polarities_for_line(line, language, spacy_nlp)
    real_sentence = utils.normalise_sentence(line[3].strip(), language, spacy_nlp)

    generated_aspects_targets = utils.get_aspect_targets(generated_polarities)
    true_aspects_targets = utils.get_aspect_targets(true_polarities)

    generated_target_idx_list = list(set(get_sentence_matched_with_targets(generated_aspects_targets, real_sentence)))

    # Not using set for true target index
    true_target_idx_list = get_sentence_matched_with_targets(true_aspects_targets, real_sentence)

    generated_target_idx_list.sort()
    true_target_idx_list.sort()

    return generated_target_idx_list, true_target_idx_list


def transform_line_for_sentiment_extraction(line, language, spacy_nlp):
    generated_polarities, true_polarities = utils.get_polarities_for_line(line, language, spacy_nlp)
    real_sentence = utils.normalise_sentence(line[3].strip(), language, spacy_nlp)

    generated_aspects_targets = utils.get_aspect_targets(generated_polarities)
    true_aspects_targets = utils.get_aspect_targets(true_polarities)

    generated_aspects_sentiments = get_sentiment(generated_polarities)
    true_aspects_sentiments = get_sentiment(true_polarities)

    generated_sentiment_idx_list = get_sentence_matched_with_targets_and_sentiments(generated_aspects_targets,
                                                                                    generated_aspects_sentiments,
                                                                                    real_sentence)
    generated_sentiment_idx_list.sort()
    generated_sentiment_idx_list_deduped = list(set(generated_sentiment_idx_list))
    # if len(generated_sentiment_idx_list_deduped) != len(generated_sentiment_idx_list):
    #     print("Here")

    true_sentiment_idx_list = get_sentence_matched_with_targets_and_sentiments(true_aspects_targets,
                                                                               true_aspects_sentiments, real_sentence)

    return generated_sentiment_idx_list_deduped, true_sentiment_idx_list


def transform_gold_and_truth(language, spacy_nlp, predictions_file, transformed_targets_predictions_file, transformed_sentiments_predictions_file):
    with open(predictions_file, 'r') as csvfile:
        os.makedirs(os.path.dirname(transformed_targets_predictions_file), exist_ok=True)
        with open(transformed_targets_predictions_file, 'w') as newfile_targets:
            os.makedirs(os.path.dirname(transformed_sentiments_predictions_file), exist_ok=True)
            with open(transformed_sentiments_predictions_file, 'w') as newfile_sentiments:
                reader = csv.reader(csvfile)
                next(reader, None)  # skip the headers
                writer_targets = csv.writer(newfile_targets)
                writer_targets.writerow(["Predicted idx", "Gold idx"])
                writer_sentiments = csv.writer(newfile_sentiments)
                writer_sentiments.writerow(["Predicted sentiment tags", "Gold sentiment tags"])
                for line in reader:
                    pred_transformed, gold_transformed = transform_line_for_target_extraction(line, language, spacy_nlp)
                    # print("Original line: " + line[3])
                    # print("Prediction transformed: " + str(pred_transformed))
                    # print("Gold transformed: " + str(gold_transformed))
                    writer_targets.writerow([pred_transformed, gold_transformed])
                    pred_sentiment_transformed, gold_sentiment_transformed = transform_line_for_sentiment_extraction(
                        line, language, spacy_nlp)
                    writer_sentiments.writerow([pred_sentiment_transformed, gold_sentiment_transformed])


# Pass argument as the language code - 'en', 'es', 'ru'

# training_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en', 'Mams_en', 'Mams_short_en']
# training_datasets = ['Rest16_en_merged', 'Rest16_es_merged', 'Rest16_ru_merged', 'Lap14_en_merged']
training_datasets = ['Lap14_en_merged']
# test_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en', 'Mams_en', 'Mams_short_en']
# test_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en']
test_datasets = ['Lap14_en']
language = {'Rest16_en': 'en', 'Rest16_es': 'es', 'Rest16_ru': 'ru', 'Lap14_en': 'en', 'Mams_en': 'en',
            'Mams_short_en': 'en', 'Rest16_en_merged': 'en', 'Rest16_es_merged': 'es', 'Rest16_ru_merged': 'ru',
            'Lap14_en_merged': 'en'}


if __name__ == '__main__':

    #### For dummy
    # for dtest in datasets:
    #     transform_gold_and_truth(language[dtest], PREDICTIONS_FILE.format(dtest),
    #                              TRANSFORMED_TARGETS_PREDICTIONS_FILE.format(dtest),
    #                              TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE.format(dtest))
