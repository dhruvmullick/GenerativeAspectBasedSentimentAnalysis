### Adapted from
# https://github.com/lixin4ever/E2E-TBSA/blob/8cb7182d890d2b36766b23c788bb82c265ca4242/evals.py#L98

import numpy as np
import csv
import ast
import sys

SMALL_POSITIVE_CONST = 1e-4

# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'generative-predictions/{}/transformed-targets_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'generative-predictions/{}/transformed-sentiments_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])

# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/combined/transformed-sentiments.csv'
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/combined/transformed-sentiments_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])

TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'spanbert-predictions-transformed/tbsa-preprocessed/train_spanbert_{}.csv/test_spanbert_{}.csv/transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'spanbert-predictions-transformed/tbsa-preprocessed/train_spanbert_{}.csv/test_spanbert_{}.csv/transformed-sentiments.csv'

# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'dummymodel/transformed/{}_transformed-targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'dummymodel/transformed/{}_transformed-sentiments.csv'

# Dhruv's e.g. evaluate_ote("O O O S O B I E", "O O O S O O O O") ->
# (0.9999000099990001, 0.9999000099990001, 0.9998500124991251)
# print(evaluate_ote(["O O O S O B I E"], ["O O O S O O O O"]))
# -> (0.9999000099990001, 0.49997500124993743, 0.6665777829629381)
# Updated if not tag conversion: print(evaluate_ote("[(6, 6), (10, 14)]", "[(6, 6), (10, 14)]"))
def evaluate_ote(gold_ot, pred_ot):
    """
    evaluate the model performance for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    # assert len(gold_ot) == len(pred_ot)
    n_samples = len(gold_ot)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        # g_ot_sequence, p_ot_sequence = tag2ot(ote_tag_sequence=g_ot), tag2ot(ote_tag_sequence=p_ot)
        g_ot_sequence, p_ot_sequence = g_ot, p_ot
        # hit number
        n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)
        n_tp_ot += n_hit_ot
        n_gold_ot += len(g_ot_sequence)
        n_pred_ot += len(p_ot_sequence)
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    ote_scores = (ot_precision, ot_recall, ot_f1)
    return ote_scores


# Dhruv's e.g. print(tag2ot("O O O S O B I E")) = [(6, 6), (10, 14)]
# print(tag2ot(["O","O","O","S","O", "B", "I", "E"])) = [(3, 3), (5, 7)]
# def tag2ot(ote_tag_sequence):
#     """
#     transform ote tag sequence to a sequence of opinion target
#     :param ote_tag_sequence: tag sequence for ote task
#     :return:
#     """
#     n_tags = len(ote_tag_sequence)
#     ot_sequence = []
#     beg, end = -1, -1
#     for i in range(n_tags):
#         tag = ote_tag_sequence[i]
#         if tag == 'S':
#             ot_sequence.append((i, i))
#         elif tag == 'B':
#             beg = i
#         elif tag == 'E':
#             end = i
#             if end > beg > -1:
#                 ot_sequence.append((beg, end))
#                 beg, end = -1, -1
#     return ot_sequence


def match_ot(gold_ote_sequence, pred_ote_sequence):
    """
    calculate the number of correctly predicted opinion target
    :param gold_ote_sequence: gold standard opinion target sequence
    :param pred_ote_sequence: predicted opinion target sequence
    :return: matched number
    """
    n_hit = 0
    for t in pred_ote_sequence:
        if t in gold_ote_sequence:
            n_hit += 1
    return n_hit

# Dhruv's e.g. Updated: print(evaluate_ts([[(1, 3, 'POS'), (4, 4, 'NEG')]], [[(1, 3, 'POS'), (4, 4, 'NEG')]]))
def evaluate_ts(gold_ts, pred_ts):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        # g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        g_ts_sequence, p_ts_sequence = g_ts, p_ts
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        # calculate macro-average scores for ts task
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    ts_macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    n_tp_total = sum(n_tp_ts)
    # total sum of TP and FN
    n_g_total = sum(n_gold_ts)
    # total sum of TP and FP
    n_p_total = sum(n_pred_ts)

    ts_micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    ts_micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST)
    ts_scores = (ts_micro_p*100, ts_micro_r*100, ts_micro_f1*100)
    return ts_scores


# Dhruv's e.g. print(tag2ts(["O", "B-POS", "I-POS", "E-POS", "S-NEG", "O"])) -> [(1, 3, 'POS'), (4, 4, 'NEG')]
# def tag2ts(ts_tag_sequence):
#     """
#     transform ts tag sequence to targeted sentiment
#     :param ts_tag_sequence: tag sequence for ts task
#     :return:
#     """
#     n_tags = len(ts_tag_sequence)
#     ts_sequence, sentiments = [], []
#     beg, end = -1, -1
#     for i in range(n_tags):
#         ts_tag = ts_tag_sequence[i]
#         # current position and sentiment
#         eles = ts_tag.split('-')
#         if len(eles) == 2:
#             pos, sentiment = eles
#         else:
#             pos, sentiment = 'O', 'O'
#         if sentiment != 'O':
#             # current word is a subjective word
#             sentiments.append(sentiment)
#         if pos == 'S':
#             # singleton
#             ts_sequence.append((i, i, sentiment))
#             sentiments = []
#         elif pos == 'B':
#             beg = i
#         elif pos == 'E':
#             end = i
#             # schema1: only the consistent sentiment tags are accepted
#             # that is, all of the sentiment tags are the same
#             if end > beg > -1 and len(set(sentiments)) == 1:
#                 ts_sequence.append((beg, end, sentiment))
#                 sentiments = []
#                 beg, end = -1, -1
#     return ts_sequence


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        # print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def read_transformed_targets(transformed_targets_predictions_file):
    predicted_data = []
    gold_data = []
    with open(transformed_targets_predictions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            predicted_idx = ast.literal_eval(line[0])
            gold_idx = ast.literal_eval(line[1])
            predicted_data.append(predicted_idx)
            gold_data.append(gold_idx)
    return predicted_data, gold_data


def read_transformed_sentiments(transformed_sentiments_predictions_file):
    predicted_data = []
    gold_data = []
    with open(transformed_sentiments_predictions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            predicted_idx = ast.literal_eval(line[0])
            gold_idx = ast.literal_eval(line[1])
            predicted_data.append(predicted_idx)
            gold_data.append(gold_idx)
    return predicted_data, gold_data


# print(evaluate_ts(["O", "B-POS", "I-POS", "E-POS", "S-NEG", "O"], ["O", "O", "O", "O", "S-NEG", "O"]))
# print(evaluate_ts([["O", "B-POS", "I-POS", "E-POS", "S-NEG", "S-POS"]], [["O", "B-POS", "I-POS", "E-POS", "S-NEG", "O"]]))
# print(evaluate_ts([[(1, 3, 'POS'), (4, 4, 'NEG')]], [[(1, 3, 'POS'), (4, 4, 'NEG')]]))

# training_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en', 'Mams_en', 'Mams_short_en']
# test_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en', 'Mams_en', 'Mams_short_en']

training_datasets = ['Rest16_en_merged', 'Rest16_es_merged', 'Rest16_ru_merged', 'Lap14_en_merged']
test_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en']


if __name__ == '__main__':
    #### For evaluating spanbert
    for dtrain in training_datasets:
        for dtest in test_datasets:
            print("EVALUATING - train: {}, test: {}".format(dtrain, dtest))
            predicted_data, gold_data = read_transformed_targets(TRANSFORMED_TARGETS_PREDICTIONS_FILE.format(dtrain, dtest))
            # print(evaluate_ote(gold_data, predicted_data))
            predicted_data, gold_data = read_transformed_sentiments(TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE.format(dtrain, dtest))
            print(evaluate_ts(gold_data, predicted_data))

    #### For dummy
    # for dtest in datasets:
    #     print("EVALUATING - test: {}".format(dtest))
    #     predicted_data, gold_data = read_transformed_targets(TRANSFORMED_TARGETS_PREDICTIONS_FILE.format(dtest))
    #     predicted_data, gold_data = read_transformed_sentiments(TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE.format(dtest))
    #     print(evaluate_ts(gold_data, predicted_data))