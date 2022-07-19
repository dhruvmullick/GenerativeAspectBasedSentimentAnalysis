import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import random

MAMS_SHORTENED_TEXT_LENGTH = 125
# lowest possible training record count in the considered datasets.
# SAMPLED_RECORD_COUNT = 857
WEIRD_CHARACTERS = '.!?,'

def join_sentence_and_annotations(words, annotations):
    sentence = ''
    for w, a in zip(words, annotations):
        weirdchar = ''
        if w[-1] in WEIRD_CHARACTERS:
            weirdchar = w[-1]
            w = w[:-1]
        new_word = w + '=' + a
        sentence = sentence + ' ' + new_word
        if weirdchar:
            new_word = weirdchar + '=O'
            sentence = sentence + ' ' + new_word

    return sentence.strip()


def write_to_file(array, path):
    with open(path, 'w') as file:
        file.write('\n'.join(array) + '\n')


def get_annotations_for_sentence(sentence_text, opinions):
    words = sentence_text.split()
    annotations = ['O'] * len(words)
    for i, row in opinions.iterrows():
        annotation = 'O'
        if row['pol'] == 'positive':
            annotation = 'T-POS'
        if row['pol'] == 'negative':
            annotation = 'T-NEG'
        if row['pol'] == 'neutral':
            annotation = 'T-NEU'
        if row['pol'] == 'conflict':
            continue
        word_count = 0
        for j in range(len(sentence_text)):
            if sentence_text[j].isspace() and j > 0 and not sentence_text[j - 1].isspace():
                word_count += 1
            if j >= int(row['from']) and j < int(row['to']):
                annotations[word_count] = annotation
            if j >= int(row['to']):
                break
    return annotations, words


def parse_semeval_xml(root):
    data = []
    datalist_span_bert = []
    for i, child in enumerate(root):
        review_id = child.attrib['rid']

        for sentence in child.findall('sentences/sentence'):
            sentence_id = "_".join(['id', sentence.attrib['id'].split(':')[1]])
            if not sentence.findall('text')[0].text:
                continue
            sentence_text = sentence.findall('text')[0].text.strip()
            if len(sentence.findall('text')) != 1:
                raise Exception("NOT EXACTLY ONE TEXT")
            if len(sentence.findall('Opinions')) > 1:
                raise Exception("MORE THAN ONE OPINIONS")
            if len(sentence.findall('Opinions')) == 0:
                continue
            opinions = []
            sentence_opinions = sentence.findall('Opinions')[0]
            for opinion in sentence_opinions.findall('Opinion'):
                if opinion.attrib['polarity'] == 'conflict':
                    continue
                opinions.append([opinion.attrib['target'], opinion.attrib['polarity'], opinion.attrib['from'],
                                 opinion.attrib['to']])

            if len(opinions) == 0:
                continue

            opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
            opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)

            annotations, words = get_annotations_for_sentence(sentence_text, opinions)

            opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]

            data.append([review_id, sentence_id, sentence_text, opinions])
            spanbert_text = '{}####{}'.format(sentence_text, join_sentence_and_annotations(words, annotations))
            datalist_span_bert.append(spanbert_text)

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data, datalist_span_bert


def parse_semeval_14_xml(root):
    data = []
    datalist_span_bert = []
    for i, child in enumerate(root):
        review_id = i
        sentence_id = child.attrib['id']
        sentence_text = child.find('text').text
        aspect_term_node = child.find('aspectTerms')
        if not aspect_term_node:
            continue

        opinions = []

        for aspectTerm in aspect_term_node:
            term = aspectTerm.attrib['term']
            polarity = aspectTerm.attrib['polarity']
            if polarity == 'conflict':
                continue
            from_idx = aspectTerm.attrib['from']
            to_idx = aspectTerm.attrib['to']
            term = term.strip()
            polarity = polarity.strip()

            if term and polarity:
                opinions.append([term, polarity, from_idx, to_idx])

        if len(opinions) == 0:
            continue

        opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
        opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)

        annotations, words = get_annotations_for_sentence(sentence_text, opinions)

        opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]
        data.append([review_id, sentence_id, sentence_text, opinions])
        spanbert_text = '{}####{}'.format(sentence_text, join_sentence_and_annotations(words, annotations))
        datalist_span_bert.append(spanbert_text)

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data, datalist_span_bert


def load_semeval(train_file, test_file):
    xml_train = open(train_file, 'r').read()  # Read file
    train, train_spanbert = parse_semeval_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test, test_spanbert = parse_semeval_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    non_null_train_idx = ~train['sentences_opinions'].str.contains('NULL')
    train = train[non_null_train_idx]
    train_spanbert = np.array(train_spanbert)[non_null_train_idx.array]
    non_null_test_idx = ~test['sentences_opinions'].str.contains('NULL')
    test = test[non_null_test_idx]
    test_spanbert = np.array(test_spanbert)[non_null_test_idx.array]

    print('Semeval: ', train.shape, test.shape)

    train, val, train_spanbert, val_spanbert = train_test_split(train, train_spanbert, test_size=0.1, random_state=0)
    return train, train_spanbert, val, val_spanbert, test, test_spanbert


def load_semeval_14(train_file, test_file):
    xml_train = open(train_file, 'r').read()  # Read file
    train, train_spanbert = parse_semeval_14_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test, test_spanbert = parse_semeval_14_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    non_null_train_idx = ~train['sentences_opinions'].str.contains('NULL')
    train = train[non_null_train_idx]
    train_spanbert = np.array(train_spanbert)[non_null_train_idx.array]
    non_null_test_idx = ~test['sentences_opinions'].str.contains('NULL')
    test = test[non_null_test_idx]
    test_spanbert = np.array(test_spanbert)[non_null_test_idx.array]

    print('Semeval: ', train.shape, test.shape)

    train, val, train_spanbert, val_spanbert = train_test_split(train, train_spanbert, test_size=0.1, random_state=0)
    return train, train_spanbert, val, val_spanbert, test, test_spanbert


def parse_MAMS_xml(root, shortened):
    data = []
    datalist_span_bert = []
    for i, child in enumerate(root):
        sentence_text = child.findall('text')[0].text.strip()
        if shortened and (len(sentence_text) > MAMS_SHORTENED_TEXT_LENGTH):
            continue

        opinions = []

        for aspectTerm in child.findall('aspectTerms')[0]:
            term = aspectTerm.attrib['term']
            polarity = aspectTerm.attrib['polarity']
            from_idx = aspectTerm.attrib['from']
            to_idx = aspectTerm.attrib['to']
            term = term.strip()
            polarity = polarity.strip()
            if polarity == 'conflict':
                continue
            if term and polarity:
                opinions.append([term, polarity, from_idx, to_idx])

        opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
        opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)

        if len(opinions) == 0:
            continue

        annotations, words = get_annotations_for_sentence(sentence_text, opinions)

        opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]
        data.append([0, 0, sentence_text, opinions])

        spanbert_text = '{}####{}'.format(sentence_text, join_sentence_and_annotations(words, annotations))
        datalist_span_bert.append(spanbert_text)

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data, datalist_span_bert


def load_MAMS(train_file, val_file, test_file, shortened):
    xml_train = open(train_file, 'r').read()  # Read file
    train, train_spanbert = parse_MAMS_xml(root=ET.XML(xml_train), shortened=shortened)
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_val = open(val_file, 'r').read()  # Read file
    val, val_spanbert = parse_MAMS_xml(root=ET.XML(xml_val), shortened=shortened)
    val['sentences_opinions'] = val['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test, test_spanbert = parse_MAMS_xml(root=ET.XML(xml_test), shortened=shortened)
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    print('MAMS: ', train.shape, val.shape, test.shape)
    return train, train_spanbert, val, val_spanbert, test, test_spanbert


def opinions_to_decoder_format(opinions_list):
    return ' <sep> '.join(opinions_list)


load_dataset = {
    ('Rest16', 'en'): (load_semeval, {'train_file': 'data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml',
                                      'test_file': 'data/semeval/test/EN_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'fr'): (load_semeval, {'train_file': 'data/semeval/training/ABSA16FR_Restaurants_Train-withcontent.xml',
                                      'test_file': 'data/semeval/test/ABSA16FR_Restaurants_Gold-withcontent.xml'}),
    ('Rest16', 'nl'): (load_semeval, {'train_file': 'data/semeval/training/restaurants_dutch_training.xml',
                                      'test_file': 'data/semeval/test/DU_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'es'): (load_semeval, {'train_file': 'data/semeval/training/SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml',
                                      'test_file': 'data/semeval/test/SP_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'ru'): (load_semeval, {'train_file': 'data/semeval/training/se16_ru_rest_train.xml',
                                      'test_file': 'data/semeval/test/RU_REST_SB1_TEST.xml.gold'}),
    ('Rest15', 'en'): (load_semeval, {'train_file': 'data/semeval-2015/ABSA-15_Restaurants_Train_Final.xml',
                                      'test_file': 'data/semeval-2015/ABSA15_Restaurants_Test.xml'}),
    ('Rest14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Restaurants_Train.xml',
                                         'test_file': 'data/semeval-2014/Restaurants_Test_Gold.xml'}),
    ('Lap14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Laptops_Train.xml',
                                        'test_file': 'data/semeval-2014/Laptops_Test_Gold.xml'}),
    ('Mams', 'en'): (load_MAMS, {'train_file': 'data/MAMS_ATSA/train.xml', 'val_file': 'data/MAMS_ATSA/val.xml',
                                 'test_file': 'data/MAMS_ATSA/test.xml', 'shortened': False}),
    ('Mams_short', 'en'): (load_MAMS, {'train_file': 'data/MAMS_ATSA/train.xml', 'val_file': 'data/MAMS_ATSA/val.xml',
                                       'test_file': 'data/MAMS_ATSA/test.xml', 'shortened': True}),
}


def preprocess_dataset(domain, language):
    random.seed(0)
    np.random.seed(0)

    print("Processing the dataset for {}.{}".format(domain, language))
    if not load_dataset.get((domain, language)):
        raise Exception("domain language combination not defined")
    method = load_dataset[(domain, language)][0]
    args = load_dataset[(domain, language)][1]
    train, train_spanbert, val, val_spanbert, test, test_spanbert = method(**args)

    assert len(train) == len(train_spanbert)
    assert len(val) == len(val_spanbert)
    assert len(test) == len(test_spanbert)

    # rows = random.sample(np.arange(0, len(train_spanbert)).tolist(), SAMPLED_RECORD_COUNT)
    rows = random.sample(np.arange(0, len(train_spanbert)).tolist(), len(train_spanbert))

    train = train.iloc[rows, :]
    train_spanbert = np.array(train_spanbert)[rows]

    test_idx = test['sentences_opinions'] != ''
    test = test[test_idx]
    test_spanbert = np.array(test_spanbert)[test_idx]

    train.to_csv('data/processed_full_train_{}_{}.csv'.format(domain, language), header=True, index=False)
    val.to_csv('data/processed_val_{}_{}.csv'.format(domain, language), header=True, index=False)
    test.to_csv('data/processed_test_{}_{}.csv'.format(domain, language), header=True, index=False)

    write_to_file(train_spanbert, 'data/train_full_spanbert_{}_{}.csv'.format(domain, language))
    write_to_file(val_spanbert, 'data/val_spanbert_{}_{}.csv'.format(domain, language))
    write_to_file(test_spanbert, 'data/test_spanbert_{}_{}.csv'.format(domain, language))


if __name__ == '__main__':
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'en')
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'es')
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'ru')
    # MAMS
    preprocess_dataset('Mams', 'en')
    # MAMS_Shortened
    preprocess_dataset('Mams_short', 'en')
    # Semeval Laptop 2014
    preprocess_dataset('Lap14', 'en')

    print('saved..')
