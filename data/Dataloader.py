import numpy as np
import torch
from torch.autograd import Variable
from data.Doc import *
import random
import copy
from collections import Counter
from scipy import stats
import numpy as np


def read_sent(inf):
    sentence = []
    for line in inf:
        line = line.strip()
        if line == '':
            yield sentence
            sentence = []
        else:
            sentence.append(line)
    if len(sentence) > 0:
        yield sentence

def read_eval_corpus(file_path):

    with open(file_path, mode='r', encoding='utf8') as inf:
        doc_data = []
        for inst in read_sent(inf):
            if inst[0].find("# newdoc id =") == 0:
                doc_name = inst[0].split('=')[1].strip()
                doc = Doc()
                doc.firstline = inst[0]
                doc.name = doc_name
                doc.sentences_conll.append(inst[1:])
                doc_data.append(doc)
            else:
                doc.sentences_conll.append(inst)
    filter_doc_data = []
    for doc in doc_data:
        doc.extract_conll()
        filter_doc_data.append(doc)
    doc_num = len(filter_doc_data)
    sent_num = 0
    for doc in filter_doc_data:
        sent_num += len(doc.sentences)
    print("Info: ", file_path)
    print("Doc num: ", doc_num)
    print("Sentence num: ", sent_num)
    return filter_doc_data

def read_corpus(file_path, min_edu_num, max_edu_num, eval=False):

    with open(file_path, mode='r', encoding='utf8') as inf:
        doc_data = []
        for inst in read_sent(inf):
            if inst[0].find("# newdoc id =") == 0:
                doc_name = inst[0].split('=')[1].strip()
                doc = Doc()
                doc.firstline = inst[0]
                doc.name = doc_name
                doc.sentences_conll.append(inst[1:])
                doc_data.append(doc)
            else:
                doc.sentences_conll.append(inst)
    filter_doc_data = []
    for doc in doc_data:
        doc.extract_conll()
        if len(doc.EDUs) >= min_edu_num and len(doc.EDUs) < max_edu_num:
            filter_doc_data.append(doc)
    doc_num = len(filter_doc_data)
    sent_num = 0
    for doc in filter_doc_data:
        sent_num += len(doc.sentences)
    if not eval:
        print("Info: ", file_path)
        print("Doc num: ", doc_num)
        print("Sentence num: ", sent_num)
    return filter_doc_data

def mask_edu(data, config, tokenizer):
    X = np.arange(1, config.geo_clip + 1, 1)
    pList = stats.geom.pmf(X, config.geo_p)
    filter_data = []
    MASK = tokenizer.key_words[4]
    masked_word_counter = Counter()
    for inst in data:
        inst.mask_indexes = []
        for EDU_index, EDU in enumerate(inst.EDUs):
            index = len(EDU) - 1
            if index < config.geo_clip:
                p = pList[index]
                rand_p = random.random()
                if rand_p < p:
                    inst.mask_indexes.append(EDU_index)

        inst.masked_EDUs = copy.deepcopy(inst.EDUs)
        for idx in inst.mask_indexes:
            EDU_len = len(inst.EDUs[idx])
            for idy in range(EDU_len):
                masked_word_counter[inst.masked_EDUs[idx][idy]] += 1
                inst.masked_EDUs[idx][idy] = MASK

        inst.masked_doc = []
        for edu in inst.masked_EDUs:
            inst.masked_doc += edu
        count = 0
        inst.masked_sentences = copy.deepcopy(inst.sentences)
        for idx, sentence in enumerate(inst.masked_sentences):
            for idy, word in enumerate(sentence):
                if inst.masked_doc[count] == MASK:
                    inst.masked_sentences[idx][idy] = MASK
                else:
                    assert word == inst.sentences[idx][idy]
                count += 1

        if len(inst.mask_indexes) > 0:
            filter_data.append(inst)
    return masked_word_counter, filter_data

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences



def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_variable_label(batch, vocab):
    max_sent_len = max([len(data[0]) for data in batch])
    gold_labels_id = []
    for idx, data in enumerate(batch):
        labels = data[1][:max_sent_len]
        label_ids = vocab.conj2id(labels)
        label_index = np.ones(len(label_ids), dtype=np.int32) * -1
        for idx, index in enumerate(label_ids):
            label_index[idx] = index
        gold_labels_id.append(label_index)
    return gold_labels_id

def batch_pretrain_variable_sent_level(batch, vocab, config, tokenizer):
    batch_size = len(batch)
    max_bert_len = -1
    max_sent_num = max([len(data.sentences) for data in batch])
    max_sent_len = max([len(sent) for data in batch for sent in data.sentences])

    batch_bert_indices = []
    batch_segments_ids = []
    batch_piece_ids = []
    for data in batch:
        sents = data.masked_sentences
        doc_bert_indices = []
        doc_semgents_ids = []
        doc_piece_ids = []
        for sent in sents:
            sent = sent[:max_sent_len]
            bert_indice, segments_id, piece_id = tokenizer.bert_ids(' '.join(sent))
            doc_bert_indices.append(bert_indice)
            doc_semgents_ids.append(segments_id)
            doc_piece_ids.append(piece_id)
            assert len(piece_id) == len(sent)
            assert len(bert_indice) == len(segments_id)
            bert_len = len(bert_indice)
            if bert_len > max_bert_len: max_bert_len = bert_len
        batch_bert_indices.append(doc_bert_indices)
        batch_segments_ids.append(doc_semgents_ids)
        batch_piece_ids.append(doc_piece_ids)
    bert_indice_input = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_mask = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_segments_ids = np.zeros((batch_size, max_sent_num, max_bert_len), dtype=int)
    bert_piece_ids = np.zeros((batch_size, max_sent_num, max_sent_len, max_bert_len), dtype=float)

    for idx in range(batch_size):
        doc_bert_indices = batch_bert_indices[idx]
        doc_semgents_ids = batch_segments_ids[idx]
        doc_piece_ids = batch_piece_ids[idx]
        sent_num = len(doc_bert_indices)
        assert sent_num == len(doc_semgents_ids)
        for idy in range(sent_num):
            bert_indice = doc_bert_indices[idy]
            segments_id = doc_semgents_ids[idy]
            bert_len = len(bert_indice)
            piece_id = doc_piece_ids[idy]
            sent_len = len(piece_id)
            assert sent_len <= bert_len
            for idz in range(bert_len):
                bert_indice_input[idx, idy, idz] = bert_indice[idz]
                bert_segments_ids[idx, idy, idz] = segments_id[idz]
                bert_mask[idx, idy, idz] = 1
            for idz in range(sent_len):
                for sid, piece in enumerate(piece_id):
                    avg_score = 1.0 / (len(piece))
                    for tid in piece:
                        bert_piece_ids[idx, idy, sid, tid] = avg_score


    bert_indice_input = torch.from_numpy(bert_indice_input)
    bert_segments_ids = torch.from_numpy(bert_segments_ids)
    bert_piece_ids = torch.from_numpy(bert_piece_ids).type(torch.FloatTensor)
    bert_mask = torch.from_numpy(bert_mask)

    return bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask

def batch_data_variable(batch, vocab, config):
    batch_size = len(batch)
    max_edu_num = max([len(data.EDUs) for data in batch])
    max_edu_len = max([len(edu) for data in batch for edu in data.EDUs])
    if max_edu_len > config.max_edu_len: max_edu_len = config.max_edu_len

    word_mask = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    word_denominator = np.ones((batch_size, max_edu_num), dtype=int) * -1
    edu_mask = np.zeros((batch_size, max_edu_num), dtype=int)

    for idx in range(batch_size):
        EDUs = batch[idx].EDUs
        edu_num = len(EDUs)
        for idy in range(edu_num):
            edu = EDUs[idy]
            edu_len = len(edu)
            if edu_len > config.max_edu_len: edu_len = config.max_edu_len
            edu_mask[idx, idy] = 1
            word_denominator[idx, idy] = edu_len
            for idz in range(edu_len):
                word_mask[idx, idy, idz] = 1

    word_mask = torch.tensor(word_mask, dtype=torch.float)
    word_denominator = torch.tensor(word_denominator, dtype=torch.float)
    edu_mask = torch.tensor(edu_mask, dtype=torch.float)
    label_mask = edu_mask[:, 1:]

    return word_mask, word_denominator, edu_mask, label_mask

def batch_sent2span_offset(batch, config):
    batch_size = len(batch)
    max_sent_len = max([len(sent) for data in batch for sent in data.sentences])
    max_edu_num = max([len(data.EDUs) for data in batch])
    max_edu_len = max([len(EDU) for data in batch for EDU in data.EDUs])
    if config.max_edu_len < max_edu_len: max_edu_len = config.max_edu_len
    index = np.ones((batch_size, max_edu_num, max_edu_len), dtype=int) * (max_sent_len)
    for idx in range(batch_size):
        data = batch[idx]
        sentences = data.sentences
        sent_index = []
        for sent_idx, sentence in enumerate(sentences):
            sent_len = len(sentence)
            for sent_idy in range(sent_len):
                sent_index.append(sent_idx * (max_sent_len + 1) + sent_idy)
        edus = data.EDUs
        id = 0
        edu_num = len(edus)
        for idy in range(edu_num):
            edu = edus[idy]
            edu_len = len(edu[:config.max_edu_len])
            for idz in range(edu_len):
                index[idx, idy, idz] = sent_index[id]
                id += 1
    index = torch.from_numpy(index).view(batch_size, max_edu_num, max_edu_len)
    return index

def batch_label_variable(batch, vocab):
    batch_size = len(batch)

    max_maskededu_num = max([len(data.mask_indexes) for data in batch])

    max_maskededu_len = max([len(data.EDUs[index]) for data in batch for index in data.mask_indexes])

    max_edu_num = max([len(data.EDUs) for data in batch])

    masked_word_indexs = np.ones((batch_size, max_maskededu_num, max_maskededu_len), dtype=int) * -1

    edu_p_indexs = np.ones((batch_size, max_maskededu_num), dtype=int) * max_edu_num
    edu_b_indexs = np.ones((batch_size, max_maskededu_num), dtype=int) * max_edu_num

    for idx in range(batch_size):
        data = batch[idx]
        for idy, index in enumerate(data.mask_indexes):
            maskedword_ids = vocab.maskedword2id(data.EDUs[index])
            offset = idx * (max_edu_num + 1) + index

            if index - 1 >= 0:
                edu_p_indexs[idx, idy] = offset - 1

            if index + 1 < max_edu_num:
                edu_b_indexs[idx, idy] = offset + 1

            for idz, id in enumerate(maskedword_ids):
                masked_word_indexs[idx, idy, idz] = id
    masked_word_indexs = torch.tensor(masked_word_indexs, dtype=torch.long)
    masked_position_ids = torch.arange(max_maskededu_len, dtype=torch.long)
    edu_p_indexs = torch.tensor(edu_p_indexs, dtype=torch.long)
    edu_b_indexs = torch.tensor(edu_b_indexs, dtype=torch.long)

    return masked_word_indexs, masked_position_ids, edu_p_indexs, edu_b_indexs