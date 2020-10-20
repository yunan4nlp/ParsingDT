import sys

sys.path.extend(["../../", "../", "./"])
import itertools
import argparse
from data.Config import *
from data.Dataloader import *
from data.Vocab import *
import pickle
import time
from BiaffineParser_NoPOS.load_parser import *


def batch_doc_variable(onebatch, vocab):
    batch_size = len(onebatch)
    max_sent_num = max([len(doc.sentences) for doc in onebatch])
    max_sent_len = max([len(sentence) for doc in onebatch for sentence in doc.sentences]) + 1# add pseudo root length
    words = np.zeros((batch_size, max_sent_num, max_sent_len), dtype=int)
    extwords = np.zeros((batch_size, max_sent_num, max_sent_len), dtype=int)
    word_masks = np.zeros((batch_size, max_sent_num, max_sent_len), dtype=int)

    doc_lengths = []
    for idx, doc in enumerate(onebatch):
        lengths = []
        for idy, sentence in enumerate(doc.sentences):
            parsing_sentence = [vocab._root_form] + sentence  # add pseudo root
            lengths.append(len(parsing_sentence))
            assert len(parsing_sentence) > 1
            for idz, word in enumerate(parsing_sentence):
                words[idx, idy, idz] = vocab.word2id(word)
                extwords[idx, idy, idz] = vocab.extword2id(word)
                word_masks[idx, idy, idz] = 1
        doc_lengths.append(lengths)

    words = torch.tensor(words, dtype=torch.long)
    extwords = torch.tensor(extwords, dtype=torch.long)
    word_masks = torch.tensor(word_masks, dtype=torch.float)

    return words, extwords, doc_lengths, word_masks


def batch_variable(onebatch, vocab):
    batch_size = len(onebatch)
    max_sent_len = max([len(sent) for sent in onebatch]) + 1  # add pseudo root length
    words = np.zeros((batch_size, max_sent_len), dtype=int)
    extwords = np.zeros((batch_size, max_sent_len), dtype=int)
    word_masks = np.zeros((batch_size, max_sent_len), dtype=int)

    lengths = []
    for idx, sentence in enumerate(onebatch):
        parsing_sentence = [vocab._root] + sentence  # add pseudo root
        lengths.append(len(parsing_sentence))
        assert len(parsing_sentence) > 1
        for idy, word in enumerate(parsing_sentence):
            words[idx, idy] = vocab.word2id(word)
            extwords[idx, idy] = vocab.extword2id(word)
            word_masks[idx, idy] = 1

    words = torch.tensor(words, dtype=torch.long)
    extwords = torch.tensor(extwords, dtype=torch.long)
    word_masks = torch.tensor(word_masks, dtype=torch.float)

    return words, extwords, lengths, word_masks


def test(test_data, parser, parser_vocab, parser_config, outputFile):
    outf = open(outputFile, mode='w', encoding='UTF8')
    parsing_sent = 0
    with torch.no_grad():
        parser.model.eval()
        start_time = time.time()
        for onebatch in data_iter(test_data, parser_config.test_batch_size, False):
            words, extwords, doc_lengths, word_masks = batch_doc_variable(onebatch, parser_vocab)
            doc_arcs_batch, doc_rels_batch = parser.parse_doc(words, extwords, doc_lengths, word_masks)

            sentence_num = 0
            for idx, doc in enumerate(onebatch):
                print("paring each sentence in doc: ", doc.firstline)
                arcs_batch = doc_arcs_batch[idx]
                rels_batch = doc_rels_batch[idx]
                outf.write(doc.firstline + '\n')
                for idx, sentence_conll in enumerate(doc.sentences_conll):
                    predict_arcs = list(arcs_batch[idx])[1:]  # delete pseudo root
                    rels = list(rels_batch[idx])[1:]  # delete pseudo relation
                    assert len(predict_arcs) == len(sentence_conll)
                    assert len(rels) == len(sentence_conll)
                    predict_rels = parser_vocab.id2rel(rels)

                    for idx, line in enumerate(sentence_conll):
                        info = line.strip().split("\t")
                        info[7] = str(predict_arcs[idx])
                        info[8] = str(predict_rels[idx])
                        predict_line = '\t'.join(info)
                        outf.write(predict_line + '\n')
                    outf.write('\n')
                sentence_num += len(doc.sentences)
            parsing_sent += sentence_num
            cost_time = time.time() - start_time
            progress = parsing_sent / parser_config.total_sent_num * 100
            print("parsing time: %.2fs, avg: %.0f sent/s, parsing num: %d, progress：%.2f"
                  %(cost_time, parsing_sent / cost_time, parsing_sent, progress))
    outf.close()
    return 0


if __name__ == '__main__':
    ### process id
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--parser_config_file', default='examples/default.cfg')
    argparser.add_argument('--test_file', default='')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()

    parser_model, parser_vocab, parser_config = load_parser(args, extra_args)

    torch.set_num_threads(args.thread)

    use_cuda = False
    if gpu and args.use_cuda: use_cuda = True
    print("\nGPU using status: ", args.use_cuda)

    if use_cuda:
        torch.backends.cudnn.enabled = True
        parser_model.cuda()

    test_data, total_sent_num = read_eval_corpus(args.test_file)
    parser_config.total_sent_num = total_sent_num

    parser = BiaffineParser(parser_model, parser_vocab.ROOT)

    if use_cuda:
        print("Using device: ", parser.device)
    else:
        print("Using device: cpu")



    test(test_data, parser, parser_vocab, parser_config, args.test_file + '.out')

    print("Parsing is done!")
