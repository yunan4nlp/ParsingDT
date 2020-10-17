import sys
sys.path.extend(["../../","../","./"])
import itertools
import argparse
from data.Config import *
from data.Dataloader import *
from data.Vocab import *
import pickle
import time
from BiaffineParser_NoPOS.load_parser import *


def batch_variable(onebatch, vocab):
    batch_size = len(onebatch)
    max_sent_len = max([len(sent) + 1 for sent in onebatch])

    words = np.zeros((batch_size, max_sent_len), dtype=int)
    extwords = np.zeros((batch_size, max_sent_len), dtype=int)
    word_masks = np.zeros((batch_size, max_sent_len), dtype=int)

    lengths = []
    for idx, sentence in enumerate(onebatch):
        lengths.append(len(sentence))
        words[idx, 0] = vocab.ROOT
        extwords[idx, 0] = vocab.ROOT
        word_masks[idx, 0] = 1
        for idy, word in enumerate(sentence):
            words[idx, idy + 1] = vocab.word2id(word)
            extwords[idx, idy + 1] = vocab.extword2id(word)
            word_masks[idx, idy + 1] = 1

    words = torch.tensor(words, dtype=torch.long)
    extwords = torch.tensor(extwords, dtype=torch.long)
    word_masks = torch.tensor(word_masks, dtype=torch.float)

    return words, extwords, lengths, word_masks

def test(test_data, parser, parser_vocab, parser_config, outputFile):
    outf = open(outputFile, mode='w', encoding='UTF8')
    with torch.no_grad():
        parser.model.eval()
        for onebatch in data_iter(test_data, parser_config.test_batch_size, True):

            for doc in onebatch:
                words, extwords, lengths, word_masks = batch_variable(doc.sentences, parser_vocab)
                arcs_batch, rels_batch = parser.parse(words, extwords, lengths, word_masks)
                outf.write(doc.firstline + '\n')
                for idx, sentence_conll in enumerate(doc.sentences_conll):

                    predict_arcs = list(arcs_batch[idx])
                    rels = list(rels_batch[idx])

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

    args.use_cuda = False
    if gpu and args.use_cuda: args.use_cuda = True
    print("\nGPU using status: ", args.use_cuda)

    test_data = read_eval_corpus(args.test_file)

    parser = BiaffineParser(parser_model, parser_vocab.ROOT)

    test(test_data, parser, parser_vocab, parser_config, args.test_file + '.out')

