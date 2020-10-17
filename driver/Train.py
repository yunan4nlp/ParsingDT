import sys
sys.path.extend(["../../","../","./"])
import itertools
import argparse
from data.Config import *
from data.Dataloader import *
from data.Vocab import *
from modules.BertModel import *
from modules.BertTokenHelper import *
import pickle
from modules.WordLSTM import *
from modules.PretrainedWordEncoder import *
from modules.EDULSTM import *
from modules.Sent2Span import *
from modules.MaskedEDU import *
from modules.Decoder import *
import time

class Optimizer:
    def __init__(self, parameter, config, lr):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon, weight_decay=config.l2_reg)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()
    def schedule(self):
        self.scheduler.step()
    def zero_grad(self):
        self.optim.zero_grad()
    @property
    def lr(self):
        return self.scheduler.get_lr()

def train(train_data, edupred, vocab, config, tokenizer):

    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             edupred.pwordEnc.parameters(),
                             edupred.sent2span.parameters(),
                             edupred.wordLSTM.parameters(),
                             edupred.EDULSTM.parameters(),
                             edupred.dec.parameters()
                         )
                         )

    model_optimizer = Optimizer(model_param, config, config.learning_rate)

    global_step = 0
    best_FF = 0
    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_label_correct,  overall_total_label = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, True):

            bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = \
                batch_pretrain_variable_sent_level(onebatch, vocab, config, tokenizer)

            sent2span_index = batch_sent2span_offset(onebatch, config)

            word_mask, word_denominator, edu_mask, label_mask = \
            batch_data_variable(onebatch, vocab, config)

            masked_word_indexs, masked_position_ids, edu_p_indexs, edu_b_indexs = batch_label_variable(onebatch, vocab)

            edupred.train()
            #with torch.autograd.profiler.profile() as prof:
            edupred.encode(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask,
                           sent2span_index,
                           word_mask, word_denominator, edu_mask)

            batch_predict_indexs = edupred.decode(masked_position_ids, edu_p_indexs, edu_b_indexs)

            loss = edupred.compute_loss(masked_word_indexs)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_labels, correct_labels = edupred.compute_accuracy(batch_predict_indexs, masked_word_indexs)

            overall_total_label += total_labels
            overall_label_correct += correct_labels
            during_time = float(time.time() - start_time)
            acc = overall_label_correct / overall_total_label
            #acc = 0
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)
                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

        if config.save_after >= 0 and iter >= config.save_after:
            segmenter_model = {
                "pwordEnc": edupred.pwordEnc.state_dict(),
                "sent2span": edupred.sent2span.state_dict(),
                "wordLSTM": edupred.wordLSTM.state_dict(),
                "EDULSTM": edupred.EDULSTM.state_dict(),
                "dec": edupred.dec.state_dict(),
            }
            torch.save(segmenter_model, config.save_model_path + "." + str(global_step))
            print('Saving model to ', config.save_model_path + "." + str(global_step))

            '''
            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                predict(dev_data, conjpred, vocab, config, tokenizer, config.dev_file + '.' + str(global_step))
                dev_FF = scripts_evaluate(config, config.dev_file, config.dev_file + '.' + str(global_step))

                if dev_FF > best_FF:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_FF, dev_FF))
                    best_FF = dev_FF
                    if config.save_after >= 0 and iter >= config.save_after:
                        segmenter_model = {
                            "pwordEnc": conjpred.pwordEnc.state_dict(),
                            "wordLSTM": conjpred.wordLSTM.state_dict(),
                            "dec": conjpred.dec.state_dict(),
                            }
                        torch.save(segmenter_model, config.save_model_path + "." + str(global_step))
                        print('Saving model to ', config.save_model_path + "." + str(global_step))
            '''

def scripts_evaluate(config, gold_file, predict_file):
    cmd = "python %s %s %s" % (config.eval_scripts, gold_file, predict_file)
    F_exec = os.popen(cmd).read()
    info = F_exec.strip().split("\n")
    fscore = info[-1].split(': ')[-1]
    print(' '.join(info))
    return float(fscore)

def evaluate(gold_file, predict_file):
    gold_data = read_corpus(gold_file, True)
    predict_data = read_corpus(predict_file, True)
    seg_metric = Metric()
    for gold_doc, predict_doc in zip(gold_data, predict_data):
        gold_edus = gold_doc.extract_EDUstr()
        predict_edus = predict_doc.extract_EDUstr()
        seg_metric.overall_label_count += len(gold_edus)
        seg_metric.predicated_label_count += len(predict_edus)
        seg_metric.correct_label_count += len(set(gold_edus) & set(predict_edus))
    print("edu seg:", end=" ")
    seg_metric.print()
    return seg_metric.getAccuracy()

def predict(data, segmenter, vocab, config, tokenizer, outputFile):
    start = time.time()
    segmenter.eval()
    outf = open(outputFile, mode='w', encoding='utf8')
    for onebatch in data_iter(data, config.test_batch_size, False):
        bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = \
            batch_pretrain_variable_sent_level(onebatch, vocab, config, tokenizer)
        sent2span_index = batch_sent2span_offset(onebatch, config)
        word_mask, word_denominator, edu_mask, label_mask = \
            batch_data_variable(onebatch, vocab, config)
        edupred.encode(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask,
                       sent2span_index,
                       word_mask, word_denominator, edu_mask)
        batch_predict_indexs = edupred.decode(label_mask)

        print("OK")

    outf.close()
    end = time.time()
    during_time = float(end - start)
    print("doc num: %d, segment time = %.2f " % (len(data), during_time))

def parse_sentence_seg(sentences_seg, vocab):
    sentences_labels = []
    for sentence_seg in sentences_seg:
        sent_label = vocab.id2conj(sentence_seg)
        sent_label[0] = 'b'
        sentences_labels += sent_label
    start = 0
    EDUs_id = []
    for idx, label in enumerate(sentences_labels):
        if idx + 1 < len(sentences_labels) and sentences_labels[idx + 1] == 'b':
            end = idx
            EDUs_id.append([start, end])
            start = end + 1
        elif idx + 1 == len(sentences_labels):
            end = idx
            EDUs_id.append([start, end])
            start = end + 1
    return EDUs_id

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
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    print('Load pretrained encoder.....')
    tok = BertTokenHelper(config.bert_dir)
    enc_model = BertExtractor(config)
    print(enc_model)
    print('Load pretrained encoder ok')

    train_data = read_corpus(config.train_file, config.min_edu_num, config.max_edu_num)
    masked_word_counter, train_data = mask_edu(train_data, config, tok)

    print("Training doc: ", len(train_data))
    vocab = creatVocab(train_data, config, masked_word_counter)

    pwordEnc = PretrainedWordEncoder(config, enc_model, enc_model.bert_hidden_size, enc_model.layer_num)
    wordLSTM = WordLSTM(vocab, config)
    sent2span = Sent2Span(vocab, config)
    EDULSTM = EDULSTM(vocab, config)
    dec = Decoder(vocab, config)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    if config.use_cuda:
        pwordEnc.cuda()
        wordLSTM.cuda()
        sent2span.cuda()
        EDULSTM.cuda()
        dec.cuda()

    edupred = EDUPred(pwordEnc, wordLSTM, sent2span, EDULSTM, dec, config)

    train(train_data, edupred, vocab, config, tok)
