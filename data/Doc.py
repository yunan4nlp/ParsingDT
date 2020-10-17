class Doc:
    def __init__(self):
        self.file_name = ""
        self.sentences_conll = []

        self.sentences = []
        self.sentences_labels = []

        self.EDUs = []

    def extract_conll(self):
        for sentence_conll in self.sentences_conll:
            sentence = []
            sentence_labels = []
            for line in sentence_conll:
                info = line.split('\t')
                assert len(info) == 10
                sentence.append(info[1].lower())
                if info[9] == 'BeginSeg=Yes':
                    sentence_labels.append('b')
                else:
                    sentence_labels.append('i')
            self.sentences.append(sentence)
            self.sentences_labels.append(sentence_labels)
        self.split_EDUs()

    def split_EDUs(self):

        for idx, sentence_label in enumerate(self.sentences_labels):
            start = 0
            for idy in range(len(sentence_label)):
                if idy + 1 < len(sentence_label) and sentence_label[idy + 1] == "b":
                    end = idy
                    EDU = self.sentences[idx][start:end + 1]
                    self.EDUs.append(EDU)
                    start = end + 1
                elif idy + 1 == len(sentence_label):
                    end = idy
                    EDU = self.sentences[idx][start:end + 1]
                    self.EDUs.append(EDU)

        EDU_total_len = 0
        for EDU in self.EDUs:
            EDU_total_len += len(EDU)
        sent_total_len = 0
        for sentence in self.sentences:
            sent_total_len += len(sentence)
        assert EDU_total_len == sent_total_len

    def extract_EDUstr(self):
        doc_labels = []
        for sentence_labels in self.sentences_labels:
            doc_labels += sentence_labels
        start = 0
        EDUs = []
        for idx in range(len(doc_labels)):
            if idx + 1 < len(doc_labels) and doc_labels[idx + 1] == "b":
                end = idx
                EDUs.append('(' + str(start) + ',' + str(end) + ')')
                start = idx + 1
            elif idx + 1 == len(doc_labels):
                end = idx
                EDUs.append('(' + str(start) + ',' + str(end) + ')')
        return EDUs

class Metric:
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and \
                    self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def getAccuracy(self):
        if self.overall_label_count + self.predicated_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return self.correct_label_count*1.0 / self.overall_label_count
        else:
            return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print(self):
        if self.predicated_label_count == 0:
            print("Accuracy:\tP=" + str(self.correct_label_count) + '/' + str(self.overall_label_count))
        else:
            print("Recall:\tP=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count), end=",\t")
            print("Accuracy:\tP=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count), end=",\t")
            print("Fmeasure:\t" + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))
