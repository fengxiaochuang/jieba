# encoding=utf-8
from __future__ import absolute_import
import os
import jieba
import jieba.posseg
from operator import itemgetter

_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                              os.path.dirname(__file__), path))
_get_abs_path = jieba._get_abs_path

DEFAULT_IDF = _get_module_path("idf.txt")


class KeywordExtractor(object):
    # 停用词
    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def set_stop_words(self, stop_words_path):
        """手动设置停用词词典"""
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader(object):
    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}  # idf词频统计
        self.median_idf = 0.0  # 平均词频，中位数，为了确定矩阵中心
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        """手动指定tf-idf词频词典"""
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]  # 排并取出中位数

    def get_idf(self):
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):
    def __init__(self, idf_path=None):
        self.tokenizer = jieba.dt  # 结巴分词器
        self.postokenizer = jieba.posseg.dt  # 带词性的分词器
        self.stop_words = self.STOP_WORDS.copy()
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        用TF-IDF算法抽取关键词
        参数:
            - topK: 返回多少个关键词，None为不限制。
            - withWeight: 如果是True, 返回一个列表，结构为：[('服务', 2.671075298535)];
                          如果是False, 返回所有word的列表，结构为['服务', '商品']。
            - allowPOS: 允许词性的列表. ['ns', 'n', 'vn', 'v','nr'].
                        如果一个词的词性不在这个列表里面，那么这个词会被过滤掉。
            - withFlag: 只有当allowPOS不为空的时候才会生效。
                        如果是True, 返回一个列表，结构为[(pair('服务', 'vn'), 2.671075298535)]，参考posseg.cut分词器
                        如果是False, 返回word的列表
        """
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            words = self.tokenizer.cut(sentence)
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0  # 统计这句话中的词频
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            # 这句话中的这个词语的权重 = 词频 * （该词语的权重/总词语数量）
            # 此算法还待论证，等我比较完其他TF-IDF再添加注释
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            return tags[:topK]
        else:
            return tags
