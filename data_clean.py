# 本脚本用于数据去重、关键词去除
import sys
import json
import functools
import tqdm
import numpy as np
from FlagEmbedding import FlagModel
from datasketch import MinHash, MinHashLSH
import shutil
import faiss

term_width = shutil.get_terminal_size(fallback=(80, 20)).columns
# 基于rouge-l的去重
class Rouge_l_Cleaner:
    @functools.lru_cache(maxsize=10000)  # 缓存最近的10000个计算结果
    def __init__(self, threshold):
        self.threshold = threshold

    def lcs_length(self, x, y):
        """
        计算两个字符串之间的最长公共子序列长度。
        """
        if not x or not y:
            return 0

        lx, ly = len(x), len(y)
        dp = [[0] * (ly + 1) for _ in range(lx + 1)]

        for i in range(1, lx + 1):
            for j in range(1, ly + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[lx][ly]

    def rouge_l(self, reference, candidate):
        """
        计算ROUGE-L分数。
        """
        lcs_len = self.lcs_length(reference, candidate)
        recall = lcs_len / len(reference)
        precision = lcs_len / len(candidate)
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0

        return {"recall": recall, "precision": precision, "f1": f1_score}

    def calculate_rouge_scores_zh(self, reference_sentence, candidate_sentences, top_n=10):
        scores = []
        for candidate_sentence in candidate_sentences:
            scores.append(self.rouge_l(reference_sentence, candidate_sentence)['f1'])
        scored_sentences = list(zip(candidate_sentences, scores))

        return scores
    # 太慢了，弃用
    def run_rouge_cleaner(self, input_data_list):
        unique_texts = []
 
        process_bar = tqdm.tqdm(total=len(input_data_list))
        caled_data = ['.']
        print('############ Beginging Run Rouge_l Cleaner ############'.center(term_width))
        for cur_data in input_data_list:
            rouge_scores = self.calculate_rouge_scores_zh(cur_data, caled_data)
            # print(max(rouge_scores))
            if max(rouge_scores) < self.threshold:
                unique_texts.append(cur_data)
                caled_data.append(cur_data)
            process_bar.update(1)
        return unique_texts
        

# 基于minhash去重
class MinHash_Cleaner:
    def __init__(self, threshold, sentence_bert_model_path) -> None:

        self.sentence_bert_model_path = sentence_bert_model_path
        self.threshold = threshold

    def sentence_bert_encode(self, orig_data_list, model_path):
        
        print('############ Beginging Sentence Bert Encoding ############'.center(term_width))
        data_list = []
        vector_list = []
        model = FlagModel(model_path, 
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关句子：",
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        progress_bar = tqdm.tqdm(total=len(orig_data_list))
        for index, i in enumerate(orig_data_list):
            progress_bar.update(1)
            data_list.append(i)
            vector_list.append(model.encode(i).tolist())
        vector_list = [np.array(i) for i in vector_list]
        return data_list, vector_list
    
    def create_minhash(self, encoded_vectors, num_perm=256):
        """
        为编码后的向量创建MinHash对象。
        """
        minhashes = []
        for vec in encoded_vectors:
            m = MinHash(num_perm=num_perm)
            for d in np.nditer(vec):
                m.update(d.tobytes())
            minhashes.append(m)
        return minhashes
    
    def detect_duplicates(self, texts, minhashes):
        """
        使用MinHashLSH检测重复文本。
        """
        
        lsh = MinHashLSH(threshold=self.threshold, num_perm=len(minhashes[0]))
        for i, m in enumerate(minhashes):
            lsh.insert(str(i), m)

        unique_texts = []
        processed = set()
        print('############ Beginging Run MinHashLSH Cleaner ############'.center(term_width))
        progress_bar = tqdm.tqdm(total=len(minhashes))
        for i, m in enumerate(minhashes):
            progress_bar.update(1)
            if str(i) in processed:
                continue
            result = lsh.query(m)
            processed.update(result)
            # print(processed)
            unique_texts.append(texts[i])
        
        return unique_texts
    
    def run_minhashLSH_cleaner(self, input_data_list):
        data_list, vector_list = self.sentence_bert_encode(input_data_list, self.sentence_bert_model_path)
        # 创建MinHash对象
        minhashes = self.create_minhash(vector_list)
        # 检测重复
        unique_texts = self.detect_duplicates(data_list, minhashes)
        
        return unique_texts

    
# 基于句向量相似度清洗, 将数据编码并简历faiss索引
class Embedding_Cleaner:
    def __init__(self, threshold, sentence_bert_model_path) -> None:
        self.sentence_bert_model_path = sentence_bert_model_path
        self.threshold = threshold

    def sentence_bert_encode(self, orig_data_list, model_path):
        print('############ Beginging Sentence Bert Encoding ############'.center(term_width))
        data_list = []
        vector_list = []
        model = FlagModel(model_path, 
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关句子：",
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        progress_bar = tqdm.tqdm(total=len(orig_data_list))
        for index, i in enumerate(orig_data_list):
            progress_bar.update(1)
            data_list.append(i)
            vector_list.append(model.encode(i).tolist())
        # vector_list = [np.array(i) for i in vector_list]
        vector_list = np.array(vector_list).astype('float32')
        return data_list, vector_list
    
    def build_faiss(self, vector_list):
        d = vector_list.shape[1]
        # index = faiss.IndexFlatL2(d)  # 使用 L2 距离
        index = faiss.IndexFlatIP(d)  # 点积
        index.add(vector_list)  # 添加向量到索引
        return index


    def run_embedding_cleaner(self, input_data_list):
        data_list, vector_list = self.sentence_bert_encode(input_data_list, self.sentence_bert_model_path)

        faiss_indexer = self.build_faiss(vector_list)
        
        processed = set()
        unique_texts = []
        progress_bar = tqdm.tqdm(total=len(vector_list))
        print('############ Beginging Run Embedding Cleaner ############'.center(term_width)) 
        for index, tmp in enumerate(vector_list):
            if index in processed:
                continue
            query_vector = np.array(tmp).reshape(1,-1).astype('float32')
            # 搜索最相似的5个
            distances, indices = faiss_indexer.search(query_vector, 2)
            distances, indices = distances.tolist()[0], indices.tolist()[0]
            
            # 删除自己
            self_index = indices.index(index)
            del distances[self_index]
            del indices[self_index]

            if max(distances) > threshold:
                processed.update(indices)

            unique_texts.append(data_list[index])
            progress_bar.update(1)
        return unique_texts

            
# 关键词去除
class Key_World_Cleaner:
    def __init__(self, ban_list) -> None:
        self.ban_list = ban_list
    def run_key_world_cleaner(self,input_data_list):
        rules_set = set(self.ban_list)
        
        unique_texts = [item for item in input_data_list if not any(rule in item for rule in rules_set)]
        return unique_texts

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    # 数据读进来
    input_data_list = [json.loads(i)['data'] for i in open(in_path, 'r', encoding='utf-8')]
    print(f'############ Loaded {len(input_data_list)} data ############'.center(term_width))

    threshold = 0.9
    rouge_cleaner = Rouge_l_Cleaner(threshold)
    unique_texts = rouge_cleaner.run_rouge_cleaner(input_data_list)

    # model_path = '/BAAI/bge-large-zh-v1.5'
    # threshold = 0.9
    # minhas_cleaner = MinHash_Cleaner(threshold, model_path)
    # unique_texts = minhas_cleaner.run_minhashLSH_cleaner(input_data_list)
    
    # model_path = '/BAAI/bge-large-zh-v1.5'
    # threshold = 0.9
    # embedding_cleaner = Embedding_Cleaner(threshold, model_path)
    # unique_texts = embedding_cleaner.run_embedding_cleaner(input_data_list)
    
    # ban_list = ['你好']
    # key_world_cleaner = Key_World_Cleaner(ban_list)
    # unique_texts = key_world_cleaner.run_key_world_cleaner(input_data_list)
    
    with open(out_path, 'w', encoding='utf-8') as file:
        for i in unique_texts:
            file.write(json.dumps({'data': i}, ensure_ascii=False) + '\n')
    print(f'############ Left {len(unique_texts)} data ############'.center(term_width))