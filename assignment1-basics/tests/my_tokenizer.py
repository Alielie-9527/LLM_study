import regex as re
from typing import Any, Dict, List, Optional, Tuple, Iterable

class Tokenizer:
    def __init__(self,
                 vocab : dict[int, bytes],
                 merges : list[tuple[bytes,bytes]],
                 special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or [] # 防止 None 会导致sorted报错
        self.word_to_idx ={v:k for k,v in self.vocab.items()}
        self.merges_index = {pair : i for i ,pair in enumerate(merges)} # 实际上如果顺序合并也是从高到低优先级



    def _merge(self,word_bytes):  # word_bytes [b'h', b'e', b'l', b'l', b'o']
        if not word_bytes:
            return []

        while True:
            pairs = self._get_pair(word_bytes)

            if not pairs:
                break

            #从pair中找到最适配的 , 如果pair不在则返回无穷大
            best_pair = min(pairs,key = lambda p:self.merges_index.get(p,float("inf")))

            if best_pair not in self.merges:
                break

            i = 0
            new_bytes =[]
            while i< len(word_bytes):
                if i <len(word_bytes)-1 and (word_bytes[i],word_bytes[i+1]) == best_pair:
                    new_bytes.append(word_bytes[i]+word_bytes[i+1])
                    i+=2
                else:
                    new_bytes.append(word_bytes[i])
                    i +=1
            word_bytes = new_bytes
        output_idx = word_bytes
        return output_idx




    def _get_pair(self,word_bytes):
        return set(zip(word_bytes,word_bytes[1:]))


    def encode(self,text: str) -> list[int]:
        if not text:
            return []

        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
         #处理方式
        tokens = []
        sorted_special_tokens = sorted(self.special_tokens,key =len,reverse= True) # 降序排列
        pattern = "|".join(map(re.escape,sorted_special_tokens))
        if pattern:
            line_list = re.split(f"({pattern})", text)
        else:
            line_list = [text] # 如果没有特殊字符 则直接是整体 因为如果进行分割，会导致完整的句子变成一个字符串

        for line in line_list:
            if not line:
                continue

            if sorted_special_tokens and line in sorted_special_tokens:
                tokens.append(self.word_to_idx[line.encode('utf-8')])
                continue
            word_list = re.findall(pat,line) # 单词列表

            for word in word_list: # "hello"
                word_bytes = word.encode("utf-8")
                word_bytes = [bytes([b]) for b in word_bytes] # [b'h', b'e', b'l', b'l', b'o']
                if tuple(word_bytes) in self.merges: # 如何这个词时merge中的一环
                    tokens.append(self.word_to_idx[word.encode('utf-8')])

                else:
                    output = self._merge(word_bytes)
                    output = [self.word_to_idx[b] for b in output]
                    tokens.extend(output)
        return tokens



    def encode_iterable(self,iterable:Iterable[str]):
        for token in iterable:
            yield from self.encode(token) #通过yield生成迭代器


    def decode(self,ids:list[int])->str:
        output = b"".join(self.vocab[i] for i in ids)
        return output.decode("utf-8", errors="replace")