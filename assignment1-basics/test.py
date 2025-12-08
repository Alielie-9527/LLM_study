# 推荐方案
word_bytes = [bytes([b]) for b in "咖".encode("utf-8")]
# 直接得到 [b'\xe5', b'\x92', b'\x96']
print(tuple(word_bytes))