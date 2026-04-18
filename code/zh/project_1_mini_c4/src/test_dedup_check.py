import hashlib

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None
    MinHashLSH = None


data_a = "Deep learning is a subset of machine learning using neural networks."
data_b = "Deep learning is a subset of machine learning using neural networks."
data_c = "Cooking pasta requires boiling water and adding salt."


def fallback_check():
    sha_a = hashlib.sha1(data_a.encode("utf-8")).hexdigest()
    sha_b = hashlib.sha1(data_b.encode("utf-8")).hexdigest()
    sha_c = hashlib.sha1(data_c.encode("utf-8")).hexdigest()

    assert sha_a == sha_b, "Identical texts should produce the same digest"
    assert sha_a != sha_c, "Different texts should produce different digests"

    print("datasketch unavailable; fallback exact-duplicate sanity check passed.")
    print({"same_text_same_hash": True, "different_text_different_hash": True})


def minhash_check():
    def get_minhash(text):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode("utf8"))
        return m

    mh_a = get_minhash(data_a)
    mh_b = get_minhash(data_b)
    mh_c = get_minhash(data_c)

    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    lsh.insert("doc_a", mh_a)

    result_b = lsh.query(mh_b)
    result_c = lsh.query(mh_c)

    assert result_b == ["doc_a"], f"Expected duplicate match for B, got: {result_b}"
    assert result_c == [], f"Expected no duplicate match for C, got: {result_c}"

    print(f"查询 B (应该重复): {result_b}")
    print(f"查询 C (不该重复): {result_c}")


if __name__ == "__main__":
    if MinHash is None or MinHashLSH is None:
        fallback_check()
    else:
        minhash_check()
