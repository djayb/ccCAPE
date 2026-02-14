import unittest


class TestChunkTickers(unittest.TestCase):
    def test_chunks_by_max_chunk(self) -> None:
        from scripts.fetch_stooq_quotes import chunk_tickers

        tickers = [f"t{i}.us" for i in range(10)]
        chunks = chunk_tickers(tickers, max_chunk=3, max_url_len=10_000)

        self.assertEqual([len(c) for c in chunks], [3, 3, 3, 1])
        self.assertEqual([t for c in chunks for t in c], tickers)

    def test_chunks_by_url_len(self) -> None:
        from scripts.fetch_stooq_quotes import STOOQ_QUOTES_URL, chunk_tickers

        tickers = ["a" * 50 + ".us", "b" * 50 + ".us", "c" * 50 + ".us"]

        base_len = len(STOOQ_QUOTES_URL) + len("?s=&f=sd2c&h&e=csv")
        max_url_len = base_len + len(tickers[0])

        chunks = chunk_tickers(tickers, max_chunk=100, max_url_len=max_url_len)
        self.assertEqual(chunks, [[t] for t in tickers])

    def test_empty(self) -> None:
        from scripts.fetch_stooq_quotes import chunk_tickers

        self.assertEqual(chunk_tickers([], max_chunk=10, max_url_len=1000), [])


if __name__ == "__main__":
    unittest.main()

