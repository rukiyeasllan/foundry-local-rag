# Evaluation Results

The final strict functional evaluation was conducted using six predefined queries against the local document collection.

## Final Results
- Tests passed: 6/6
- Functional success rate on the defined test set: 100%
- Average end-to-end query time: 22.14 seconds
- Answerable queries: 4/4 passed
- Unsupported queries: 2/2 correctly returned the fallback response

## Test Cases
1. Foundry Local ne ise yarar? - PASS
2. RAG sistemi nasil calisir? - PASS
3. Embedding nedir? - PASS
4. SQLite bu projede ne icin kullaniliyor? - PASS
5. Turkiye Cumhuriyetinin baskenti neresidir? - PASS (fallback)
6. Dunyanin en yuksek dagi hangisidir? - PASS (fallback)

## Notes
These results represent functional testing on a small local knowledge base containing two indexed chunks. The 100% result should not be interpreted as general RAG accuracy.

The system uses a minimum cosine similarity threshold of 0.35. Queries below this threshold are rejected without invoking answer generation.

Keeping the embedding and chat models loaded in memory reduced the observed average query time from 62.94 seconds to 22.14 seconds.
