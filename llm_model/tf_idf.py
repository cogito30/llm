from sklearn.feature_extraction.text import TfidfVectorizer

"""
tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
    input="content",       # 입력될 데이터의 형태(content: 문자열 데이터, 바이트 형태의 입력값, file: 파일 객체, filename: 파일 경로)
    encoding="utf-8",      # 사용할 텍스트 인코딩 값
    lowercase=True,        # 입력받은 데이터를 소문자로 변환할지 여부
    stop_words=None,       # 분석에 도움이 되지 않는 의미 없는 단어 제외
    ngram_range=(1, 1),    # N-gram의 범위 (min, max)
    max_df=1.0,            # 전체 문서 중 일정 횟수 이상 등장한 단어는 불용어로 처리. 1이하의 경우 비율을 초과하는 단어를 불용어 처리
    min_df=1,              # 전체 문서 중 일정 횟수 미만으로 등장한 단어를 불용어로 처리
    vocabulary=None,       # 미리 구축한 단어사전이 있다면 해당 단어 사전을 사용, 없을 경우 TF-IDF 학습 시 자동으로 구축
    smooth_idf=True,       # IDF 분모처리: 계산 시 분모에 1을 더한다
)
"""

corpus = [
    "That movie is famous movie",
    "I like that actor",
    "I don't like that actor"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)
tfidf_matrix = tfidf_vectorizer.transform(corpus)

print(tfidf_matrix.toarray())            # TF-IDF를 넘파이 배열로 변환. (문서 수) x (단어 수)의 형태. 행은 하나의 문서, 열을 단어를 의미
print(tfidf_vectorizer.vocabulary_)      # TF-IDF에 사용된 단어 사전을 의미. 키는 고유한 단어를 값, 값은 단어의 색인 값을 의미