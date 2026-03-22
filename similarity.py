
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

t1 = input("Enter Text 1: ")
t2 = input("Enter Text 2: ")
score = check_similarity(t1, t2)
print(f"Similarity Score: {score}%")

if score > 70:
    print("Verdict: Highly Similar ✅")
elif score > 40:
    print("Verdict: Somewhat Similar 🔶")
else:
    print("Verdict: Not Similar ❌")
