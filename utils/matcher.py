import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Skills that ResumeX AI can currently recognize
SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "express",
    "spring boot",
    "django",
    "flask",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "artificial intelligence",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "tableau",
    "power bi",
    "excel",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "google cloud",
    "rest api",
    "api",
    "spring",
    "linux",
]


def normalize_text(text):
    """
    Convert text to lowercase and normalize spaces.
    """

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    """
    Extract known technical skills from text.
    """

    text = normalize_text(text)

    found_skills = []

    for skill in SKILLS:

        # Escape special characters such as +, # and .
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


def calculate_similarity(resume_text, job_description):
    """
    Calculate similarity between a resume and job description
    using TF-IDF and cosine similarity.
    """

    resume_text = normalize_text(resume_text)
    job_description = normalize_text(job_description)

    if not resume_text or not job_description:
        return 0.0

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


def compare_skills(resume_text, job_description):
    """
    Compare skills found in the resume and job description.
    """

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    matched_skills = sorted(
        resume_skills.intersection(job_skills)
    )

    missing_skills = sorted(
        job_skills.difference(resume_skills)
    )

    additional_skills = sorted(
        resume_skills.difference(job_skills)
    )

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills
    }


def analyze_match(resume_text, job_description):
    """
    Perform complete resume-job analysis.
    """

    similarity_score = calculate_similarity(
        resume_text,
        job_description
    )

    skill_comparison = compare_skills(
        resume_text,
        job_description
    )

    return {
        "match_score": similarity_score,
        **skill_comparison
    }